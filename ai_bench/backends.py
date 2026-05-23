"""Backend classes for Ollama, LM Studio, and oMLX.

Extracted from bench.py for modular access to model-serving backends.
"""

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from ai_bench.log import log, warn

IS_MAC = platform.system() == "Darwin"

OMLX_BIN = Path.home() / ".local" / "share" / "omlx-venv" / "bin" / "omlx"
OMLX_MODEL_DIR = Path.home() / ".omlx" / "models"

OLLAMA_MANIFESTS = (
    Path.home() / ".ollama" / "models" / "manifests"
    / "registry.ollama.ai" / "library"
)


def which(cmd):
    return shutil.which(cmd)


def run(cmd, *, check=True, capture=False, env=None, timeout=None):
    shell = isinstance(cmd, str)
    if capture:
        r = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, env=env, timeout=timeout
        )
        if check and r.returncode != 0:
            raise RuntimeError(f"{cmd} failed: {r.stderr}")
        return r.returncode, r.stdout, r.stderr
    r = subprocess.run(cmd, shell=shell, env=env, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd} failed (rc={r.returncode})")
    return r.returncode, "", ""


class Ollama:
    proc = None

    @staticmethod
    def is_up():
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False

    @classmethod
    def start(cls):
        if cls.is_up():
            return
        log("Starting `ollama serve`…")
        cls.proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            if cls.is_up():
                return
            time.sleep(0.5)
        raise RuntimeError("ollama failed to start")

    @classmethod
    def stop(cls):
        if cls.proc:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
            cls.proc = None
        # Force ollama to unload all models from GPU memory so the next backend
        # (e.g. oMLX) has room. Ollama pins models and won't free them until
        # the keep-alive expires (default 5 min). Setting it to 0 forces immediate unload.
        run(["pkill", "-f", "ollama.serve"], check=False)
        time.sleep(2)


class LMStudio:
    @staticmethod
    def is_up():
        try:
            urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2)
            return True
        except Exception:
            return False

    SETTINGS_PATH = Path.home() / ".lmstudio" / "settings.json"
    SETTINGS_BAK  = SETTINGS_PATH.with_suffix(".json.bench-bak")

    @classmethod
    def relax_guardrails(cls):
        """Disable LM Studio's model-load resource guardrail for this run.

        The default guardrail over-estimates MoE / large-context models' memory
        needs (e.g. claims ~58 GiB for a 44 GB MLX-4bit Qwen MoE on a 64 GB
        machine where it actually runs fine). Backs up the original settings so
        cls.restore_guardrails() can put them back at end-of-run.
        """
        if not cls.SETTINGS_PATH.exists():
            return
        try:
            settings = json.loads(cls.SETTINGS_PATH.read_text())
        except Exception:
            return
        g = settings.get("modelLoadingGuardrails") or {}
        if g.get("mode") == "off" and g.get("alwaysAllowLoadAnyway") is True:
            return  # already relaxed
        if not cls.SETTINGS_BAK.exists():
            shutil.copy(cls.SETTINGS_PATH, cls.SETTINGS_BAK)
        g["mode"] = "off"
        g["alwaysAllowLoadAnyway"] = True
        settings["modelLoadingGuardrails"] = g
        cls.SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        log("Relaxed LM Studio model-load guardrails (will restore at end of run).")

    @classmethod
    def restore_guardrails(cls):
        if cls.SETTINGS_BAK.exists():
            shutil.move(str(cls.SETTINGS_BAK), str(cls.SETTINGS_PATH))

    @classmethod
    def _ensure_daemon(cls):
        """LM Studio app must launch at least once to unpack its daemon.

        `lms server start` silently fails if ~/.lmstudio/.internal/utils/ doesn't
        exist yet.  If it's missing, open the app in the background, wait up to
        30 s for the daemon directory to appear, then quit the GUI.
        """
        daemon_dir = Path.home() / ".lmstudio" / ".internal" / "utils"
        if daemon_dir.exists():
            return
        if not IS_MAC:
            return
        log("LM Studio daemon not initialised — launching app once to finish first-time setup…")
        # Use the direct path — `open -a` relies on LaunchServices which may not
        # have indexed the app yet immediately after a fresh brew cask install.
        app_path = Path("/Applications/LM Studio.app")
        if app_path.exists():
            subprocess.Popen(["open", str(app_path)])
        else:
            subprocess.Popen(["open", "-a", "LM Studio"])
        for _ in range(60):  # 30 s
            if daemon_dir.exists():
                break
            time.sleep(0.5)
        else:
            warn(f"LM Studio daemon dir ({daemon_dir}) didn't appear after 30 s — `lms bootstrap` may fail.")
        time.sleep(2)
        run(["osascript", "-e", 'quit app "LM Studio"'], check=False)
        time.sleep(2)

    @classmethod
    def start(cls):
        if cls.is_up():
            return
        if not which("lms"):
            warn("`lms` CLI not found — start LM Studio server manually if you want LM Studio combos.")
            return
        cls._ensure_daemon()
        log("Starting LM Studio server…")
        run(["lms", "server", "start"], check=False)
        for _ in range(40):
            if cls.is_up():
                return
            time.sleep(0.5)
        warn("LM Studio server didn't come up in time.")

    @classmethod
    def stop(cls):
        if not which("lms"):
            return
        log("Stopping LM Studio server…")
        run(["lms", "server", "stop"], check=False)
        # lms server stop only drops the API endpoint; worker processes keep
        # model weights in RAM. Kill them explicitly so memory is freed before
        # the next backend starts.
        run(["pkill", "-f", r"\.lmstudio/.internal/utils/node"], check=False)
        time.sleep(2)


class OMLX:
    proc = None

    @staticmethod
    def get_api_key():
        """Get oMLX API key from config file or environment variable."""
        if os.environ.get("OMLX_API_KEY"):
            return os.environ["OMLX_API_KEY"]
        settings_path = Path.home() / ".omlx" / "settings.json"
        try:
            settings = json.loads(settings_path.read_text())
            return settings.get("auth", {}).get("api_key", "")
        except Exception:
            return ""

    @staticmethod
    def is_up():
        try:
            # Check with a simple request to /health to avoid auth issues
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            return True
        except Exception:
            return False

    @classmethod
    def check_model_load(cls, model_alias, *, timeout=30):
        """Ask oMLX for one token so memory/model-load failures surface early."""
        payload = json.dumps({
            "model": model_alias,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
            "stream": False,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = cls.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            "http://localhost:8000/v1/chat/completions",
            data=payload,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response.read()
            return True, ""
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""
            message = body or str(e)
            return False, f"oMLX preflight failed for {model_alias}: HTTP {e.code} {message}"
        except Exception as e:
            return False, f"oMLX preflight failed for {model_alias}: {e}"

    @classmethod
    def start(cls):
        # Restart any running server so stale pinned models from a previous
        # session do not starve the benchmarked model.
        if cls.is_up():
            cls.stop()
            time.sleep(1)
        omlx_bin = str(OMLX_BIN) if OMLX_BIN.exists() else which("omlx")
        if not omlx_bin:
            warn("`omlx` CLI not found — start oMLX server manually if you want oMLX combos.")
            return
        log("Starting oMLX server…")
        OMLX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.proc = subprocess.Popen(
            [omlx_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(120):
            if cls.is_up():
                return
            time.sleep(0.5)
        warn("oMLX server didn't come up in time (60s).")

    @classmethod
    def stop(cls):
        if cls.proc:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
            cls.proc = None
        run(["pkill", "-f", "omlx serve"], check=False)
        time.sleep(2)
