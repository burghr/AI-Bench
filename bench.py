#!/usr/bin/env python3
"""
agent-bench: benchmark coding agents (pi, opencode) against local LLM
backends (LM Studio, Ollama) end-to-end.

Default flow:
  1. Pre-flight: record what's already installed.
  2. Install anything missing (pi, opencode, ollama, lm-studio on Mac).
  3. Start backends, ensure model is present.
  4. Configure agents to point at each backend.
  5. For each combo (pi+lmstudio, pi+ollama, opencode+lmstudio, opencode+ollama):
     - run a warmup, then N timed iterations
     - capture wall time, TTFT, output, est tokens/sec
     - save raw model output to results/<ts>/<combo>__iter-N.txt
  6. Write results/<ts>/results.json
  7. Prompt to clean up (only removes things we installed).

Usage:
  python3 bench.py [--config bench.config.json] [--skip-install]
                   [--cleanup-only]
"""

import argparse
import json
import os
import platform
import select
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).parent.resolve()
STATE_FILE = ROOT / ".bench-state.json"

# Load .env from the repo root so HF_TOKEN etc. are available without
# requiring the user to export them in their shell.
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
# Results live under the user's home so re-cloning the repo doesn't clobber
# them (and so you can collect runs from multiple checkouts in one place).
# Override with $AGENT_BENCH_RESULTS_DIR.
RESULTS_DIR = Path(os.environ.get("AGENT_BENCH_RESULTS_DIR")
                    or Path.home() / "ai-bench" / "results")
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_ARM64 = platform.machine() in ("arm64", "aarch64")
# LM Studio on Mac requires Apple Silicon. On Linux it's AppImage-only.
LMSTUDIO_SUPPORTED = IS_MAC and IS_ARM64

# oMLX requires Apple Silicon
OMLX_SUPPORTED = IS_MAC and IS_ARM64
OMLX_CLONE_DIR = Path.home() / ".local" / "share" / "omlx-src"
OMLX_VENV_DIR  = Path.home() / ".local" / "share" / "omlx-venv"
OMLX_BIN       = OMLX_VENV_DIR / "bin" / "omlx"
OMLX_MODEL_DIR = Path.home() / ".omlx" / "models"


def log(msg, prefix="•"):
    print(f"{prefix} {msg}", flush=True)


def warn(msg):
    log(msg, prefix="!")


def err(msg):
    log(msg, prefix="x")


def which(cmd):
    return shutil.which(cmd)


class Ticker:
    """Background ticker that overwrites a single stderr line with elapsed time.

    Use as a context manager around a long-running call:
        with Ticker("    iter-0"):
            ...do work...

    Cleanly clears its line on exit, so subsequent log() output prints normally.
    Disabled (becomes a no-op) if stderr isn't a TTY — keeps log files clean.
    """
    INTERVAL = 1.0

    def __init__(self, prefix):
        self.prefix = prefix
        self._stop = threading.Event()
        self._thread = None
        self._enabled = sys.stderr.isatty()

    def __enter__(self):
        if not self._enabled:
            return self
        self._start = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if not self._enabled:
            return False
        self._stop.set()
        self._thread.join(timeout=2)
        # Clear the line so the next log() print doesn't collide.
        sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()
        return False

    def _run(self):
        while not self._stop.is_set():
            elapsed = int(time.monotonic() - self._start)
            sys.stderr.write(f"\r{self.prefix} ⏱ {elapsed}s")
            sys.stderr.flush()
            self._stop.wait(self.INTERVAL)


def cpu_info():
    """Best-effort human-readable CPU brand string.

    On macOS: 'Apple M2 Pro', 'Apple M1 Max', 'Intel(R) Core(TM) i9-9880H ...'.
    On Linux: the 'Model name' line from /proc/cpuinfo.
    Returns None if it can't determine.
    """
    try:
        if IS_MAC:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        if IS_LINUX:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


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


# ---- state -----------------------------------------------------------------
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "installed_by_us": {},
        "model_pulled_by_us": {"ollama": [], "lmstudio": []},
    }


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2))


# ---- detection -------------------------------------------------------------
def have_pi():
    return bool(which("pi"))


def have_opencode():
    return bool(which("opencode"))


def have_ollama():
    return bool(which("ollama"))


def have_lmstudio():
    if not LMSTUDIO_SUPPORTED:
        return False
    # The app must be present — lms CLI alone can't start the daemon without it.
    if IS_MAC and not Path("/Applications/LM Studio.app").exists():
        return False
    return bool(which("lms"))


def have_omlx():
    """Check if oMLX is available (supports OpenAI-compatible API)."""
    if not OMLX_SUPPORTED:
        return False
    if which("omlx") or OMLX_BIN.exists():
        return True
    # Check if oMLX server is already running
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
        return True
    except Exception:
        return False


def have_ollama_model(name):
    # Read manifest directly so this works pre-flight (before the server is up)
    # and matches what _list_ollama_installed() reports.
    manifests_root = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    if ":" in name:
        family, tag = name.split(":", 1)
    else:
        family, tag = name, "latest"
    return (manifests_root / family / tag).exists()


def _lmstudio_dir_complete(d):
    """Return True only if a model dir is FULLY downloaded — no .part files
    and at least one .safetensors / .gguf weight file present."""
    if not d.is_dir():
        return False
    files = list(d.iterdir())
    if not files:
        return False
    if any(f.name.startswith("downloading_") or f.name.endswith(".part") for f in files):
        return False
    has_weights = any(
        f.suffix in (".safetensors", ".gguf") and not f.name.startswith("downloading_")
        for f in files
    )
    return has_weights


def have_lmstudio_model(name):
    # Filesystem check is the ground truth: LM Studio's `lms ls` renames models
    # (strips publisher + quant suffix) so substring matching there is unreliable.
    # The download lives under ~/.lmstudio/models/<hf_path>/ regardless. We
    # treat partial downloads (any `.part` file present) as NOT present, so the
    # bench doesn't try to load a half-finished model.
    if not name:
        return False
    path = name
    if path.startswith("http"):
        path = path.rstrip("/").split("huggingface.co/")[-1]
    lms_root = Path.home() / ".lmstudio" / "models"
    candidate = lms_root / path
    if _lmstudio_dir_complete(candidate):
        return True
    if lms_root.exists():
        target = path.split("/")[-1].lower()
        for d in lms_root.rglob("*"):
            if d.is_dir() and d.name.lower() == target and _lmstudio_dir_complete(d):
                return True
    return False


def _lmstudio_resolve_api_id(hf_path):
    """Return LM Studio's API model ID for an HF repo path, or None.

    LM Studio assigns a shorter alias (e.g. `qwen3-coder-next-mlx` for
    `lmstudio-community/Qwen3-Coder-Next-MLX-4bit`). Both `lms load` and the
    OpenAI-compatible /v1/* endpoints want that alias, not the HF path.
    """
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5) as r:
            data = json.loads(r.read())
        ids = [m.get("id", "") for m in (data.get("data") or []) if m.get("id")]
    except Exception:
        return None
    if not ids:
        return None
    target = hf_path.split("/")[-1].lower()
    for api_id in ids:
        if api_id.lower() == target:
            return api_id
    for api_id in ids:
        api_lower = api_id.lower()
        if target.startswith(api_lower) or api_lower.startswith(target):
            return api_id
    for api_id in ids:
        if api_id.lower() in target or target in api_id.lower():
            return api_id
    return None


def have_omlx_model(name):
    """Check if oMLX model is fully downloaded (has weight files on disk)."""
    if not have_omlx():
        return False
    model_path = OMLX_MODEL_DIR / name
    if not model_path.exists():
        return False
    return any(model_path.glob("*.safetensors")) or any(model_path.glob("*.gguf"))


def _hf_get_chat_template(hf_repo, headers):
    """Fetch chat_template string from a HuggingFace repo's tokenizer_config.json, or None."""
    try:
        url = f"https://huggingface.co/{hf_repo}/resolve/main/tokenizer_config.json"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get("chat_template")
    except Exception:
        return None


def _hf_get_base_model(hf_repo, headers):
    """Return the base_model field from a HuggingFace model card, or None."""
    try:
        url = f"https://huggingface.co/api/models/{hf_repo}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            card = json.loads(r.read()).get("cardData") or {}
        base = card.get("base_model")
        # base_model can be a string or a list
        return (base[0] if isinstance(base, list) else base) or None
    except Exception:
        return None


def _patch_omlx_chat_template(model_dir, hf_repo):
    """Ensure tokenizer_config.json has chat_template.

    Tries hf_repo first; if its tokenizer_config also lacks the template (common
    for third-party quantisations like Outlier-Ai), falls back to the base_model
    repo declared in the HuggingFace model card.
    """
    tc_path = model_dir / "tokenizer_config.json"
    if not tc_path.exists():
        return
    try:
        tc = json.loads(tc_path.read_text())
        if tc.get("chat_template"):
            return
        token = os.environ.get("HF_TOKEN", "")
        headers = {"User-Agent": "agent-bench/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        log(f"chat_template missing — fetching from {hf_repo}…")
        chat_template = _hf_get_chat_template(hf_repo, headers)

        if not chat_template:
            base_repo = _hf_get_base_model(hf_repo, headers)
            if base_repo:
                log(f"Not found in {hf_repo} — trying base model {base_repo}…")
                chat_template = _hf_get_chat_template(base_repo, headers)

        if not chat_template:
            warn(f"Could not find chat_template for {hf_repo} — oMLX will return 400 on chat requests.")
            return

        tc["chat_template"] = chat_template
        tc_path.write_text(json.dumps(tc, indent=2))
        log("chat_template patched OK.")
    except Exception as e:
        warn(f"Could not patch chat_template: {e}")


def download_omlx_model(name, hf_repo=None):
    """Download an MLX model from HuggingFace into OMLX_MODEL_DIR.

    hf_repo defaults to mlx-community/<name> — the standard convention for
    pre-quantised MLX models on HuggingFace.
    """
    repo = hf_repo or f"mlx-community/{name}"
    dest = OMLX_MODEL_DIR / name
    if dest.exists():
        _patch_omlx_chat_template(dest, repo)
        return
    OMLX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading oMLX model '{repo}' → {dest} …")
    venv_python = OMLX_VENV_DIR / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    run([
        python, "-c",
        f"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo!r}, local_dir={str(dest)!r})",
    ])
    _patch_omlx_chat_template(dest, repo)


# ---- install ---------------------------------------------------------------
def install_ollama(state):
    if have_ollama():
        return
    log("Installing Ollama…")
    if IS_MAC and which("brew"):
        run("brew install ollama")
    else:
        run("curl -fsSL https://ollama.com/install.sh | sh")
    state["installed_by_us"]["ollama"] = True


def install_lmstudio(state):
    if have_lmstudio():
        return
    if not LMSTUDIO_SUPPORTED:
        if IS_MAC and not IS_ARM64:
            warn("LM Studio requires Apple Silicon — skipping on Intel Mac.")
        else:
            warn("LM Studio install on Linux not automated — skipping LM Studio combos.")
        return
    if not which("brew"):
        warn("Homebrew required to auto-install LM Studio — skipping.")
        return
    log("Installing LM Studio…")
    run("brew install --cask lm-studio")
    # Open the app once so it can finish first-time setup (creates ~/.lmstudio/).
    # _ensure_daemon() is a no-op if already initialised.
    LMStudio._ensure_daemon()
    # Bootstrap the lms CLI so it's on PATH.
    bundled = Path("/Applications/LM Studio.app/Contents/Resources/app/.webpack/lms")
    user_lms = Path.home() / ".lmstudio" / "bin" / "lms"
    if bundled.exists():
        run([str(bundled), "bootstrap"], check=False)
    # If bootstrap still didn't create the user binary, symlink the bundled one.
    if not user_lms.exists() and bundled.exists():
        user_lms.parent.mkdir(parents=True, exist_ok=True)
        user_lms.symlink_to(bundled)
    # Make lms available to the rest of this process without a shell restart.
    lmstudio_bin = Path.home() / ".lmstudio" / "bin"
    if lmstudio_bin.exists():
        os.environ["PATH"] = str(lmstudio_bin) + os.pathsep + os.environ.get("PATH", "")
    state["installed_by_us"]["lmstudio"] = True


def install_omlx(state):
    if have_omlx():
        return
    if not OMLX_SUPPORTED:
        warn("oMLX requires Apple Silicon — skipping.")
        return
    log("Installing oMLX from source (https://github.com/jundot/omlx)…")
    OMLX_CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not OMLX_CLONE_DIR.exists():
        run(["git", "clone", "https://github.com/jundot/omlx", str(OMLX_CLONE_DIR)])
    if not OMLX_VENV_DIR.exists():
        # oMLX depends on mlx>=0.31.2 which brew installs for Python 3.14.
        # Use brew's python3.14 explicitly so the venv has access to it.
        python = (which("python3.14")
                  or "/opt/homebrew/opt/python@3.14/bin/python3.14"
                  or sys.executable)
        run([python, "-m", "venv", "--system-site-packages", str(OMLX_VENV_DIR)])
    venv_pip = OMLX_VENV_DIR / "bin" / "pip"
    run([str(venv_pip), "install", "--quiet", str(OMLX_CLONE_DIR)])
    state["installed_by_us"]["omlx"] = True


def install_fzf(state):
    if which("fzf"):
        return
    if IS_MAC and which("brew"):
        log("Installing fzf (for live model search)…")
        run("brew install fzf")
        state["installed_by_us"]["fzf"] = True
    elif IS_LINUX and which("apt-get"):
        log("Installing fzf (for live model search)…")
        run("sudo apt-get install -y fzf")
        state["installed_by_us"]["fzf"] = True
    else:
        warn("fzf not found — model picker will use numbered list instead. Install fzf for live filtering.")


def install_pi(state):
    if have_pi():
        return
    log("Installing pi…")
    if which("npm"):
        run("npm install -g @mariozechner/pi-coding-agent")
    else:
        run("curl -fsSL https://pi.dev/install.sh | sh")
    state["installed_by_us"]["pi"] = True


def install_opencode(state):
    if have_opencode():
        return
    log("Installing opencode…")
    run("curl -fsSL https://opencode.ai/install | bash")
    state["installed_by_us"]["opencode"] = True


# ---- backends --------------------------------------------------------------
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
    def start(cls):
        # If we already own a running server, stop it so it restarts pointing at
        # the current model-dir (a stale server may have started before models
        # downloaded). If something *else* holds the port, trust it — we can't
        # safely kill it, and trying to start a duplicate would fail to bind.
        if cls.is_up():
            if cls.proc is None:
                log("oMLX server already running (not started by us) — using existing instance.")
                return
            cls.stop()
            time.sleep(1)
        omlx_bin = str(OMLX_BIN) if OMLX_BIN.exists() else which("omlx")
        if not omlx_bin:
            warn("`omlx` CLI not found — start oMLX server manually if you want oMLX combos.")
            return
        log("Starting oMLX server…")
        OMLX_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.proc = subprocess.Popen(
            [omlx_bin, "serve", "--model-dir", str(OMLX_MODEL_DIR)],
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


# ---- agent invocation ------------------------------------------------------
def run_agent_streamed(cmd, env, *, total_timeout=900):
    """Spawn agent, stream stdout, time first→last byte and total wall.

    Stderr is drained on a background thread so it can't deadlock when the
    agent writes more than the pipe buffer (~64 KB on macOS) before we
    finish reading stdout. Stdout is byte-by-byte until we see the first
    non-whitespace character (to pin TTFT precisely) then chunked, which
    avoids one syscall per byte contaminating wall-time on chatty agents.
    """
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=0,
    )
    start = time.monotonic()
    first_byte_t = None
    last_byte_t = None
    chunks = []

    stderr_chunks = []
    def _drain_stderr():
        try:
            while True:
                chunk = p.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)
        except Exception:
            pass
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    try:
        while True:
            now = time.monotonic()
            if now - start > total_timeout:
                p.kill()
                break
            rlist, _, _ = select.select([p.stdout], [], [], 1.0)
            if not rlist:
                if p.poll() is not None:
                    break
                continue
            if first_byte_t is None:
                ch = p.stdout.read(1)
                if not ch:
                    break
                now = time.monotonic()
                if ch.strip():
                    first_byte_t = now
                last_byte_t = now
                chunks.append(ch)
            else:
                chunk = p.stdout.read(4096)
                if not chunk:
                    break
                last_byte_t = time.monotonic()
                chunks.append(chunk)
    finally:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
        stderr_thread.join(timeout=5)
    end = time.monotonic()
    out = b"".join(chunks).decode("utf-8", errors="replace")
    err_out = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    return {
        "rc": p.returncode,
        "wall_s": end - start,
        "ttft_s": (first_byte_t - start) if first_byte_t else None,
        "last_byte_s": (last_byte_t - start) if last_byte_t else None,
        "stdout": out,
        "stderr": err_out,
        "end_reason": "exit" if p.returncode is not None else "killed",
    }


def estimate_tokens(text):
    if not text:
        return 0
    return max(1, round(len(text) / 4))




# ---- agent configs ---------------------------------------------------------
OPENCODE_CFG = Path.home() / ".config/opencode/opencode.json"
OPENCODE_CFG_BAK = OPENCODE_CFG.with_suffix(".json.bench-bak")
OPENCODE_AGENT_DIR = Path.home() / ".config/opencode/agent"
OPENCODE_NOTOOLS_AGENT = OPENCODE_AGENT_DIR / "notools.md"
OPENCODE_NOTOOLS_BAK = OPENCODE_AGENT_DIR / "notools.md.bench-bak"

PI_MODELS_CFG = Path.home() / ".pi" / "agent" / "models.json"
PI_MODELS_BAK = PI_MODELS_CFG.with_suffix(".json.bench-bak")

NOTOOLS_AGENT_CONTENT = """---
description: Bench harness — answer the prompt as plain text, no tools.
mode: primary
tools:
  "*": false
permission:
  edit: deny
  bash: deny
  webfetch: deny
---
Answer the user's prompt directly as plain text. Do not call any tools. Do not ask clarifying questions.
"""


def write_pi_models_config(ollama_models, lmstudio_models, omlx_models):
    """Write ~/.pi/agent/models.json so pi can reach local backends."""
    PI_MODELS_CFG.parent.mkdir(parents=True, exist_ok=True)
    if PI_MODELS_CFG.exists() and not PI_MODELS_BAK.exists():
        shutil.copy(PI_MODELS_CFG, PI_MODELS_BAK)
    providers = {}
    if ollama_models:
        providers["ollama"] = {
            "baseUrl": "http://localhost:11434/v1",
            "api": "openai-completions",
            "apiKey": "ollama",
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [{"id": m} for m in ollama_models],
        }
    if lmstudio_models:
        providers["lmstudio"] = {
            "baseUrl": "http://127.0.0.1:1234/v1",
            "api": "openai-completions",
            "apiKey": "lmstudio",
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [{"id": m} for m in lmstudio_models],
        }
    if omlx_models:
        omlx_key = OMLX.get_api_key() or "omlx"
        providers["omlx"] = {
            "baseUrl": "http://localhost:8000/v1",
            "api": "openai-completions",
            "apiKey": omlx_key,
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [{"id": m} for m in omlx_models],
        }
    PI_MODELS_CFG.write_text(json.dumps({"providers": providers}, indent=2))


def restore_pi_models_config():
    if PI_MODELS_BAK.exists():
        shutil.move(str(PI_MODELS_BAK), str(PI_MODELS_CFG))
    elif PI_MODELS_CFG.exists():
        PI_MODELS_CFG.unlink()


def write_opencode_notools_agent():
    OPENCODE_AGENT_DIR.mkdir(parents=True, exist_ok=True)
    if OPENCODE_NOTOOLS_AGENT.exists() and not OPENCODE_NOTOOLS_BAK.exists():
        shutil.copy(OPENCODE_NOTOOLS_AGENT, OPENCODE_NOTOOLS_BAK)
    OPENCODE_NOTOOLS_AGENT.write_text(NOTOOLS_AGENT_CONTENT)


def restore_opencode_notools_agent():
    if OPENCODE_NOTOOLS_BAK.exists():
        shutil.move(str(OPENCODE_NOTOOLS_BAK), str(OPENCODE_NOTOOLS_AGENT))
    elif OPENCODE_NOTOOLS_AGENT.exists():
        OPENCODE_NOTOOLS_AGENT.unlink()


def write_opencode_config(ollama_models, lmstudio_models, omlx_models):
    """Write opencode provider config registering all model aliases per backend."""
    OPENCODE_CFG.parent.mkdir(parents=True, exist_ok=True)
    if OPENCODE_CFG.exists() and not OPENCODE_CFG_BAK.exists():
        shutil.copy(OPENCODE_CFG, OPENCODE_CFG_BAK)
    providers = {}
    if ollama_models:
        providers["ollama"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Ollama",
            "options": {"baseURL": "http://localhost:11434/v1"},
            "models": {m: {"name": m} for m in ollama_models},
        }
    if lmstudio_models:
        providers["lmstudio"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "LM Studio",
            "options": {"baseURL": "http://127.0.0.1:1234/v1"},
            "models": {m: {"name": m} for m in lmstudio_models},
        }
    if omlx_models:
        providers["omlx"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "oMLX",
            "options": {"baseURL": "http://localhost:8000/v1"},
            "models": {m: {"name": m} for m in omlx_models},
        }
    cfg = {"$schema": "https://opencode.ai/config.json", "provider": providers}
    OPENCODE_CFG.write_text(json.dumps(cfg, indent=2))


# ---- agent registry --------------------------------------------------------
# Each agent declares: which backends it supports, and how to build the command
# given (model_alias, backend, prompt). Returns (cmd_list, extra_env_dict).
def _pi_cmd(model_alias, backend, prompt):
    # pi has built-in providers for ollama, lmstudio, and omlx
    # For omlx we use the built-in provider (not openai/<model>)
    if backend == "ollama":
        return (
            ["pi", "-nt", "-p", "--model", f"ollama/{model_alias}", prompt],
            {},
        )
    elif backend == "lmstudio":
        # Use pi's custom lmstudio provider (registered in ~/.pi/agent/models.json)
        # rather than the openai provider — pi ignores OPENAI_BASE_URL overrides.
        return (
            ["pi", "-nt", "-p", "--model", f"lmstudio/{model_alias}", prompt],
            {},
        )
    elif backend == "omlx":
        # pi has a built-in omlx provider for local oMLX server
        return (
            ["pi", "-nt", "-p", "--model", f"omlx/{model_alias}", prompt],
            {},
        )
    raise ValueError(f"pi: unsupported backend {backend}")


def _opencode_cmd(model_alias, backend, prompt):
    # `--agent notools` disables all tool exposure — small local models otherwise
    # get confused by tool schemas and respond with greetings or schema errors
    # instead of executing the task.
    return (
        ["opencode", "run", "--pure", "--agent", "notools",
         "-m", f"{backend}/{model_alias}", prompt],
        {},
    )


AGENTS = {
    "pi": {
        "supports_backends": ["ollama", "lmstudio", "omlx"],
        "build_cmd": _pi_cmd,
        "is_installed": lambda: bool(which("pi")),
        "desc": "pi — lightweight agentic CLI; fast, minimal overhead",
    },
    "opencode": {
        "supports_backends": ["ollama", "lmstudio", "omlx"],
        "build_cmd": _opencode_cmd,
        "is_installed": lambda: bool(which("opencode")),
        "desc": "opencode — full coding agent (file reads, edits, shell); higher overhead",
    },
    # `direct` is a pseudo-agent that hits the backend's HTTP API directly,
    # bypassing any CLI wrapper. Lets us measure the *real* model generation
    # rate (in tokens/sec, from the backend's own counters) so you can tell
    # how much of the agent runs is overhead vs raw model speed.
    "direct": {
        "supports_backends": ["ollama"],   # lmstudio direct mode is feasible but TODO
        "build_cmd": None,                  # special-cased — no subprocess
        "is_installed": lambda: True,
        "direct": True,
        "desc": "direct — raw HTTP to backend API; no agent wrapper, measures pure model speed (ollama only)",
    },
}


def run_backend_direct_ollama(model_alias, prompt, *, total_timeout=900):
    """Hit ollama's /api/generate (non-streaming) and capture precise stats."""
    body = json.dumps({"model": model_alias, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=total_timeout) as r:
            payload = r.read()
        end = time.monotonic()
        j = json.loads(payload.decode("utf-8"))
        out = j.get("response", "") or ""
        # All durations are in nanoseconds.
        load_s        = (j.get("load_duration")        or 0) / 1e9
        prompt_eval_s = (j.get("prompt_eval_duration") or 0) / 1e9
        eval_s        = (j.get("eval_duration")        or 0) / 1e9
        prompt_count  = j.get("prompt_eval_count") or 0
        eval_count    = j.get("eval_count") or 0
        return {
            "rc": 0,
            "wall_s": end - start,
            "ttft_s": None,
            "last_byte_s": None,
            "stdout": out,
            "stderr": "",
            "end_reason": "exit",
            "backend_stats": {
                "load_s": round(load_s, 4),
                "prompt_eval_count": prompt_count,
                "prompt_eval_s": round(prompt_eval_s, 4),
                "prompt_tok_per_s": round(prompt_count / prompt_eval_s, 2) if prompt_eval_s > 0 else None,
                "eval_count": eval_count,
                "eval_s": round(eval_s, 4),
                "eval_tok_per_s": round(eval_count / eval_s, 2) if eval_s > 0 else None,
            },
        }
    except Exception as e:
        end = time.monotonic()
        return {
            "rc": 1,
            "wall_s": end - start,
            "ttft_s": None,
            "last_byte_s": None,
            "stdout": "",
            "stderr": str(e),
            "end_reason": "exit",
            "backend_stats": None,
        }


def restore_opencode_config():
    if OPENCODE_CFG_BAK.exists():
        shutil.move(str(OPENCODE_CFG_BAK), str(OPENCODE_CFG))
    elif OPENCODE_CFG.exists():
        OPENCODE_CFG.unlink()


# ---- benchmark -------------------------------------------------------------
def bench_one(label, cmd, env, run_dir, n_iter, warmup, progress=None,
              direct=None):
    """Run one combination N+warmup times and summarize.

    If `direct` is set, it should be a dict {backend, model_alias, prompt}
    and we'll bypass the subprocess path and hit the backend's HTTP API
    directly to capture precise model stats.
    """
    log(f"  ▶ {label}  (warmup={warmup}, iters={n_iter})")
    iters = []
    for i in range(warmup + n_iter):
        is_warmup = i < warmup
        tag = f"warmup-{i}" if is_warmup else f"iter-{i - warmup}"
        if progress is not None:
            done, total = progress["done"], progress["total"]
            elapsed = time.monotonic() - progress["start"]
            rate = elapsed / done if done else 0
            eta = rate * (total - done) if rate else 0
            mins = int(eta // 60)
            log(f"    [{done + 1}/{total}] starting {label} {tag}"
                f" — elapsed {int(elapsed)}s"
                + (f", est remaining ~{mins}m" if rate else ""))
        with Ticker(f"      {tag}"):
            if direct:
                if direct["backend"] == "ollama":
                    result = run_backend_direct_ollama(direct["model_alias"], direct["prompt"])
                else:
                    result = {"rc": 1, "wall_s": 0, "ttft_s": None, "last_byte_s": None,
                              "stdout": "", "stderr": f"direct mode not supported for {direct['backend']}",
                              "end_reason": "exit", "backend_stats": None}
            else:
                result = run_agent_streamed(cmd, env)
        if progress is not None:
            progress["done"] += 1
        out_path = run_dir / f"{label}__{tag}.txt"
        out_path.write_text(result["stdout"])
        if result["stderr"]:
            (run_dir / f"{label}__{tag}.stderr.log").write_text(result["stderr"])
        backend_stats = result.get("backend_stats")
        extra_log = ""
        if backend_stats and backend_stats.get("eval_tok_per_s") is not None:
            extra_log = (f" eval={backend_stats['eval_count']}tok @ "
                         f"{backend_stats['eval_tok_per_s']}t/s")
        log(
            f"    {tag}: wall={result['wall_s']:.2f}s "
            f"ttft={'%.2f' % result['ttft_s'] if result['ttft_s'] else '–'}s "
            f"chars={len(result['stdout'])} rc={result['rc']}{extra_log}"
        )
        if is_warmup:
            continue
        tokens = estimate_tokens(result["stdout"])
        wall_s = max(result["wall_s"], 1e-6)
        throughput = round(tokens / wall_s, 2) if tokens else 0.0
        streamed = bool(
            result["ttft_s"] is not None
            and result.get("last_byte_s") is not None
            and (result["last_byte_s"] - result["ttft_s"]) > 0.5
        )
        text = result["stdout"].lower()
        iter_row = {
            "iter": i - warmup,
            "wall_s": round(result["wall_s"], 3),
            "ttft_s": round(result["ttft_s"], 3) if result["ttft_s"] else None,
            "output_chars": len(result["stdout"]),
            "output_tokens_est": tokens,
            "throughput_tok_per_s_est": throughput,
            "streamed": streamed,
            "rc": result["rc"],
            "output_file": out_path.name,
            "looks_like_html": ("<html" in text) or ("<!doctype" in text),
            "has_button": "<button" in text,
            "has_script": "<script" in text,
            "end_reason": result.get("end_reason"),
        }
        if backend_stats:
            iter_row["backend_stats"] = backend_stats
        iters.append(iter_row)
    return summarize(label, iters)


def summarize(label, iters):
    if not iters:
        return {"label": label, "iterations": [], "summary": None}
    walls = [r["wall_s"] for r in iters]
    ttfts = [r["ttft_s"] for r in iters if r["ttft_s"] is not None]
    tps = [r["throughput_tok_per_s_est"] for r in iters]
    return {
        "label": label,
        "iterations": iters,
        "summary": {
            "wall_s_mean": round(mean(walls), 3),
            "wall_s_median": round(median(walls), 3),
            "wall_s_stdev": round(pstdev(walls), 3) if len(walls) > 1 else 0.0,
            "ttft_s_mean": round(mean(ttfts), 3) if ttfts else None,
            "throughput_tok_per_s_mean_est": round(mean(tps), 2),
            "streamed_all": all(r["streamed"] for r in iters),
            "all_runs_html": all(r["looks_like_html"] for r in iters),
            "all_runs_have_buttons": all(r["has_button"] for r in iters),
        },
    }


# ---- cleanup ---------------------------------------------------------------
def cleanup(state, cfg):
    """Uninstall apps and remove pulled models. Config restoration happens
    separately and unconditionally at end-of-run."""
    log("Uninstalling…")
    # Restore configs idempotently in case this is called via --cleanup-only.
    restore_pi_models_config()
    restore_opencode_config()
    restore_opencode_notools_agent()
    LMStudio.restore_guardrails()
    inst = state.get("installed_by_us", {})
    pulled = state.get("model_pulled_by_us", {})

    # Remove models directly from disk — avoids needing a running server.
    ollama_manifests = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    for name in (pulled.get("ollama") or []):
        # name is e.g. "qwen3:1.7b" → family="qwen3", tag="1.7b"
        if ":" in name:
            family, tag = name.split(":", 1)
        else:
            family, tag = name, "latest"
        manifest = ollama_manifests / family / tag
        if manifest.exists():
            manifest.unlink()
            log(f"Removed ollama model {name}")

    for name in (pulled.get("lmstudio") or []):
        # LM Studio models live under ~/.lmstudio/models/<publisher>/<repo>/
        lms_models = Path.home() / ".lmstudio" / "models"
        if "/" in name:
            model_path = lms_models / name.replace("/", os.sep)
        else:
            model_path = None
        # Search for any directory matching the last path component
        found = list(lms_models.rglob(name.split("/")[-1])) if lms_models.exists() else []
        for p in found:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                log(f"Removed LM Studio model {name}")

    for name in (pulled.get("omlx") or []):
        model_path = OMLX_MODEL_DIR / name
        if model_path.exists():
            shutil.rmtree(model_path, ignore_errors=True)
            log(f"Removed oMLX model {name}")

    Ollama.stop()
    OMLX.stop()
    if inst.get("fzf") and which("brew"):
        run("brew uninstall fzf", check=False)
    if inst.get("pi"):
        if which("npm"):
            run("npm uninstall -g @mariozechner/pi-coding-agent", check=False)
        shutil.rmtree(Path.home() / ".pi", ignore_errors=True)
    if inst.get("opencode"):
        if which("opencode"):
            run(["opencode", "uninstall"], check=False)
        shutil.rmtree(Path.home() / ".opencode", ignore_errors=True)
    if inst.get("ollama") and IS_MAC and which("brew"):
        run("brew uninstall ollama", check=False)
        shutil.rmtree(Path.home() / ".ollama", ignore_errors=True)
    if inst.get("lmstudio") and IS_MAC and which("brew"):
        run("brew uninstall --cask lm-studio", check=False)
        shutil.rmtree(Path.home() / ".lmstudio", ignore_errors=True)
    if inst.get("omlx"):
        if OMLX_VENV_DIR.exists():
            shutil.rmtree(OMLX_VENV_DIR, ignore_errors=True)
        if OMLX_CLONE_DIR.exists():
            shutil.rmtree(OMLX_CLONE_DIR, ignore_errors=True)
        shutil.rmtree(Path.home() / ".omlx", ignore_errors=True)
    STATE_FILE.unlink(missing_ok=True)
    log("Cleanup complete.")


# ---- interactive model picker -----------------------------------------------

_DEFAULT_PROMPT = (
    "Build a single-page website in plain HTML, CSS, and JavaScript with two buttons. "
    "The first button fetches a random joke from https://icanhazdadjoke.com/ (send header "
    "'Accept: application/json') and displays it on the page. The second button toggles dark "
    "mode by adding/removing a 'dark' CSS class on the body, with appropriate styles for both "
    "modes. Output a single complete HTML file with inline <style> and <script> tags. No build "
    "tools, no frameworks. Output only the HTML, no commentary. Ask no questions and make any "
    "required assumptions yourself."
)


def _fmt_size(size_bytes):
    if not size_bytes:
        return ""
    gb = size_bytes / 1e9
    return f"{gb:.1f} GB" if gb >= 1 else f"{size_bytes / 1e6:.0f} MB"


# Strip vendor prefix, lowercase, normalise separators, and drop common
# quant/format suffixes so the same model picked from different backends
# (e.g. ollama "qwen3:1.7b", lmstudio "qwen/qwen3-1.7b") lands on the same
# label and gets grouped side-by-side in the viewer.
_LABEL_SUFFIXES = (
    "-mlx-bf16", "-mlx-4bit", "-mlx-8bit", "-mlx",
    "-gguf", "-q4_k_m", "-q4_k_s", "-q5_k_m", "-q5_k_s", "-q8_0",
    "-4bit", "-8bit", "-bf16", "-fp16",
)


def _normalize_label(model_id):
    s = model_id.split("/")[-1].lower()
    for sep in (":", "_", " "):
        s = s.replace(sep, "-")
    changed = True
    while changed:
        changed = False
        for suf in _LABEL_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return s


def _list_ollama_installed():
    """Return locally pulled Ollama models by reading manifests directly.

    Reads ~/.ollama/models/manifests/registry.ollama.ai/library/ so it works
    even when the Ollama server isn't running yet.
    """
    manifests_root = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    if not manifests_root.exists():
        return []
    results = []
    for family_dir in sorted(manifests_root.iterdir()):
        if not family_dir.is_dir():
            continue
        for tag_file in sorted(family_dir.iterdir()):
            if not tag_file.is_file():
                continue
            model_id = f"{family_dir.name}:{tag_file.name}"
            size = ""
            try:
                manifest = json.loads(tag_file.read_text())
                total = sum(layer.get("size", 0) for layer in manifest.get("layers", []))
                size = _fmt_size(total) if total else ""
            except Exception:
                pass
            results.append({"id": model_id, "size": size})
    return results


def _search_ollama(term):
    """Search Ollama by scraping the library page for all tag variants + sizes.

    Heuristic: the first whitespace-separated word is the family slug. Try it
    verbatim first; if that returns nothing and the term has multiple words,
    retry with the first two words concatenated (handles "llama 3" → "llama3").
    Any remaining words are treated as a substring filter on the tag.
    """
    import re

    parts = term.lower().strip().replace("_", "-").split()
    if not parts:
        return []

    def _fetch(family, tag_filter=""):
        try:
            req = urllib.request.Request(
                f"https://ollama.com/library/{family}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=6) as r:
                html = r.read().decode("utf-8", errors="ignore")
            tag_pat = re.compile(rf'{re.escape(family)}:[a-zA-Z0-9._-]+')
            size_pat = re.compile(r'(\d+\.?\d*)\s*(GB|MB)')
            seen = {}
            for m in tag_pat.finditer(html):
                tag = m.group(0)
                if tag in seen:
                    continue
                ctx = html[m.start(): m.start() + 400]
                sm = size_pat.search(ctx)
                seen[tag] = f"{sm.group(1)} {sm.group(2)}" if sm else ""
            tags = sorted(seen.keys())
            if tag_filter:
                tags = [t for t in tags if tag_filter in t.lower()]
            return [{"id": t, "size": seen.get(t, "")} for t in tags]
        except Exception:
            return []

    first_results = _fetch(parts[0], " ".join(parts[1:]))
    if first_results:
        return first_results
    if len(parts) >= 2:
        joined = parts[0] + parts[1]
        return _fetch(joined, " ".join(parts[2:]))
    return []


def _list_lmstudio_installed():
    """Return locally downloaded LM Studio models from `lms ls`."""
    if not which("lms"):
        return []
    try:
        rc, out, _ = run(["lms", "ls"], capture=True, check=False)
        if rc != 0:
            return []
        results = []
        for line in out.splitlines():
            # lms ls rows look like: "qwen/qwen3-1.7b (1 variant)  1.7B  qwen3  1.14 GB  Local"
            parts = line.split()
            if not parts or parts[0].startswith(("LLM", "EMBED", "You ", "─")):
                continue
            model_id = parts[0].rstrip("*")  # strip any trailing marker
            if "/" not in model_id and "." not in model_id:
                continue  # skip header-like lines
            # Extract size — look for "X.XX GB" or "X MB" pattern
            size = ""
            for i, p in enumerate(parts):
                if p in ("GB", "MB") and i > 0:
                    size = f"{parts[i-1]} {p}"
                    break
            results.append({"id": model_id, "size": size, "desc": "installed"})
        return results
    except Exception:
        return []


def _hf_fetch_model_info(hf_id):
    """Fetch the full HuggingFace model API payload, or {} on failure."""
    headers = {"User-Agent": "agent-bench/1.0"}
    token = os.environ.get("HF_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{hf_id}",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            return json.loads(r.read()) or {}
    except Exception:
        return {}


def _hf_fetch_size(hf_id):
    """Return usedStorage bytes for a single HuggingFace repo, or 0 on failure."""
    return _hf_fetch_model_info(hf_id).get("usedStorage") or 0


def _hf_fill_sizes(results, id_key="id"):
    """Parallel-fetch usedStorage for each result and fill in the 'size' field."""
    import concurrent.futures
    hf_ids = [m[id_key] for m in results]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(_hf_fetch_size, hf_ids))
    for item, size_bytes in zip(results, sizes):
        item["size"] = _fmt_size(size_bytes) if size_bytes else ""


def _search_lmstudio_online(term):
    """Search HuggingFace for models downloadable via `lms get`.

    Searches both GGUF and (on Apple Silicon) MLX repos, since LM Studio
    supports both formats. Filters out repos that have neither a .gguf nor
    .safetensors file. `desc` notes the format + file count so the user can
    tell e.g. a single-file GGUF from a multi-quant repo.
    """
    try:
        import urllib.parse, concurrent.futures
        tags_to_search = ["gguf"]
        if LMSTUDIO_SUPPORTED:
            tags_to_search.append("mlx")

        all_ids = []
        seen = set()
        for tag in tags_to_search:
            params = urllib.parse.urlencode({
                "search": term, "tags": tag,
                "sort": "downloads", "limit": 15,
            })
            req = urllib.request.Request(
                f"https://huggingface.co/api/models?{params}",
                headers={"User-Agent": "agent-bench/1.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=6) as r:
                    items = json.loads(r.read())
            except Exception:
                continue
            for m in items:
                mid = m.get("id") or m.get("modelId", "")
                if mid and mid not in seen:
                    seen.add(mid)
                    all_ids.append(mid)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            infos = list(pool.map(_hf_fetch_model_info, all_ids))
        results = []
        for hf_id, info in zip(all_ids, infos):
            siblings = info.get("siblings") or []
            filenames = [(s.get("rfilename") or "").lower() for s in siblings]
            gguf_count = sum(1 for f in filenames if f.endswith(".gguf"))
            st_count   = sum(1 for f in filenames if f.endswith(".safetensors"))
            if gguf_count == 0 and st_count == 0:
                continue
            tags = info.get("tags") or []
            is_mlx = "mlx" in tags or "-mlx-" in hf_id.lower() or hf_id.lower().endswith("-mlx")
            if gguf_count > 0:
                desc = f"GGUF · {gguf_count} file{'s' if gguf_count > 1 else ''}"
            elif is_mlx:
                desc = "MLX (Apple Silicon)"
            else:
                desc = f"safetensors · {st_count} shard{'s' if st_count > 1 else ''}"
            size_bytes = info.get("usedStorage") or 0
            results.append({
                "id": hf_id,
                "size": _fmt_size(size_bytes),
                "desc": desc,
            })
            if len(results) >= 16:
                break
        return results
    except Exception:
        return []


def _list_omlx_installed():
    """Return locally downloaded oMLX models from OMLX_MODEL_DIR."""
    if not OMLX_MODEL_DIR.exists():
        return []
    results = []
    for d in sorted(OMLX_MODEL_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        except Exception:
            size_bytes = None
        results.append({"id": d.name, "size": _fmt_size(size_bytes)})
    return results


def _search_hf_mlx(term):
    """Search HuggingFace for MLX safetensors models across all publishers.

    Uses the `mlx` tag rather than filtering to a single org so packagers like
    mlx-community, lmstudio-community, and others all surface. Skips repos
    with no .safetensors files (mis-tagged or weight-less READMEs).
    """
    try:
        import urllib.parse, concurrent.futures
        params = urllib.parse.urlencode({
            "search": term, "tags": "mlx",
            "sort": "downloads", "limit": 20,
        })
        req = urllib.request.Request(
            f"https://huggingface.co/api/models?{params}",
            headers={"User-Agent": "agent-bench/1.0"},
        )
        with urllib.request.urlopen(req, timeout=6) as r:
            items = json.loads(r.read())
        hf_ids = [m.get("id") or m.get("modelId", "") for m in items]
        hf_ids = [h for h in hf_ids if h]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            infos = list(pool.map(_hf_fetch_model_info, hf_ids))
        results = []
        for hf_id, info in zip(hf_ids, infos):
            siblings = info.get("siblings") or []
            has_st = any(
                (s.get("rfilename") or "").lower().endswith(".safetensors")
                for s in siblings
            )
            if not has_st:
                continue
            tags = info.get("tags") or []
            is_mlx = "mlx" in tags or "mlx" in hf_id.lower()
            if not is_mlx:
                continue
            org = hf_id.split("/")[0] if "/" in hf_id else ""
            size_bytes = info.get("usedStorage") or 0
            results.append({
                "id": hf_id.split("/")[-1],
                "hf_id": hf_id,
                "size": _fmt_size(size_bytes),
                "desc": org,
            })
            if len(results) >= 12:
                break
        return results
    except Exception:
        return []


def _fzf_available():
    return bool(which("fzf"))


def _fzf_select(items, header=""):
    """Use fzf to interactively filter and multi-select from items.
    Returns list of chosen items. Tab selects, Enter confirms."""
    lines = []
    for m in items:
        size = f"\t{m['size']}" if m.get("size") else ""
        desc = f"  {m.get('desc', '')}" if m.get("desc") else ""
        lines.append(f"{m['id']}{size}{desc}")
    fzf_input = "\n".join(lines)
    cmd = [
        "fzf",
        "--multi",
        "--prompt", "  filter> ",
        "--height", "~40%",
        "--border", "rounded",
        "--info", "inline",
        "--bind", "space:toggle+down",
        "--header", header or "Space to select, Enter to confirm",
    ]
    r = subprocess.run(cmd, input=fzf_input, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    selected_ids = {line.split("\t")[0].strip() for line in r.stdout.strip().splitlines()}
    return [m for m in items if m["id"] in selected_ids]


def _display_list(items):
    for i, m in enumerate(items, 1):
        size = f"  {m['size']}" if m.get("size") else ""
        desc = f"  — {m.get('desc', '')}" if m.get("desc") else ""
        print(f"  {i:2}. {m['id']}{size}{desc}")


def _pick_indices(items, prompt="  Add (numbers, blank to skip): "):
    """Numbered fallback picker when fzf is not available."""
    if not items:
        return []
    raw = input(prompt).strip()
    chosen = []
    for tok in raw.split():
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(items):
                chosen.append(items[idx])
    return chosen


def _select_from(items, header=""):
    """Select one or more items from a list. Uses fzf if available, numbered list otherwise."""
    if not items:
        return []
    if _fzf_available():
        return _fzf_select(items, header)
    _display_list(items)
    return _pick_indices(items)


def _pick_backend_models(backend_label, search_fn=None, list_fn=None):
    """Interactive picker for one backend. Returns list of {id, [hf_id], size} dicts.

    list_fn  — callable returning [{id, size, ...}] of locally installed models
    search_fn — callable(term) returning [{id, size, ...}] from remote search
    Either or both may be provided. LM Studio has only list_fn; oMLX has only
    search_fn; Ollama has both (installed first, then optional remote search).
    """
    print(f"\n── {backend_label} {'─' * max(0, 50 - len(backend_label))}")
    if not _fzf_available():
        print("  (install fzf for live filtering: brew install fzf)")

    selected = []

    # Show installed models first (Ollama + LM Studio)
    if list_fn:
        installed = list_fn()
        if installed:
            chosen = _select_from(installed, f"{backend_label} — installed models (Space=multi-select, Enter=confirm)")
            selected.extend(chosen)
        else:
            print("  No models installed locally.")

    if search_fn and not selected:
        while True:
            term = input("  Search online (blank to skip): ").strip()
            if not term:
                break
            print("  Fetching…", end="", flush=True)
            results = search_fn(term)
            print()
            if not results:
                print("  No results — try a different term.")
                continue
            chosen = _select_from(results, f"{backend_label} — {term} (Space=multi-select, Enter=confirm)")
            selected.extend(chosen)
            if chosen and input("  Search again? [y/N] ").strip().lower() not in ("y", "yes"):
                break

    return selected


def model_picker(cfg_path, force=False):
    """Interactive model picker. Returns a config dict and saves it to cfg_path."""
    print()

    # Offer to reuse existing config (unless --configure forced the picker open)
    if not force and cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
            if existing.get("models"):
                ids = [m["id"] for m in existing["models"]]
                n = existing.get("iterations", 1)
                w = existing.get("warmup", 0)
                print(f"Last config: {', '.join(ids)}")
                print(f"  backends: {', '.join(existing.get('backends', []))}  |  {n} iter(s), {w} warmup")
                print("  (run with --configure to change models)")
                ans = input("Reuse last config? [Y/n] ").strip().lower()
                if ans not in ("n", "no"):
                    return existing
        except Exception:
            pass

    print("\nPick models to benchmark per backend.")
    print("Models with the same label are compared side-by-side in the viewer.\n")

    # Collect selections per backend
    raw = {}  # backend -> list of {id, [hf_id], size}

    raw["ollama"]   = _pick_backend_models("Ollama",             search_fn=_search_ollama,   list_fn=_list_ollama_installed)
    raw["lmstudio"] = _pick_backend_models("LM Studio",          search_fn=_search_lmstudio_online, list_fn=_list_lmstudio_installed)
    raw["omlx"]     = _pick_backend_models("oMLX / HuggingFace", search_fn=_search_hf_mlx, list_fn=_list_omlx_installed)

    # Build model entries. Each selected item gets a label; same label = grouped.
    # We accumulate into a dict keyed by label.
    model_map = {}  # label -> entry dict

    for backend, items in raw.items():
        if not items:
            continue
        for item in items:
            label = _normalize_label(item["id"])
            if label not in model_map:
                model_map[label] = {"id": label}
            entry = model_map[label]
            if backend == "omlx":
                entry["omlx"]    = item["id"]
                entry["omlx_hf"] = item.get("hf_id", f"mlx-community/{item['id']}")
            else:
                entry[backend] = item["id"]

    models = list(model_map.values())

    if not models:
        print("\nNo models selected.")
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())
        sys.exit(1)

    print("\n── Config summary " + "─" * 34)
    for m in models:
        parts = [f"{k}={v}" for k, v in m.items() if k not in ("id", "omlx_hf")]
        print(f"  {m['id']}: {', '.join(parts)}")

    # Other settings
    print()
    print("\n── Agents " + "─" * 41)
    agent_items = [{"id": name, "desc": meta["desc"]} for name, meta in AGENTS.items()]
    chosen_agents = _select_from(agent_items, "Agents to benchmark (Space=multi-select, Enter=confirm)")
    agents = [a["id"] for a in chosen_agents] if chosen_agents else ["pi"]

    n_str = input("Iterations per combo [3]: ").strip()
    n_iter = int(n_str) if n_str.isdigit() else 3

    w_str = input("Warmup runs [1]: ").strip()
    warmup = int(w_str) if w_str.isdigit() else 1

    all_backends = {k for m in models for k in m if k not in ("id", "omlx_hf")}
    backends = [b for b in ("ollama", "lmstudio", "omlx") if b in all_backends]

    prompt = _DEFAULT_PROMPT
    if cfg_path.exists():
        try:
            prompt = json.loads(cfg_path.read_text()).get("prompt", _DEFAULT_PROMPT)
        except Exception:
            pass

    cfg = {
        "models": models,
        "agents": agents,
        "backends": backends,
        "iterations": n_iter,
        "warmup": warmup,
        "prompt": prompt,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"\nSaved → {cfg_path.name}")
    return cfg


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Config file to use (skips interactive picker)")
    ap.add_argument("--configure", action="store_true",
                    help="Open the interactive model picker even if a config already exists")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--cleanup-only", action="store_true")
    args = ap.parse_args()

    default_cfg_path = ROOT / "bench.config.json"
    explicit_config  = args.config is not None
    cfg_path         = Path(args.config) if args.config else default_cfg_path

    # Run the interactive picker when:
    #  - stdin is a TTY (interactive session)
    #  - no explicit --config flag (use --config to pin a file)
    #  - not a cleanup-only run
    # Always runs if --configure is passed (forces picker open).
    use_picker = (args.configure or (sys.stdin.isatty() and not explicit_config)) \
                 and not args.cleanup_only
    if use_picker:
        state = load_state()
        install_fzf(state); save_state(state)
        cfg = model_picker(cfg_path, force=args.configure)
    else:
        cfg = json.loads(cfg_path.read_text())
    state = load_state()

    if args.cleanup_only:
        cleanup(state, cfg)
        return

    # Validate new schema.
    required = ["models", "agents", "backends", "prompt", "iterations", "warmup"]
    missing = [k for k in required if k not in cfg]
    if missing:
        err(f"Config missing required keys: {missing}. See bench.config.json for the schema.")
        sys.exit(2)

    requested_agents   = list(cfg["agents"])
    requested_backends = list(cfg["backends"])
    models             = list(cfg["models"])

    unknown_agents = [a for a in requested_agents if a not in AGENTS]
    if unknown_agents:
        err(f"Unknown agents in config: {unknown_agents}. Known: {list(AGENTS)}")
        sys.exit(2)

    pre = {
        "agents":   {a: AGENTS[a]["is_installed"]() for a in requested_agents},
        "backends": {b: (have_ollama() if b == "ollama"
                          else have_lmstudio() if b == "lmstudio"
                          else have_omlx() if b == "omlx"
                          else False) for b in requested_backends},
        "ollama_models":   {m["id"]: have_ollama_model(m["ollama"])
                             for m in models if "ollama" in m},
        "lmstudio_models": {m["id"]: have_lmstudio_model(m["lmstudio"])
                             for m in models if "lmstudio" in m},
        "omlx_models":     {m["id"]: have_omlx_model(m["omlx"]) if "omlx" in m else False
                             for m in models},
    }
    state["pre_existing"] = pre
    save_state(state)
    log(f"Pre-flight: {pre}")

    if not args.skip_install:
        if "ollama" in requested_backends:
            install_ollama(state); save_state(state)
        if "lmstudio" in requested_backends:
            install_lmstudio(state); save_state(state)
        if "omlx" in requested_backends:
            install_omlx(state); save_state(state)
        if "pi" in requested_agents:
            install_pi(state); save_state(state)
        if "opencode" in requested_agents:
            install_opencode(state); save_state(state)

    # Download oMLX models before any server starts — the server scans
    # model-dir on startup, so models must be present first.
    if "omlx" in requested_backends and have_omlx():
        for m in models:
            if "omlx" in m and not have_omlx_model(m["omlx"]):
                try:
                    download_omlx_model(m["omlx"], hf_repo=m.get("omlx_hf"))
                    state["model_pulled_by_us"].setdefault("omlx", []).append(m["omlx"])
                    save_state(state)
                except Exception as e:
                    warn(f"oMLX model download failed for '{m['omlx']}': {e}")
                    warn("Set HF_TOKEN env var if the repo requires authentication.")

    # Pull models for each backend one at a time — never run two backends simultaneously.
    # Ollama pull phase: start ollama, pull missing models, stop before moving on.
    if "ollama" in requested_backends:
        Ollama.start()
        for m in models:
            if "ollama" in m and not have_ollama_model(m["ollama"]):
                log(f"Pulling {m['ollama']} via ollama…")
                run(["ollama", "pull", m["ollama"]])
                state["model_pulled_by_us"].setdefault("ollama", []).append(m["ollama"])
                save_state(state)
        Ollama.stop()
        time.sleep(2)

    # LM Studio pull phase: start server, download missing models, stop.
    if "lmstudio" in requested_backends and have_lmstudio():
        # Relax model-loading guardrails before the server first reads its
        # settings — the LM Studio process caches settings.json at startup.
        LMStudio.relax_guardrails()
        LMStudio.start()
        for m in models:
            if "lmstudio" in m and not have_lmstudio_model(m["lmstudio"]):
                alias = m["lmstudio"]
                # Force the right format when the model name carries one — `lms
                # get -y` otherwise picks based on hardware/preferences and may
                # pull a different quant.
                fmt_flag = []
                low = alias.lower()
                if "-mlx" in low or "mlx-" in low:
                    fmt_flag = ["--mlx"]
                elif "gguf" in low:
                    fmt_flag = ["--gguf"]

                # `lms get <org/name>` resolves against the LM Studio catalog
                # first. For HF repos not in the curated catalog (e.g. brand-new
                # models), that lookup fails — retry with the full HF URL form.
                attempts = [alias]
                if "/" in alias and not alias.startswith("http"):
                    attempts.append(f"https://huggingface.co/{alias}")

                rc = None
                last_stderr = ""
                for target in attempts:
                    log(f"Downloading {target} via lms… (large models may take several minutes)")
                    r = subprocess.run(
                        ["lms", "get", target, "-y", *fmt_flag],
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        text=True, check=False,
                    )
                    rc = r.returncode
                    last_stderr = r.stderr or ""
                    if rc == 0 and have_lmstudio_model(alias):
                        break
                    if "does not exist" not in last_stderr and "permission" not in last_stderr:
                        break  # other failure mode — don't try fallback

                if rc == 0 and have_lmstudio_model(alias):
                    state["model_pulled_by_us"].setdefault("lmstudio", []).append(alias)
                    save_state(state)
                elif rc == 0:
                    warn(f"`lms get {alias}` returned rc=0 but model not detected on disk — load it manually in LM Studio.")
                else:
                    warn(f"`lms get {alias}` failed (rc={rc}) — load it manually in LM Studio.")
                    if last_stderr.strip():
                        warn(last_stderr.strip()[:500])

        # Resolve each lmstudio entry's HF download path to LM Studio's API
        # alias (e.g. "lmstudio-community/Qwen3-Coder-Next-MLX-4bit" →
        # "qwen3-coder-next-mlx") while the server is still up. Stored in a
        # separate field so the original HF path stays usable for the on-disk
        # detection check.
        for m in models:
            if "lmstudio" in m and have_lmstudio_model(m["lmstudio"]):
                api_id = _lmstudio_resolve_api_id(m["lmstudio"])
                if api_id:
                    if api_id != m["lmstudio"]:
                        log(f"LM Studio API ID for {m['lmstudio']} → {api_id}")
                    m["lmstudio_api_id"] = api_id

        LMStudio.stop()
        time.sleep(2)

    # Configure opencode with all model aliases per backend.
    # For LM Studio, prefer the resolved API ID (set during the download phase
    # above) because pi/opencode will send that as the OpenAI `model` field.
    _ollama_models = [m["ollama"] for m in models if "ollama" in m] if "ollama" in requested_backends else []
    _lmstudio_models = [m.get("lmstudio_api_id") or m["lmstudio"] for m in models if "lmstudio" in m] if "lmstudio" in requested_backends else []
    _omlx_models = [m["omlx"] for m in models if "omlx" in m] if "omlx" in requested_backends else []

    if "pi" in requested_agents:
        write_pi_models_config(
            ollama_models=_ollama_models,
            lmstudio_models=_lmstudio_models,
            omlx_models=_omlx_models,
        )
    if "opencode" in requested_agents:
        write_opencode_config(
            ollama_models=_ollama_models,
            lmstudio_models=_lmstudio_models,
            omlx_models=_omlx_models,
        )
        write_opencode_notools_agent()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Results → {run_dir}")
    (run_dir / "prompt.txt").write_text(cfg["prompt"])

    prompt = cfg["prompt"]
    n_iter, warmup = cfg["iterations"], cfg["warmup"]
    combos = []

    # ---- Build the matrix (backend-first so we can start/stop each backend
    #      exclusively — no two backends running at the same time) ----
    planned = []
    for backend in requested_backends:
        backend_up = (have_ollama() if backend == "ollama"
                      else have_omlx() if backend == "omlx"
                      else have_lmstudio())
        if not backend_up:
            warn(f"Skipping all {backend} combos — backend not available.")
            continue
        for agent in requested_agents:
            if not AGENTS[agent]["is_installed"]():
                # warn once at top level, not per backend
                continue
            if backend not in AGENTS[agent]["supports_backends"]:
                warn(f"Skipping {agent}+{backend} — agent doesn't support that backend.")
                continue
            for m in models:
                if backend not in m:
                    warn(f"Skipping {agent}+{backend}+{m['id']} — no '{backend}' alias for model.")
                    continue
                # For LM Studio, the HF download path differs from the API alias.
                # Detect with the HF path, but invoke agents with the API alias.
                detect_alias = m[backend]
                runtime_alias = (
                    m.get("lmstudio_api_id") or m[backend]
                    if backend == "lmstudio" else m[backend]
                )
                model_present = (
                    have_ollama_model(detect_alias)   if backend == "ollama"   else
                    have_lmstudio_model(detect_alias) if backend == "lmstudio" else
                    have_omlx_model(detect_alias)     if backend == "omlx"     else
                    False
                )
                if not model_present:
                    warn(f"Skipping {agent}+{backend}+{m['id']} — model '{detect_alias}' not present on {backend} (download likely failed).")
                    continue
                model_alias = runtime_alias
                label = f"{agent}+{backend}+{m['id']}"
                is_direct = AGENTS[agent].get("direct", False)
                if is_direct:
                    cmd, extra_env = None, {}
                else:
                    cmd, extra_env = AGENTS[agent]["build_cmd"](model_alias, backend, prompt)
                env = os.environ.copy()
                env.update(extra_env)
                planned.append({
                    "label": label, "agent": agent, "backend": backend,
                    "model_id": m["id"], "model_alias": model_alias,
                    "cmd": cmd, "env": env,
                    "direct": {"backend": backend, "model_alias": model_alias, "prompt": prompt} if is_direct else None,
                })

    # Warn once for any agent that isn't installed (backend loop above skips silently).
    for agent in requested_agents:
        if not AGENTS[agent]["is_installed"]():
            warn(f"Skipping agent '{agent}' — not installed.")

    if not planned:
        err("No combos to run after filtering. Check config + installed agents/backends/models.")
        sys.exit(1)

    total_runs = len(planned) * (warmup + n_iter)
    progress = {"done": 0, "total": total_runs, "start": time.monotonic()}
    log(f"Plan: {len(planned)} combos × {warmup + n_iter} runs = {total_runs} total")

    _BACKEND_START = {
        "ollama":   Ollama.start,
        "lmstudio": LMStudio.start,
        "omlx":     OMLX.start,
    }
    _BACKEND_STOP = {
        "ollama":   Ollama.stop,
        "lmstudio": LMStudio.stop,
        "omlx":     OMLX.stop,
    }
    current_backend = None
    current_lmstudio_model = None
    for p in planned:
        if p["backend"] != current_backend:
            # Stop the previous backend before starting the next one.
            if current_backend is not None:
                log(f"Stopping {current_backend} backend…")
                _BACKEND_STOP[current_backend]()
                time.sleep(2)
            log(f"Starting {p['backend']} backend…")
            _BACKEND_START[p["backend"]]()
            current_backend = p["backend"]
            current_lmstudio_model = None

        # For LM Studio, explicitly unload all models then load the target before
        # each combo — auto-load on first request is unreliable, and we need to
        # ensure only one model is in RAM at a time (especially for large models).
        if p["backend"] == "lmstudio" and p["model_alias"] != current_lmstudio_model:
            if current_lmstudio_model is not None:
                log(f"Unloading all LM Studio models…")
                run(["lms", "unload", "--all"], check=False)
                time.sleep(3)
            log(f"Loading LM Studio model {p['model_alias']}…")
            run(["lms", "load", p["model_alias"], "-y"], check=False)
            current_lmstudio_model = p["model_alias"]

        result = bench_one(p["label"], p["cmd"], p["env"], run_dir, n_iter, warmup,
                           progress, direct=p.get("direct"))
        # Tag combo with its dimensions for the viewer.
        result.update({"agent": p["agent"], "backend": p["backend"], "model_id": p["model_id"]})
        combos.append(result)

    # Stop the last backend.
    if current_backend is not None:
        _BACKEND_STOP[current_backend]()
    results = {
        "timestamp": ts,
        "platform": platform.platform(),
        "cpu": cpu_info(),
        "machine": platform.machine(),
        "config": cfg,
        "pre_existing": pre,
        "combos": combos,
    }
    out_file = run_dir / "results.json"
    out_file.write_text(json.dumps(results, indent=2))
    log(f"Wrote {out_file}")
    log(f"View: open {ROOT / 'viewer.html'} and load this results.json")

    print("\n=== Summary ===")
    print(f"{'combo':<32} {'wall_med':>10} {'ttft_mean':>10} {'tok/s':>8} {'stream':>7} {'html?':>6} {'btns?':>6}")
    for c in combos:
        s = c["summary"] or {}
        print(
            f"{c['label']:<32} "
            f"{str(s.get('wall_s_median','–')):>10} "
            f"{str(s.get('ttft_s_mean','–')):>10} "
            f"{str(s.get('throughput_tok_per_s_mean_est','–')):>8} "
            f"{str(s.get('streamed_all','–')):>7} "
            f"{str(s.get('all_runs_html','–')):>6} "
            f"{str(s.get('all_runs_have_buttons','–')):>6}"
        )

    # Always restore opencode config + notools agent — transient scaffolding,
    # not something the user opted into installing.
    restore_pi_models_config()
    restore_opencode_config()
    restore_opencode_notools_agent()
    LMStudio.restore_guardrails()
    log("Restored agent configs (pi models, opencode notools/config, LM Studio guardrails).")

    print()
    if sys.stdin.isatty():
        answer = input("Uninstall apps and remove downloaded models? [y/N] ").strip().lower()
    else:
        answer = "n"
    if answer == "y":
        cleanup(state, cfg)
    else:
        Ollama.stop()
        log("Apps left in place. Re-run with --cleanup-only later to uninstall.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        err("Interrupted.")
        sys.exit(130)
