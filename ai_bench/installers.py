"""Installer and cleanup utilities for AI-Bench.

Detects, installs, starts, stops, and cleans up backend tooling
(Ollama, LM Studio, oMLX) and agents (pi, opencode).
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ai_bench.agents import (
    restore_opencode_config,
    restore_opencode_notools_agent,
    restore_pi_models_config,
)
from ai_bench.backends import Ollama, LMStudio, OMLX, OLLAMA_MANIFESTS
from ai_bench.log import log, warn, err
from ai_bench.state import STATE_FILE, load_state

# ---- platform constants ----------------------------------------------------
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_ARM64 = platform.machine() in ("arm64", "aarch64")
LMSTUDIO_SUPPORTED = IS_MAC and IS_ARM64

# oMLX requires Apple Silicon
OMLX_SUPPORTED = IS_MAC and IS_ARM64
OMLX_CLONE_DIR = Path.home() / ".local" / "share" / "omlx-src"
OMLX_VENV_DIR  = Path.home() / ".local" / "share" / "omlx-venv"
OMLX_BIN       = OMLX_VENV_DIR / "bin" / "omlx"
OMLX_MODEL_DIR = Path.home() / ".omlx" / "models"


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
    created_dirs = state.setdefault("created_dirs", [])
    OMLX_CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not OMLX_CLONE_DIR.exists():
        run(["git", "clone", "https://github.com/jundot/omlx", str(OMLX_CLONE_DIR)])
        created_dirs.append(str(OMLX_CLONE_DIR))
    if not OMLX_VENV_DIR.exists():
        # oMLX depends on mlx>=0.31.2 which brew installs for Python 3.14.
        # Use brew's python3.14 explicitly so the venv has access to it.
        python = (which("python3.14")
                  or "/opt/homebrew/opt/python@3.14/bin/python3.14"
                  or sys.executable)
        run([python, "-m", "venv", "--system-site-packages", str(OMLX_VENV_DIR)])
        created_dirs.append(str(OMLX_VENV_DIR))
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


# ---- lifecycle dispatch dicts ----------------------------------------------
HAVE_BACKEND = {
    "ollama": have_ollama,
    "lmstudio": have_lmstudio,
    "omlx": have_omlx,
}

BACKEND_START = {
    "ollama": Ollama.start,
    "lmstudio": LMStudio.start,
    "omlx": OMLX.start,
}

BACKEND_STOP = {
    "ollama": Ollama.stop,
    "lmstudio": LMStudio.stop,
    "omlx": OMLX.stop,
}


# ---- cleanup -----------------------------------------------------------------
def _get_artifacts_to_cleanup(state, cfg):
    """Return structured list of artifacts to cleanup, grouped by type."""
    inst = state.get("installed_by_us", {})
    pulled = state.get("model_pulled_by_us", {})
    
    artifacts = {
        "tools_to_uninstall": [],
        "files_to_remove": [],
        "directories_to_remove": [],
        "models_to_remove": [],
    }
    
    # Tools
    if inst.get("fzf"):
        artifacts["tools_to_uninstall"].append(("fzf", "brew uninstall fzf"))
    if inst.get("pi"):
        artifacts["tools_to_uninstall"].append(("pi", "npm uninstall -g @mariozechner/pi-coding-agent"))
    if inst.get("opencode"):
        artifacts["tools_to_uninstall"].append(("opencode", "opencode uninstall"))
    if inst.get("ollama") and IS_MAC:
        artifacts["tools_to_uninstall"].append(("ollama", "brew uninstall ollama"))
    if inst.get("lmstudio") and IS_MAC:
        artifacts["tools_to_uninstall"].append(("lmstudio", "brew uninstall --cask lm-studio"))
    
    for path in state.get("created_dirs", []):
        artifacts["directories_to_remove"].append({
            "path": path,
            "reason": "AI-Bench recorded this directory as created by the benchmark",
        })
    
    # Models
    for name in (pulled.get("ollama") or []):
        artifacts["models_to_remove"].append({
            "backend": "ollama",
            "name": name,
            "path": str(OLLAMA_MANIFESTS / name.split(":")[0] / name.split(":")[1] if ":" in name else OLLAMA_MANIFESTS / name.split(":")[0] / "latest"),
        })
    
    for name in (pulled.get("lmstudio") or []):
        # For LM Studio, we need to find the actual path
        lms_models = Path.home() / ".lmstudio" / "models"
        if "/" in name:
            candidate = lms_models / name.replace("/", os.sep)
        else:
            candidate = None
        artifacts["models_to_remove"].append({
            "backend": "lmstudio", 
            "name": name,
            "path": str(candidate) if candidate else "searching...",
        })
    
    for name in (pulled.get("omlx") or []):
        artifacts["models_to_remove"].append({
            "backend": "omlx",
            "name": name,
            "path": str(OMLX_MODEL_DIR / name),
        })
    
    return artifacts


def _print_cleanup_summary(artifacts, dry_run=True):
    """Print a summary of what would be removed during cleanup."""
    log("Cleanup Summary:")

    if artifacts["tools_to_uninstall"]:
        log("  Tools to uninstall:")
        for name, cmd in artifacts["tools_to_uninstall"]:
            log(f"    - {name} ({cmd})")

    if artifacts["files_to_remove"]:
        log("  Files to remove:")
        for item in artifacts["files_to_remove"]:
            log(f"    - {item['path']} ({item.get('reason', '')})")

    if artifacts["directories_to_remove"]:
        log("  Directories to remove:")
        for item in artifacts["directories_to_remove"]:
            log(f"    - {item['path']} ({item['reason']})")

    if artifacts["models_to_remove"]:
        log("  Models to remove:")
        for item in artifacts["models_to_remove"]:
            log(f"    - {item['name']} ({item['backend']})")

    if dry_run:
        log("  (dry run - nothing will be removed)")

    if not any([artifacts["tools_to_uninstall"], artifacts["files_to_remove"],
                artifacts["directories_to_remove"], artifacts["models_to_remove"]]):
        log("  No AI-Bench artifacts to cleanup.")


def cleanup(state, cfg, dry_run=False):
    """Uninstall apps and remove pulled models. Config restoration happens
    separately and unconditionally at end-of-run."""
    artifacts = _get_artifacts_to_cleanup(state, cfg)
    
    if dry_run:
        _print_cleanup_summary(artifacts, dry_run=True)
        return
    
    # Interactive confirmation
    if sys.stdin.isatty():
        _print_cleanup_summary(artifacts, dry_run=False)
        answer = input("\nProceed with cleanup? [y/N] ").strip().lower()
        if answer != "y":
            log("Cleanup cancelled.")
            return
        log("Uninstalling…")
    else:
        log("Uninstalling…")
    
    # Restore configs idempotently in case this is called via --cleanup-only.
    restore_pi_models_config()
    restore_opencode_config()
    restore_opencode_notools_agent()
    LMStudio.restore_guardrails()
    inst = state.get("installed_by_us", {})
    pulled = state.get("model_pulled_by_us", {})

    for name in (pulled.get("ollama") or []):
        if which("ollama"):
            run(["ollama", "rm", name], check=False)
            log(f"Removed ollama model {name}")
            continue
        family, tag = name.split(":", 1) if ":" in name else (name, "latest")
        manifest = OLLAMA_MANIFESTS / family / tag
        if manifest.exists():
            manifest.unlink()
            log(f"Removed ollama model {name}")

    for name in (pulled.get("lmstudio") or []):
        lms_models = Path.home() / ".lmstudio" / "models"
        candidates = []
        if "/" in name:
            candidates.append(lms_models / name.replace("/", os.sep))
        candidates.append(lms_models / name)
        for p in candidates:
            if p.exists() and p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                log(f"Removed LM Studio model {name}")
                break

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
    if inst.get("opencode"):
        if which("opencode"):
            run(["opencode", "uninstall"], check=False)
    if inst.get("ollama") and IS_MAC and which("brew"):
        run("brew uninstall ollama", check=False)
    if inst.get("lmstudio") and IS_MAC and which("brew"):
        run("brew uninstall --cask lm-studio", check=False)
    for item in artifacts["directories_to_remove"]:
        shutil.rmtree(Path(item["path"]), ignore_errors=True)
    STATE_FILE.unlink(missing_ok=True)
    log("Cleanup complete.")
