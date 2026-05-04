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
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).parent.resolve()
STATE_FILE = ROOT / ".bench-state.json"
RESULTS_DIR = ROOT / "results"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_ARM64 = platform.machine() in ("arm64", "aarch64")
# LM Studio on Mac requires Apple Silicon. On Linux it's AppImage-only.
LMSTUDIO_SUPPORTED = IS_MAC and IS_ARM64


def log(msg, prefix="•"):
    print(f"{prefix} {msg}", flush=True)


def warn(msg):
    log(msg, prefix="!")


def err(msg):
    log(msg, prefix="x")


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
    if which("lms"):
        return True
    if IS_MAC and Path("/Applications/LM Studio.app").exists():
        return True
    return False


def have_ollama_model(name):
    if not have_ollama():
        return False
    rc, out, _ = run(["ollama", "list"], capture=True, check=False)
    if rc != 0:
        return False
    base = name.split(":")[0]
    for line in out.splitlines()[1:]:
        if not line.strip():
            continue
        if line.split()[0].split(":")[0] == base:
            return True
    return False


def have_lmstudio_model(name):
    if not which("lms"):
        return False
    rc, out, _ = run(["lms", "ls"], capture=True, check=False)
    if rc != 0:
        return False
    return name.split(":")[0].lower() in out.lower()


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
    # bootstrap lms cli
    run(
        "~/.cache/lm-studio/bin/lms bootstrap "
        "|| /Applications/'LM Studio.app'/Contents/Resources/app/.webpack/main/lms bootstrap "
        "|| true",
        check=False,
    )
    state["installed_by_us"]["lmstudio"] = True


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

    @classmethod
    def start(cls):
        if cls.is_up():
            return
        if not which("lms"):
            warn("`lms` CLI not found — start LM Studio server manually if you want LM Studio combos.")
            return
        log("Starting LM Studio server…")
        run("lms server start", check=False)
        for _ in range(40):
            if cls.is_up():
                return
            time.sleep(0.5)
        warn("LM Studio server didn't come up in time.")


# ---- agent invocation ------------------------------------------------------
def run_agent_streamed(cmd, env, *, total_timeout=900):
    """Spawn agent, stream stdout, time first→last byte and total wall."""
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
    try:
        while True:
            ch = p.stdout.read(1)
            if not ch:
                break
            now = time.monotonic()
            if first_byte_t is None and ch.strip():
                first_byte_t = now
            last_byte_t = now
            chunks.append(ch)
            if now - start > total_timeout:
                p.kill()
                break
    finally:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
    end = time.monotonic()
    out = b"".join(chunks).decode("utf-8", errors="replace")
    try:
        err_out = (p.stderr.read() or b"").decode("utf-8", errors="replace")
    except Exception:
        err_out = ""
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
    return max(1, round(len(text) / 4))




# ---- agent configs ---------------------------------------------------------
OPENCODE_CFG = Path.home() / ".config/opencode/opencode.json"
OPENCODE_CFG_BAK = OPENCODE_CFG.with_suffix(".json.bench-bak")
OPENCODE_AGENT_DIR = Path.home() / ".config/opencode/agent"
OPENCODE_NOTOOLS_AGENT = OPENCODE_AGENT_DIR / "notools.md"
OPENCODE_NOTOOLS_BAK = OPENCODE_AGENT_DIR / "notools.md.bench-bak"

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


def write_opencode_config(ollama_models, lmstudio_models):
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
    cfg = {"$schema": "https://opencode.ai/config.json", "provider": providers}
    OPENCODE_CFG.write_text(json.dumps(cfg, indent=2))


# ---- agent registry --------------------------------------------------------
# Each agent declares: which backends it supports, and how to build the command
# given (model_alias, backend, prompt). Returns (cmd_list, extra_env_dict).
def _pi_cmd(model_alias, backend, prompt):
    # pi has a built-in 'ollama' provider. For other OpenAI-compatible
    # backends we point pi at a custom URL via env vars and use 'openai/<model>'.
    if backend == "ollama":
        return (
            ["pi", "-nt", "-p", "--model", f"ollama/{model_alias}", prompt],
            {},
        )
    elif backend == "lmstudio":
        return (
            ["pi", "-nt", "-p", "--model", f"openai/{model_alias}", prompt],
            {"OPENAI_BASE_URL": "http://127.0.0.1:1234/v1", "OPENAI_API_KEY": "sk-local"},
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
        "supports_backends": ["ollama", "lmstudio"],
        "build_cmd": _pi_cmd,
        "is_installed": lambda: bool(which("pi")),
    },
    "opencode": {
        "supports_backends": ["ollama", "lmstudio"],
        "build_cmd": _opencode_cmd,
        "is_installed": lambda: bool(which("opencode")),
    },
}


def restore_opencode_config():
    if OPENCODE_CFG_BAK.exists():
        shutil.move(str(OPENCODE_CFG_BAK), str(OPENCODE_CFG))
    elif OPENCODE_CFG.exists():
        OPENCODE_CFG.unlink()


# ---- benchmark -------------------------------------------------------------
def bench_one(label, cmd, env, run_dir, n_iter, warmup, progress=None):
    log(f"  ▶ {label}  (warmup={warmup}, iters={n_iter})")
    iters = []
    for i in range(warmup + n_iter):
        is_warmup = i < warmup
        tag = f"warmup-{i}" if is_warmup else f"iter-{i - warmup}"
        if progress is not None:
            progress["done"] += 0  # noop; counter advances after run
            done, total = progress["done"], progress["total"]
            elapsed = time.monotonic() - progress["start"]
            rate = elapsed / done if done else 0
            eta = rate * (total - done) if rate else 0
            mins = int(eta // 60)
            log(f"    [{done + 1}/{total}] starting {label} {tag}"
                f" — elapsed {int(elapsed)}s"
                + (f", est remaining ~{mins}m" if rate else ""))
        result = run_agent_streamed(cmd, env)
        if progress is not None:
            progress["done"] += 1
        out_path = run_dir / f"{label}__{tag}.txt"
        out_path.write_text(result["stdout"])
        if result["stderr"]:
            (run_dir / f"{label}__{tag}.stderr.log").write_text(result["stderr"])
        log(
            f"    {tag}: wall={result['wall_s']:.2f}s "
            f"ttft={'%.2f' % result['ttft_s'] if result['ttft_s'] else '–'}s "
            f"chars={len(result['stdout'])} rc={result['rc']}"
        )
        if is_warmup:
            continue
        tokens = estimate_tokens(result["stdout"])
        wall_s = max(result["wall_s"], 1e-6)
        # `streamed` = first byte arrived meaningfully before the last byte.
        # When false (e.g. pi -p), tok/s_streaming is meaningless, so we report
        # only end-to-end throughput.
        streamed = bool(
            result["ttft_s"] is not None
            and result.get("last_byte_s") is not None
            and (result["last_byte_s"] - result["ttft_s"]) > 0.5
        )
        text = result["stdout"].lower()
        iters.append(
            {
                "iter": i - warmup,
                "wall_s": round(result["wall_s"], 3),
                "ttft_s": round(result["ttft_s"], 3) if result["ttft_s"] else None,
                "output_chars": len(result["stdout"]),
                "output_tokens_est": tokens,
                "throughput_tok_per_s_est": round(tokens / wall_s, 2),
                "streamed": streamed,
                "rc": result["rc"],
                "output_file": out_path.name,
                "looks_like_html": ("<html" in text) or ("<!doctype" in text),
                "has_button": "<button" in text,
                "has_script": "<script" in text,
                "end_reason": result.get("end_reason"),
            }
        )
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
    restore_opencode_config()
    restore_opencode_notools_agent()
    inst = state.get("installed_by_us", {})
    pulled = state.get("model_pulled_by_us", {})
    for name in (pulled.get("ollama") or []):
        if which("ollama"):
            run(["ollama", "rm", name], check=False)
    for name in (pulled.get("lmstudio") or []):
        if which("lms"):
            run(["lms", "rm", name, "-y"], check=False)
    Ollama.stop()
    if inst.get("pi"):
        if which("npm"):
            run("npm uninstall -g @mariozechner/pi-coding-agent", check=False)
    if inst.get("opencode") and which("opencode"):
        run(["opencode", "uninstall"], check=False)
    if inst.get("ollama") and IS_MAC and which("brew"):
        run("brew uninstall ollama", check=False)
    if inst.get("lmstudio") and IS_MAC and which("brew"):
        run("brew uninstall --cask lm-studio", check=False)
    STATE_FILE.unlink(missing_ok=True)
    log("Cleanup complete.")


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "bench.config.json"))
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--cleanup-only", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
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
                          else False) for b in requested_backends},
        "ollama_models":   {m["id"]: have_ollama_model(m["ollama"])
                             for m in models if "ollama" in m},
        "lmstudio_models": {m["id"]: have_lmstudio_model(m["lmstudio"])
                             for m in models if "lmstudio" in m},
    }
    state["pre_existing"] = pre
    save_state(state)
    log(f"Pre-flight: {pre}")

    if not args.skip_install:
        if "ollama" in requested_backends:
            install_ollama(state); save_state(state)
        if "lmstudio" in requested_backends:
            install_lmstudio(state); save_state(state)
        if "pi" in requested_agents:
            install_pi(state); save_state(state)
        if "opencode" in requested_agents:
            install_opencode(state); save_state(state)

    if "ollama" in requested_backends:
        Ollama.start()
    if "lmstudio" in requested_backends and have_lmstudio():
        LMStudio.start()

    # Pull each requested model from each requested backend.
    for m in models:
        if "ollama" in requested_backends and "ollama" in m:
            if not have_ollama_model(m["ollama"]):
                log(f"Pulling {m['ollama']} via ollama…")
                run(["ollama", "pull", m["ollama"]])
                state["model_pulled_by_us"].setdefault("ollama", []).append(m["ollama"]) \
                    if isinstance(state["model_pulled_by_us"].get("ollama"), list) \
                    else state["model_pulled_by_us"].update({"ollama": [m["ollama"]]})
                save_state(state)
        if "lmstudio" in requested_backends and "lmstudio" in m and have_lmstudio():
            if not have_lmstudio_model(m["lmstudio"]):
                log(f"Downloading {m['lmstudio']} via lms…")
                rc, _, _ = run(["lms", "get", m["lmstudio"], "-y"], check=False)
                if rc == 0:
                    state["model_pulled_by_us"].setdefault("lmstudio", []).append(m["lmstudio"]) \
                        if isinstance(state["model_pulled_by_us"].get("lmstudio"), list) \
                        else state["model_pulled_by_us"].update({"lmstudio": [m["lmstudio"]]})
                    save_state(state)
                else:
                    warn(f"`lms get {m['lmstudio']}` failed — load it manually in LM Studio.")

    if "lmstudio" in requested_backends and have_lmstudio() and which("lms"):
        for m in models:
            if "lmstudio" in m:
                run(["lms", "load", m["lmstudio"], "-y"], check=False)

    # Configure opencode with all model aliases per backend.
    if "opencode" in requested_agents:
        write_opencode_config(
            ollama_models=[m["ollama"] for m in models if "ollama" in m] if "ollama" in requested_backends else [],
            lmstudio_models=[m["lmstudio"] for m in models if "lmstudio" in m] if "lmstudio" in requested_backends else [],
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

    # ---- Build the matrix ----
    planned = []
    for agent in requested_agents:
        if not AGENTS[agent]["is_installed"]():
            warn(f"Skipping agent '{agent}' — not installed.")
            continue
        for backend in requested_backends:
            if backend not in AGENTS[agent]["supports_backends"]:
                warn(f"Skipping {agent}+{backend} — agent doesn't support that backend.")
                continue
            backend_up = (have_ollama() if backend == "ollama" else have_lmstudio())
            if not backend_up:
                warn(f"Skipping {agent}+{backend} — backend not available.")
                continue
            for m in models:
                if backend not in m:
                    warn(f"Skipping {agent}+{backend}+{m['id']} — no '{backend}' alias for model.")
                    continue
                model_alias = m[backend]
                label = f"{agent}+{backend}+{m['id']}"
                cmd, extra_env = AGENTS[agent]["build_cmd"](model_alias, backend, prompt)
                env = os.environ.copy()
                env.update(extra_env)
                planned.append({
                    "label": label, "agent": agent, "backend": backend,
                    "model_id": m["id"], "model_alias": model_alias,
                    "cmd": cmd, "env": env,
                })

    if not planned:
        err("No combos to run after filtering. Check config + installed agents/backends/models.")
        sys.exit(1)

    total_runs = len(planned) * (warmup + n_iter)
    progress = {"done": 0, "total": total_runs, "start": time.monotonic()}
    log(f"Plan: {len(planned)} combos × {warmup + n_iter} runs = {total_runs} total")

    for p in planned:
        result = bench_one(p["label"], p["cmd"], p["env"], run_dir, n_iter, warmup, progress)
        # Tag combo with its dimensions for the viewer.
        result.update({"agent": p["agent"], "backend": p["backend"], "model_id": p["model_id"]})
        combos.append(result)
    results = {
        "timestamp": ts,
        "platform": platform.platform(),
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
    restore_opencode_config()
    restore_opencode_notools_agent()
    log("Restored opencode config (notools agent removed, original config restored if any).")

    print()
    answer = input("Uninstall apps and remove downloaded models? [y/N] ").strip().lower()
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
