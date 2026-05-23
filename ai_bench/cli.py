"""CLI argument parsing and top-level orchestration for AI-Bench."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

from ai_bench.agents import (
    AGENTS,
    restore_opencode_config,
    restore_opencode_notools_agent,
    restore_pi_models_config,
    write_opencode_config,
    write_opencode_notools_agent,
    write_pi_models_config,
)
from ai_bench.backends import Ollama, LMStudio, OMLX
from ai_bench.config import RESULTS_DIR, _DEFAULT_PROMPT, validate_config
from ai_bench.installers import (
    BACKEND_START,
    BACKEND_STOP,
    OMLX_CLONE_DIR,
    OMLX_VENV_DIR,
    cleanup,
    have_lmstudio,
    have_ollama,
    have_omlx,
    install_lmstudio,
    install_ollama,
    install_opencode,
    install_pi,
)
from ai_bench.models import (
    _lmstudio_resolve_api_id,
    download_omlx_model,
    have_lmstudio_model,
    have_ollama_model,
    have_omlx_model,
)
from ai_bench.picker import model_picker
from ai_bench.runner import bench_one, run_agent_streamed, run_backend_direct_ollama
from ai_bench.results import cpu_info, estimate_tokens, summarize
from ai_bench.state import STATE_FILE, load_state, save_state
from ai_bench.log import err, log, warn


def _write_run_viewer(root, run_dir, results):
    """Copy viewer.html into a run directory with embedded initial results."""
    viewer_src = root / "viewer.html"
    viewer_dst = run_dir / "viewer.html"
    marker = '<script id="embedded-results-json" type="application/json"></script>'
    embedded = (
        '<script id="embedded-results-json" type="application/json">'
        + json.dumps(results).replace("</", "<\\/")
        + "</script>"
    )
    html = viewer_src.read_text()
    if marker in html:
        html = html.replace(marker, embedded, 1)
    viewer_dst.write_text(html)
    return viewer_dst


def _preflight_failure_result(label, run_dir, n_iter, reason, validators=None):
    """Build a combo result when a backend/model preflight fails before warmup."""
    stderr_path = run_dir / f"{label}__preflight.stderr.log"
    stderr_path.write_text(reason)
    validators_cfg = validators or {}
    iters = []
    for i in range(n_iter):
        out_path = run_dir / f"{label}__iter-{i}.txt"
        out_path.write_text("")
        iters.append({
            "iter": i,
            "wall_s": 0.0,
            "ttft_s": None,
            "output_chars": 0,
            "output_tokens_est": 0,
            "throughput_tok_per_s_est": 0.0,
            "streamed": False,
            "rc": 1,
            "output_file": out_path.name,
            "stderr_file": stderr_path.name,
            "looks_like_html": False if validators_cfg.get("html", True) else None,
            "has_button": False if validators_cfg.get("button", True) else None,
            "has_script": False if validators_cfg.get("script", True) else None,
            "end_reason": "preflight_failed",
        })
    return summarize(label, iters)


def _stop_other_backends(target_backend):
    for backend, stop in BACKEND_STOP.items():
        if backend != target_backend:
            stop()


def _launch_viewer(viewer_path, opener=webbrowser.open):
    """Open the generated static viewer in the user's default browser."""
    viewer_url = Path(viewer_path).resolve().as_uri()
    try:
        opened = opener(viewer_url)
    except Exception as exc:
        warn(f"Could not launch viewer automatically: {exc}")
        return False
    if not opened:
        warn(f"Could not launch viewer automatically. Open manually: {viewer_path}")
        return False
    log(f"Launched viewer: {viewer_path}")
    return True


# ---- lifecycle context -------------------------------------------------------
class RunContext:
    """Track which user-side mutations this benchmark run performed so the
    finally block can restore only what was actually touched.

    Usage:
        ctx = RunContext()
        try:
            ctx.pi_config_modified = True   # after write_pi_models_config
            ctx.backend_start("ollama")
            ...
        finally:
            ctx.restore()
    """

    def __init__(self):
        self.pi_config_modified = False
        self.opencode_config_modified = False
        self.opencode_notools_written = False
        self.lmstudio_guardrails_relaxed = False
        self._started_backends: list[str] = []
        self.current_backend: str | None = None

    def backend_start(self, name):
        self.current_backend = name
        if name not in self._started_backends:
            self._started_backends.append(name)

    def backend_stop(self, name):
        if self.current_backend == name:
            self.current_backend = None
        if name in self._started_backends:
            self._started_backends.remove(name)

    def restore(self):
        """Restore everything this run modified. Idempotent."""
        if self.pi_config_modified:
            restore_pi_models_config()
        if self.opencode_config_modified:
            restore_opencode_config()
        if self.opencode_notools_written:
            restore_opencode_notools_agent()
        if self.lmstudio_guardrails_relaxed:
            LMStudio.restore_guardrails()
        for backend in list(self._started_backends):
            BACKEND_STOP.get(backend, lambda: None)()
            self.backend_stop(backend)
        self.pi_config_modified = False
        self.opencode_config_modified = False
        self.opencode_notools_written = False
        self.lmstudio_guardrails_relaxed = False


# ---- main ------------------------------------------------------------------
def main():
    ROOT = Path(__file__).resolve().parent.parent
    ctx = RunContext()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Config file to use (skips interactive picker)")
    ap.add_argument("--configure", action="store_true",
                    help="Open the interactive model picker even if a config already exists")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--cleanup-only", action="store_true")
    ap.add_argument("--no-open-viewer", action="store_true",
                    help="Do not automatically open the run-local results viewer when the benchmark finishes")
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
        cfg = model_picker(cfg_path, force=args.configure)
    else:
        cfg = json.loads(cfg_path.read_text())
    state = load_state()

    if args.cleanup_only:
        cleanup(state, cfg)
        return

    # Validate config
    errors = validate_config(cfg)
    if errors:
        err("Invalid config:")
        for error in errors:
            err(f"  - {error}")
        sys.exit(2)

    requested_agents   = list(cfg["agents"])
    requested_backends = list(cfg["backends"])
    models             = list(cfg["models"])

    pre = {
        "agents":   {a: AGENTS[a]["is_installed"]() for a in requested_agents},
        "backends": {b: b in ("ollama", "lmstudio", "omlx") and
                      {"ollama": have_ollama, "lmstudio": have_lmstudio, "omlx": have_omlx}[b]()
                     for b in requested_backends},
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

    try:
        if not args.skip_install:
            if "ollama" in requested_backends:
                install_ollama(state); save_state(state)
            if "lmstudio" in requested_backends:
                install_lmstudio(state); save_state(state)
            if "omlx" in requested_backends:
                from ai_bench.installers import install_omlx
                install_omlx(state); save_state(state)
            if "pi" in requested_agents:
                install_pi(state); save_state(state)
            if "opencode" in requested_agents:
                install_opencode(state); save_state(state)

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

        if "ollama" in requested_backends:
            Ollama.start()
            ctx.backend_start("ollama")
            for m in models:
                if "ollama" in m and not have_ollama_model(m["ollama"]):
                    log(f"Pulling {m['ollama']} via ollama…")
                    subprocess.run(["ollama", "pull", m["ollama"]], check=True)
                    state["model_pulled_by_us"].setdefault("ollama", []).append(m["ollama"])
                    save_state(state)
            Ollama.stop()
            ctx.backend_stop("ollama")
            time.sleep(2)

        if "lmstudio" in requested_backends and have_lmstudio():
            LMStudio.relax_guardrails()
            ctx.lmstudio_guardrails_relaxed = True
            LMStudio.start()
            ctx.backend_start("lmstudio")
            for m in models:
                if "lmstudio" not in m or have_lmstudio_model(m["lmstudio"]):
                    continue
                alias = m["lmstudio"]
                fmt_flag = []
                low = alias.lower()
                if "-mlx" in low or "mlx-" in low:
                    fmt_flag = ["--mlx"]
                elif "gguf" in low:
                    fmt_flag = ["--gguf"]

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
                        break

                if rc == 0 and have_lmstudio_model(alias):
                    state["model_pulled_by_us"].setdefault("lmstudio", []).append(alias)
                    save_state(state)
                elif rc == 0:
                    warn(f"`lms get {alias}` returned rc=0 but model not detected on disk — load it manually in LM Studio.")
                else:
                    warn(f"`lms get {alias}` failed (rc={rc}) — load it manually in LM Studio.")
                    if last_stderr.strip():
                        warn(last_stderr.strip()[:500])

            for m in models:
                if "lmstudio" in m and have_lmstudio_model(m["lmstudio"]):
                    api_id = _lmstudio_resolve_api_id(m["lmstudio"])
                    if api_id:
                        if api_id != m["lmstudio"]:
                            log(f"LM Studio API ID for {m['lmstudio']} → {api_id}")
                        m["lmstudio_api_id"] = api_id

            LMStudio.stop()
            ctx.backend_stop("lmstudio")
            time.sleep(2)

        _ollama_models = [m["ollama"] for m in models if "ollama" in m] if "ollama" in requested_backends else []
        _lmstudio_models = [m.get("lmstudio_api_id") or m["lmstudio"] for m in models if "lmstudio" in m] if "lmstudio" in requested_backends else []
        _omlx_models = [m["omlx"] for m in models if "omlx" in m] if "omlx" in requested_backends else []

        if "pi" in requested_agents:
            write_pi_models_config(_ollama_models, _lmstudio_models, _omlx_models)
            ctx.pi_config_modified = True
        if "opencode" in requested_agents:
            write_opencode_config(_ollama_models, _lmstudio_models, _omlx_models)
            ctx.opencode_config_modified = True
            write_opencode_notools_agent()
            ctx.opencode_notools_written = True

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RESULTS_DIR / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"Results → {run_dir}")
        (run_dir / "prompt.txt").write_text(cfg["prompt"])

        prompt = cfg["prompt"]
        n_iter, warmup = cfg["iterations"], cfg["warmup"]
        timeout_s = cfg.get("timeout_s", 900)
        no_output_timeout_s = cfg.get("no_output_timeout_s", min(360, timeout_s))
        combos = []
        skipped = []

        planned = []
        for backend in requested_backends:
            backend_up = {"ollama": have_ollama, "lmstudio": have_lmstudio, "omlx": have_omlx}.get(backend, lambda: False)()
            if not backend_up:
                warn(f"Skipping all {backend} combos — backend not available.")
                for agent in requested_agents:
                    for m in models:
                        skipped.append({"agent": agent, "backend": backend, "model_id": m["id"], "reason": f"backend {backend} not available"})
                continue
            for agent in requested_agents:
                if not AGENTS[agent]["is_installed"]():
                    continue
                if backend not in AGENTS[agent]["supports_backends"]:
                    warn(f"Skipping {agent}+{backend} — agent doesn't support that backend.")
                    for m in models:
                        skipped.append({"agent": agent, "backend": backend, "model_id": m["id"], "reason": f"agent {agent} doesn't support backend {backend}"})
                    continue
                for m in models:
                    if backend not in m:
                        warn(f"Skipping {agent}+{backend}+{m['id']} — no '{backend}' alias for model.")
                        skipped.append({"agent": agent, "backend": backend, "model_id": m["id"], "reason": f"no {backend} alias for model {m['id']}"})
                        continue
                    detect_alias = m[backend]
                    runtime_alias = (m.get("lmstudio_api_id") or m[backend]) if backend == "lmstudio" else m[backend]
                    model_present = {
                        "ollama": lambda: have_ollama_model(detect_alias),
                        "lmstudio": lambda: have_lmstudio_model(detect_alias),
                        "omlx": lambda: have_omlx_model(detect_alias),
                    }.get(backend, lambda: False)()
                    if not model_present:
                        warn(f"Skipping {agent}+{backend}+{m['id']} — model '{detect_alias}' not present on {backend} (download likely failed).")
                        skipped.append({"agent": agent, "backend": backend, "model_id": m["id"], "reason": f"model {detect_alias} not present on {backend}"})
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

        for agent in requested_agents:
            if not AGENTS[agent]["is_installed"]():
                warn(f"Skipping agent '{agent}' — not installed.")

        if not planned:
            err("No combos to run after filtering. Check config + installed agents/backends/models.")
            sys.exit(1)

        total_runs = len(planned) * (warmup + n_iter)
        progress = {"done": 0, "total": total_runs, "start": time.monotonic()}
        log(f"Plan: {len(planned)} combos × {warmup + n_iter} runs = {total_runs} total")

        current_backend = None
        current_lmstudio_model = None
        omlx_preflight = {}
        for p in planned:
            if p["backend"] != current_backend:
                if current_backend is not None:
                    log(f"Stopping {current_backend} backend…")
                    BACKEND_STOP[current_backend]()
                    ctx.backend_stop(current_backend)
                    time.sleep(2)
                log(f"Stopping other model backends before {p['backend']}…")
                _stop_other_backends(p["backend"])
                log(f"Starting {p['backend']} backend…")
                BACKEND_START[p["backend"]]()
                ctx.backend_start(p["backend"])
                current_backend = p["backend"]
                current_lmstudio_model = None

            if p["backend"] == "omlx":
                preflight_key = p["model_alias"]
                if preflight_key not in omlx_preflight:
                    log(f"Preflighting oMLX model {p['model_alias']}…")
                    omlx_preflight[preflight_key] = OMLX.check_model_load(p["model_alias"])
                ok, reason = omlx_preflight[preflight_key]
                if not ok:
                    warn(reason)
                    if progress is not None:
                        progress["done"] += warmup + n_iter
                    result = _preflight_failure_result(
                        p["label"], run_dir, n_iter, reason,
                        validators=cfg.get("validators"),
                    )
                    result.update({
                        "agent": p["agent"], "backend": p["backend"],
                        "model_id": p["model_id"], "model_alias": p["model_alias"],
                        "timeout_s": timeout_s,
                        "no_output_timeout_s": no_output_timeout_s,
                    })
                    if p["cmd"]:
                        result["cmd"] = p["cmd"]
                    combos.append(result)
                    continue

            if p["backend"] == "lmstudio" and p["model_alias"] != current_lmstudio_model:
                if current_lmstudio_model is not None:
                    log("Unloading all LM Studio models…")
                    subprocess.run(["lms", "unload", "--all"], check=False)
                    time.sleep(3)
                log(f"Loading LM Studio model {p['model_alias']}…")
                subprocess.run(["lms", "load", p["model_alias"], "-y"], check=False)
                current_lmstudio_model = p["model_alias"]

            result = bench_one(
                p["label"], p["cmd"], p["env"], run_dir, n_iter, warmup,
                progress, direct=p.get("direct"), total_timeout=timeout_s,
                validators=cfg.get("validators"),
                no_output_timeout=no_output_timeout_s if p["cmd"] else None,
            )
            result.update({
                "agent": p["agent"], "backend": p["backend"],
                "model_id": p["model_id"], "model_alias": p["model_alias"],
                "timeout_s": timeout_s,
                "no_output_timeout_s": no_output_timeout_s,
            })
            if p["cmd"]:
                result["cmd"] = p["cmd"]
            combos.append(result)

        if current_backend is not None:
            BACKEND_STOP[current_backend]()
            ctx.backend_stop(current_backend)

        results = {
            "schema_version": 1,
            "timestamp": ts,
            "platform": platform.platform(),
            "cpu": cpu_info(),
            "machine": platform.machine(),
            "config": cfg,
            "pre_existing": pre,
            "combos": combos,
            "skipped": skipped,
        }
        out_file = run_dir / "results.json"
        out_file.write_text(json.dumps(results, indent=2))
        run_viewer = _write_run_viewer(ROOT, run_dir, results)
        log(f"Wrote {out_file}")
        log(f"View: open {run_viewer}")

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

        if not args.no_open_viewer:
            _launch_viewer(run_viewer)

        ctx.restore()

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
    finally:
        ctx.restore()
