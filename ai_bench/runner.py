"""Runner utilities for spawning agents, streaming output, and benchmarking iterations."""

import json
import select
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from ai_bench.log import log
from ai_bench.results import estimate_tokens, summarize


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


def run_agent_streamed(cmd, env, *, total_timeout=900, no_output_timeout=None):
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
    timed_out = False
    no_output_timed_out = False
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
                timed_out = True
                break
            if (
                no_output_timeout is not None
                and first_byte_t is None
                and now - start > no_output_timeout
            ):
                p.kill()
                no_output_timed_out = True
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
            if not timed_out:
                timed_out = True
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
        "end_reason": "no_output_timeout" if no_output_timed_out else ("timeout" if timed_out else "exit"),
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


def bench_one(label, cmd, env, run_dir, n_iter, warmup, progress=None,
              direct=None, total_timeout=900, validators=None, no_output_timeout=None):
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
                    result = run_backend_direct_ollama(direct["model_alias"], direct["prompt"],
                                                       total_timeout=total_timeout)
                else:
                    result = {"rc": 1, "wall_s": 0, "ttft_s": None, "last_byte_s": None,
                              "stdout": "", "stderr": f"direct mode not supported for {direct['backend']}",
                              "end_reason": "exit", "backend_stats": None}
            else:
                result = run_agent_streamed(
                    cmd, env,
                    total_timeout=total_timeout,
                    no_output_timeout=no_output_timeout,
                )
        if progress is not None:
            progress["done"] += 1
        out_path = run_dir / f"{label}__{tag}.txt"
        out_path.write_text(result["stdout"])
        stderr_path = None
        if result["stderr"]:
            stderr_path = run_dir / f"{label}__{tag}.stderr.log"
            stderr_path.write_text(result["stderr"])
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
        validators_cfg = validators or {}
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
            "stderr_file": stderr_path.name if stderr_path else None,
            "looks_like_html": (("<html" in text) or ("<!doctype" in text)) if validators_cfg.get("html", True) else None,
            "has_button": ("<button" in text) if validators_cfg.get("button", True) else None,
            "has_script": ("<script" in text) if validators_cfg.get("script", True) else None,
            "end_reason": result.get("end_reason"),
        }
        if backend_stats:
            iter_row["backend_stats"] = backend_stats
        iters.append(iter_row)
    return summarize(label, iters)
