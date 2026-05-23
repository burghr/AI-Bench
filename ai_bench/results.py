"""Benchmark result helpers.

Provides utilities for gathering CPU information, estimating token counts,
and summarising benchmark iteration data.
"""

import platform
import re
import subprocess
from pathlib import Path
from statistics import mean, median, pstdev

IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


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


def estimate_tokens(text):
    if not text:
        return 0
    return max(1, round(len(text) / 4))


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
