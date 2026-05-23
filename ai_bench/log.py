"""Minimal logging helpers for AI-Bench.

This module has NO imports from other ai_bench sub-packages to avoid
circular dependency issues.  Import from here whenever you need log/warn/err
at the top of a file.
"""


def log(msg, prefix="•"):
    print(f"{prefix} {msg}", flush=True)


def warn(msg):
    log(msg, prefix="!")


def err(msg):
    log(msg, prefix="x")
