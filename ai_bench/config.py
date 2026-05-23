"""Configuration constants, loaders, and validation for ai-bench."""

import json
import os
import platform
import argparse
from pathlib import Path

from ai_bench.agents import AGENTS

ROOT = Path(__file__).resolve().parent.parent

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

_DEFAULT_PROMPT = (
    "Build a single-page website in plain HTML, CSS, and JavaScript with two buttons. "
    "The first button fetches a random joke from https://icanhazdadjoke.com/ (send header "
    "'Accept: application/json') and displays it on the page. The second button toggles dark "
    "mode by adding/removing a 'dark' CSS class on the body, with appropriate styles for both "
    "modes. Output a single complete HTML file with inline <style> and <script> tags. No build "
    "tools, no frameworks. Output only the HTML, no commentary. Ask no questions and make any "
    "required assumptions yourself."
)

# Valid backends
VALID_BACKENDS = frozenset({"ollama", "lmstudio", "omlx"})

# Valid agents
VALID_AGENTS = frozenset(AGENTS.keys())


def validate_config(cfg):
    """Validate config and return list of error strings. Returns empty list if valid."""
    errors = []

    # models must be a non-empty list
    if "models" not in cfg:
        errors.append("missing required field: models")
    elif not isinstance(cfg["models"], list):
        errors.append("models must be a list")
    elif len(cfg["models"]) == 0:
        errors.append("models must be a non-empty list")
    else:
        seen_model_ids = set()
        for i, model in enumerate(cfg["models"]):
            if not isinstance(model, dict):
                errors.append(f"model[{i}] must be a dict")
                continue

            # each model has a non-empty string id
            if "id" not in model:
                errors.append(f"model[{i}] missing required field: id")
            elif not isinstance(model["id"], str):
                errors.append(f"model[{i}].id must be a string")
            elif len(model["id"]) == 0:
                errors.append(f"model[{i}].id must be non-empty")
            elif model["id"] in seen_model_ids:
                errors.append(f"model[{i}].id must be unique: {model['id']}")
            else:
                seen_model_ids.add(model["id"])

            # each model has at least one backend alias
            backend_aliases = {"ollama", "lmstudio", "omlx"}
            has_backend = any(alias in model for alias in backend_aliases)
            if not has_backend:
                errors.append(f"model[{i}] must have at least one backend alias among {backend_aliases}")

    # agents must be a non-empty list
    if "agents" not in cfg:
        errors.append("missing required field: agents")
    elif not isinstance(cfg["agents"], list):
        errors.append("agents must be a list")
    elif len(cfg["agents"]) == 0:
        errors.append("agents must be a non-empty list")
    else:
        for i, agent in enumerate(cfg["agents"]):
            if agent not in VALID_AGENTS:
                errors.append(f"unknown agent at index {i}: {agent}. Known: {sorted(VALID_AGENTS)}")

    # backends must be a non-empty list of valid backends
    if "backends" not in cfg:
        errors.append("missing required field: backends")
    elif not isinstance(cfg["backends"], list):
        errors.append("backends must be a list")
    elif len(cfg["backends"]) == 0:
        errors.append("backends must be a non-empty list")
    else:
        for i, backend in enumerate(cfg["backends"]):
            if backend not in VALID_BACKENDS:
                errors.append(f"unknown backend at index {i}: {backend}. Valid: {sorted(VALID_BACKENDS)}")

    # iterations must be an integer greater than 0
    if "iterations" not in cfg:
        errors.append("missing required field: iterations")
    elif not isinstance(cfg["iterations"], int):
        errors.append("iterations must be an integer")
    elif cfg["iterations"] <= 0:
        errors.append("iterations must be greater than 0")

    # warmup must be an integer >= 0
    if "warmup" not in cfg:
        errors.append("missing required field: warmup")
    elif not isinstance(cfg["warmup"], int):
        errors.append("warmup must be an integer")
    elif cfg["warmup"] < 0:
        errors.append("warmup must be greater than or equal to 0")

    # prompt must be a non-empty string
    if "prompt" not in cfg:
        errors.append("missing required field: prompt")
    elif not isinstance(cfg["prompt"], str):
        errors.append("prompt must be a string")
    elif len(cfg["prompt"]) == 0:
        errors.append("prompt must be non-empty")

    if "direct" in cfg:
        if not isinstance(cfg["direct"], bool):
            errors.append("direct must be a boolean")
        elif cfg["direct"] and "ollama" not in cfg.get("backends", []):
            errors.append("direct mode requires ollama backend")

    if "direct" in cfg.get("agents", []) and "ollama" not in cfg.get("backends", []):
        errors.append("direct agent requires ollama backend")

    # timeout_s must be a positive integer, defaults to 900 when omitted
    if "timeout_s" in cfg:
        if not isinstance(cfg["timeout_s"], int):
            errors.append("timeout_s must be an integer")
        elif cfg["timeout_s"] <= 0:
            errors.append("timeout_s must be greater than 0")

    # no_output_timeout_s kills silent agent subprocesses before the full timeout.
    if "no_output_timeout_s" in cfg:
        if not isinstance(cfg["no_output_timeout_s"], int):
            errors.append("no_output_timeout_s must be an integer")
        elif cfg["no_output_timeout_s"] <= 0:
            errors.append("no_output_timeout_s must be greater than 0")

    # validators is optional, defaults to all true when omitted
    if "validators" in cfg:
        val = cfg["validators"]
        if not isinstance(val, dict):
            errors.append("validators must be a dict")
        else:
            for key in ("html", "button", "script"):
                if key in val and not isinstance(val[key], bool):
                    errors.append(f"validators.{key} must be a boolean")

    return errors
