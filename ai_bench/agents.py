"""Agent registry, configuration writers, and command builders for bench harness."""

import json
import os
import shutil
from pathlib import Path

from ai_bench.backends import OMLX


# ---- constants ---------------------------------------------------------------
OPENCODE_CFG = Path.home() / ".config/opencode/opencode.json"
OPENCODE_CFG_BAK = OPENCODE_CFG.with_suffix(".json.bench-bak")
OPENCODE_CFG_CREATED = OPENCODE_CFG.with_suffix(".json.bench-created")
OPENCODE_AGENT_DIR = Path.home() / ".config/opencode/agent"
OPENCODE_NOTOOLS_AGENT = OPENCODE_AGENT_DIR / "notools.md"
OPENCODE_NOTOOLS_BAK = OPENCODE_AGENT_DIR / "notools.md.bench-bak"
OPENCODE_NOTOOLS_CREATED = OPENCODE_AGENT_DIR / "notools.md.bench-created"

PI_MODELS_CFG = Path.home() / ".pi" / "agent" / "models.json"
PI_MODELS_BAK = PI_MODELS_CFG.with_suffix(".json.bench-bak")
PI_MODELS_CREATED = PI_MODELS_CFG.with_suffix(".json.bench-created")

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
    elif not PI_MODELS_CFG.exists():
        PI_MODELS_CREATED.write_text("")
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


def _restore_backed_up(original: Path, bak: Path, created: Path | None = None):
    if bak.exists():
        shutil.move(str(bak), str(original))
    elif created and created.exists() and original.exists():
        original.unlink()
    if created and created.exists():
        created.unlink()


def restore_pi_models_config():
    _restore_backed_up(PI_MODELS_CFG, PI_MODELS_BAK, PI_MODELS_CREATED)


def write_opencode_notools_agent():
    OPENCODE_AGENT_DIR.mkdir(parents=True, exist_ok=True)
    if OPENCODE_NOTOOLS_AGENT.exists() and not OPENCODE_NOTOOLS_BAK.exists():
        shutil.copy(OPENCODE_NOTOOLS_AGENT, OPENCODE_NOTOOLS_BAK)
    elif not OPENCODE_NOTOOLS_AGENT.exists():
        OPENCODE_NOTOOLS_CREATED.write_text("")
    OPENCODE_NOTOOLS_AGENT.write_text(NOTOOLS_AGENT_CONTENT)


def restore_opencode_notools_agent():
    _restore_backed_up(OPENCODE_NOTOOLS_AGENT, OPENCODE_NOTOOLS_BAK, OPENCODE_NOTOOLS_CREATED)


def write_opencode_config(ollama_models, lmstudio_models, omlx_models):
    """Write opencode provider config registering all model aliases per backend."""
    OPENCODE_CFG.parent.mkdir(parents=True, exist_ok=True)
    if OPENCODE_CFG.exists() and not OPENCODE_CFG_BAK.exists():
        shutil.copy(OPENCODE_CFG, OPENCODE_CFG_BAK)
    elif not OPENCODE_CFG.exists():
        OPENCODE_CFG_CREATED.write_text("")
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
        omlx_key = OMLX.get_api_key() or "omlx"
        providers["omlx"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "oMLX",
            "options": {"baseURL": "http://localhost:8000/v1", "apiKey": omlx_key},
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
        "is_installed": lambda: bool(shutil.which("pi")),
        "desc": "pi — lightweight agentic CLI; fast, minimal overhead",
    },
    "opencode": {
        "supports_backends": ["ollama", "lmstudio", "omlx"],
        "build_cmd": _opencode_cmd,
        "is_installed": lambda: bool(shutil.which("opencode")),
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


def restore_opencode_config():
    _restore_backed_up(OPENCODE_CFG, OPENCODE_CFG_BAK, OPENCODE_CFG_CREATED)
