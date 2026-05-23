#!/usr/bin/env python3
"""
agent-bench: benchmark coding agents (pi, opencode) against local LLM
backends (LM Studio, Ollama) end-to-end.

Thin entrypoint — delegates to ai_bench.cli.main().

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
                   [--cleanup-only] [--no-open-viewer]
"""

import sys

from ai_bench.cli import RunContext, main
from ai_bench.log import err


# ---- backward-compat re-exports for tests and external scripts ---------------
from ai_bench.agents import AGENTS, _restore_backed_up, restore_opencode_config, restore_opencode_notools_agent, restore_pi_models_config, write_opencode_config, write_opencode_notools_agent, write_pi_models_config  # noqa: F401
from ai_bench.backends import LMStudio, OMLX, Ollama  # noqa: F401
from ai_bench.config import VALID_AGENTS, VALID_BACKENDS, _DEFAULT_PROMPT, validate_config  # noqa: F401
from ai_bench.installers import BACKEND_START, BACKEND_STOP, _get_artifacts_to_cleanup, cleanup, have_lmstudio, have_ollama, have_omlx  # noqa: F401
from ai_bench.models import _lmstudio_dir_complete, _lmstudio_resolve_api_id, download_omlx_model, have_lmstudio_model, have_ollama_model, have_omlx_model  # noqa: F401
from ai_bench.picker import _normalize_label  # noqa: F401
from ai_bench.runner import bench_one, run_agent_streamed, run_backend_direct_ollama  # noqa: F401
from ai_bench.results import cpu_info, estimate_tokens, summarize  # noqa: F401
from ai_bench.state import STATE_FILE, load_state, save_state  # noqa: F401


if __name__ == "__main__":
    ctx = RunContext()
    try:
        main()
    except KeyboardInterrupt:
        err("Interrupted.")
        ctx.restore()
        sys.exit(130)
    except Exception:
        ctx.restore()
        raise
    finally:
        ctx.restore()
