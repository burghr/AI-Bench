# AI-Bench

End-to-end benchmark harness for **local AI coding agents** running against **local LLM backends**.

Supported backends: **Ollama**, **LM Studio**, **oMLX** (Apple Silicon MLX inference).

Supported agents: **pi**, **opencode**, **direct** (raw HTTP, no agent wrapper).

For any matrix of `agents × backends × models`, `bench.py` runs the same prompt N times per combination, captures wall time, time-to-first-token, output tokens, and the raw model output, then writes a `results.json` and a per-iteration text file you can open to verify the model actually did the task.

A standalone `viewer.html` reads any `results.json` and renders a sortable, color-coded report with column tooltips and bar charts.

## What's measured

For each iteration of each combination:

- **wall time** - total elapsed seconds, launch to last byte of output.
- **time-to-first-token** - seconds from launch to the first non-whitespace stdout byte.
- **estimated output tokens** - `chars / 4` heuristic.
- **throughput tok/s** - tokens ÷ wall time. End-to-end, includes all agent and model overhead.
- **streamed?** - whether the agent emitted output incrementally or buffered it all to the end.
- **valid HTML / has buttons** - quick task-completion checks (configurable via the prompt).

When the `direct` agent is enabled, the bench also captures **precise backend stats** from Ollama's `/api/generate` response: real `eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration`. From these it computes the model's actual **eval tok/s** (raw generation rate) and **prompt tok/s** (input ingestion rate) with no agent overhead. The viewer surfaces these in a "Direct backend stats" section so you can see how much of an agent's wall time is overhead vs. raw model speed.

Each results file also captures the host's **CPU brand string** (e.g. `Apple M2 Pro`, `Intel Core i9-9880H`) so the viewer can label runs precisely.

Each iteration's full stdout is saved to disk so you can open the generated artifact and confirm it actually works.

## Prerequisites

These must be present before running `bench.py`. The script does **not** install them.

| Prerequisite | Why needed | Install |
|---|---|---|
| **macOS** (Apple Silicon for full matrix) | Linux supports Ollama only; LM Studio and oMLX require Apple Silicon | - |
| **Python 3.9+** | Runs `bench.py` | Ships with macOS; or `brew install python` |
| **Homebrew** | Installs `ollama`, `lm-studio`, `fzf` | [brew.sh](https://brew.sh) |
| **Node.js 18+ / npm** | Installs the `pi` coding agent | `brew install node` |
| **git** | Clones the oMLX source repo | Ships with Xcode CLT; or `brew install git` |
| **Internet access** | First-time agent/backend/model downloads | - |

## Cleanup

`bench.py --cleanup-only` removes **only** artifacts this script created. It tracks ownership in `.bench-state.json` and will **never** delete:

- Tools you installed manually (Ollama, LM Studio, oMLX, pi, opencode that were present before the run).
- Models you pulled by hand (they won't appear in `.bench-state.json` so cleanup skips them).
- Config files you wrote yourself - backups created by AI-Bench (`.bench-bak` suffix) are restored; your originals are untouched.
- The `~/.ollama`, `~/.lmstudio`, `~/.pi`, `~/.opencode` directories unless the script can prove it created them.

Before performing destructive cleanup the script prints a summary of what will be removed and asks for confirmation in interactive mode.

Cleanup restores these user config files unconditionally at end-of-run (even without `--cleanup-only`):

- `~/.pi/agent/models.json`
- `~/.config/opencode/opencode.json`
- `~/.config/opencode/agent/notools.md`
- `~/.lmstudio/settings.json` (LM Studio guardrails)

## Installation

`bench.py` auto-installs missing backends and agents. Each tool is installed via its standard method and tracked in `.bench-state.json` so cleanup knows what is safe to remove.

| Tool | Install method |
|---|---|
| **ollama** | `brew install ollama` |
| **LM Studio + `lms` CLI** | `brew install --cask lm-studio` |
| **oMLX** | git clone + Python venv in `~/.local/share/omlx-{src,venv}` |
| **pi** coding agent | `npm install -g @mariozechner/pi-coding-agent` |
| **opencode** | `curl -fsSL https://opencode.ai/install | bash` |
| **fzf** | `brew install fzf` (used by the interactive model picker) |
| LLM models | pulled on demand per backend during the run |

### `--skip-install`

Use `--skip-install` when everything is already present and you just want to run the benchmark. The script still performs pre-flight checks (detecting installed tools and available models). It only skips the install step. If a required tool is missing after skipping install, the relevant combinations are skipped with a warning.

## Quick start

```bash
git clone https://github.com/burghr/AI-Bench.git
cd AI-Bench
python3 bench.py
```

On first run the script installs missing tools, opens the interactive model picker, then runs the benchmark. Subsequent runs offer to reuse the last config so you can skip straight to running.

Skip auto-install if everything is already present:

```bash
python3 bench.py --skip-install
```

Force the model picker open even if a saved config exists:

```bash
python3 bench.py --configure
```

Pin a specific config file (bypasses the picker entirely):

```bash
python3 bench.py --config bench.config.json
```

Skip automatically opening the results viewer when the benchmark finishes:

```bash
python3 bench.py --no-open-viewer
```

Uninstall everything the script added:

```bash
python3 bench.py --cleanup-only
```

## Interactive model picker

When running interactively without a pinned `--config`, the script opens a **per-backend model picker** before starting the benchmark. It requires no prior config knowledge. Just search and select.

### Flow

1. **fzf is installed first** (if not already present) so the picker is always available.
2. For each backend (Ollama, LM Studio, oMLX), the picker shows:
   - **Locally installed models** with sizes. Space to multi-select, Enter to confirm.
   - If nothing is selected, falls through to an **online search** (blank to skip the backend entirely).
3. After all backends, you choose which **agents** to benchmark from a multi-select list with descriptions.
4. You set **iterations** and **warmup** count.
5. The resulting config is saved to `bench.config.json` and shown as a summary before the run starts.

On the next run, the picker offers to **reuse the last config**. Press Enter to skip straight to running with no network access needed.

### Online search sources

| Backend | Search source |
|---|---|
| Ollama | Scrapes `ollama.com/library/<family>` for all variants and sizes |
| LM Studio | Searches HuggingFace for GGUF models (`tags=gguf`) |
| oMLX | Searches HuggingFace `mlx-community` for MLX safetensors models |

Sizes for HuggingFace results are fetched in parallel (8 concurrent requests) so the "Fetching…" step stays fast.

### Agents

| Agent | Description |
|---|---|
| `pi` | Lightweight agentic CLI; fast, minimal overhead |
| `opencode` | Full coding agent with file read/edit/shell tools; higher overhead |
| `direct` | Raw HTTP to the backend API; no agent wrapper - measures pure model speed (Ollama only) |

## Configuration

The picker saves to `bench.config.json`. You can also edit it directly:

```json
{
  "models": [
    {
      "id": "qwen3-1.7b",
      "ollama": "qwen3:1.7b",
      "lmstudio": "lmstudio-community/Qwen3-1.7B-GGUF",
      "omlx": "Qwen3-1.7B-4bit",
      "omlx_hf": "mlx-community/Qwen3-1.7B-4bit"
    }
  ],
  "agents":     ["pi", "opencode", "direct"],
  "backends":   ["ollama", "lmstudio", "omlx"],
  "iterations": 3,
  "warmup":     1,
  "timeout_s":  900,
  "no_output_timeout_s": 360,
  "prompt":     "Build a single-page website..."
}
```

| Field | Meaning |
|---|---|
| `models` | List of models to test. Each entry has a human-readable `id` and per-backend aliases. A model is silently skipped for any backend it has no alias for. Models with the **same `id`** across backends are treated as the same model in the viewer (cross-backend comparison). |
| `models[].ollama` | Tag passed to `ollama run` / `ollama pull`, e.g. `qwen3:1.7b`. |
| `models[].lmstudio` | HuggingFace repo ID passed to `lms get`, e.g. `lmstudio-community/Qwen3-1.7B-GGUF`. Also used as the API model ID - run `curl -s http://127.0.0.1:1234/v1/models` to see exact IDs if the API rejects it. |
| `models[].omlx` | Directory name under `~/.omlx/models/`. |
| `models[].omlx_hf` | HuggingFace repo used to download the oMLX model (defaults to `mlx-community/<omlx>`). |
| `agents` | Agents to run. `pi`, `opencode`, `direct` are built-in. |
| `backends` | Backends to use. `ollama`, `lmstudio`, `omlx`. |
| `iterations` | Timed runs per combination. |
| `warmup` | Untimed runs per combination executed first - hides cold model-load latency. |
| `timeout_s` | Maximum wall-clock seconds for one run. Defaults to `900`. |
| `no_output_timeout_s` | Maximum seconds an agent subprocess may run without producing stdout before it is killed. Defaults to `min(360, timeout_s)`. |
| `prompt` | Prompt sent to every run. Change this to benchmark different task types. |

The script expands the cartesian product `agents × backends × models` and skips invalid combinations (unsupported backend for an agent, missing model alias) with a warning.

### oMLX API key

oMLX requires a HuggingFace token to download models. Set it in `.env` at the repo root (gitignored):

```
HF_TOKEN=hf_...
```

Or export it in your shell before running:

```bash
export HF_TOKEN=hf_...
python3 bench.py
```

## The viewer

```bash
open results/<timestamp>/viewer.html
```

Each run writes a copy of `viewer.html` into its results directory with that run's `results.json` preloaded, then automatically opens it in your default browser when the benchmark finishes. You can still open the run-local viewer manually, drop additional `results.json` files into the page, or click to choose them, to compare runs side-by-side. The viewer renders:

- A summary table with `agent / backend / model` columns and ★ winners per metric.
- Per-iteration detail tables.
- The exact prompt and pre-flight inventory for that run.
- Tooltips on every column header explaining what the value means.

The viewer is fully static. No server, no dependencies, just open the file. The root `viewer.html` also works as an empty comparison viewer if you want to load files manually. It works against any `results.json` from any run, current or historical. Sortable table headers are wired for the summary and comparison views. The viewer handles both the current schema (`schema_version: 1`) and legacy results files transparently, so historical results continue to render.

## Adding a new agent

Each agent declares its CLI invocation and supported backends in `ai_bench/agents.py`:

```python
def _myagent_cmd(model_alias, backend, prompt):
    return (
        ["myagent", "--model", f"{backend}/{model_alias}", "--prompt", prompt],
        {},   # extra env vars, if any
    )

AGENTS["myagent"] = {
    "supports_backends": ["ollama", "lmstudio"],
    "build_cmd": _myagent_cmd,
    "is_installed": lambda: bool(which("myagent")),
    "desc": "myagent - one-line description shown in the picker",
}
```

Then add `"myagent"` to `agents` in the config (or pick it in the interactive picker). The matrix expansion picks it up automatically.

## Troubleshooting

- **pi hangs forever on the first run, with low CPU usage.** Some pi extensions hold stdout open and prevent the process from exiting after `-p` mode finishes. Run `pi list` to see installed extensions, then `pi remove <name>` for any non-default ones, and re-run.
- **`pi --list-models` shows no `ollama/...` entries.** Pi's ollama integration extension isn't installed yet. Run `ollama launch pi --model <any-model> --yes -- --list-models` once. It installs `@ollama/pi-coding-agent` and registers local models. Plain `pi --list-models` should show them afterwards.
- **LM Studio model not found / API rejects the model ID.** LM Studio's API model IDs don't always match the HuggingFace repo name. Run `curl -s http://127.0.0.1:1234/v1/models` to see exact IDs the running server accepts, then update `lmstudio` in `bench.config.json` to match.
- **LM Studio first-time setup.** On a fresh machine the `lms` daemon may not be initialized. The script detects this and opens LM Studio.app once to finish setup, then quits it automatically.
- **oMLX model not found.** The bench downloads oMLX models from HuggingFace using `omlx_hf`. If download fails, check that `HF_TOKEN` is set in `.env` or your shell, or verify the repo ID is correct. `curl -s http://localhost:8000/v1/models` shows what the running server has loaded.
- **oMLX exits with memory errors or hangs before output.** Before running oMLX agent combinations, the bench stops other model backends, restarts the oMLX server to clear stale pinned models, and asks oMLX for a one-token response. Any remaining load failure is recorded as `preflight_failed` instead of spending every iteration on the full timeout. Agent subprocesses also stop after `no_output_timeout_s` seconds without stdout.
- **LM Studio / oMLX not available on Intel Mac.** Both require Apple Silicon. The script detects the architecture and skips those backends automatically on Intel hardware.
- **Config validation errors (exit code 2).** Before any install or backend startup the script validates the config and reports all errors at once. Common issues: missing `models`, `agents`, or `backends` fields; unknown agent or backend names; zero or negative `iterations`; duplicate model `id` values. Fix the reported issues in `bench.config.json` and re-run.
- **`direct` agent only supports Ollama.** The `direct` pseudo-agent hits the backend's HTTP API to measure pure model speed without agent overhead. Currently only Ollama's `/api/generate` endpoint is supported because it returns precise token-level timing stats. LM Studio and oMLX direct modes are planned but not yet implemented.

## Notes from real-world testing

- **`pi -nt -p`** - pi must run with `-nt` (no tools) for benchmarking, otherwise it sometimes uses its `write` tool to save HTML to disk and outputs a status message instead of the code.
- **`opencode --pure --agent notools`** - opencode's default `build` agent advertises file-editing tools; small/local models get confused and emit invalid tool calls. The bench writes a temporary `~/.config/opencode/agent/notools.md` that disables all tools, then restores the original at end of run.
- **buffered vs streamed output** - pi's print mode buffers the entire response, so TTFT equals wall time for pi runs. The viewer surfaces this as `streamed: no` so you can interpret the ttft column accordingly.
- **Token counts are estimates** (`chars / 4`). Relative comparisons within a model family are meaningful; absolute numbers are not authoritative.
- **Backend ordering** - the script runs all combinations for one backend before starting the next, so backends are never running concurrently and don't compete for memory.

## Development

```bash
# Syntax check
python3 -m py_compile bench.py ai_bench/*.py

# Run the test suite (no local backends required)
pytest

# CLI smoke test
python3 bench.py --help
```

Tests use `pytest` and cover config validation, restore helpers, subprocess streaming, summarization, and token estimation. Integration tests against real backends are optional and gated by environment variables (e.g. `AI_BENCH_INTEGRATION_OLLAMA=1`).

## Project layout

```
AI-Bench/
├── bench.py              # thin entrypoint → ai_bench.cli.main()
├── ai_bench/             # modular package
│   ├── agents.py         # agent registry, config writers, command builders
│   ├── backends.py       # Ollama, LM Studio, oMLX lifecycle classes
│   ├── cli.py            # argparse, RunContext, top-level orchestration
│   ├── config.py         # config loading, defaults, validation
│   ├── installers.py     # install/uninstall/detect units, cleanup
│   ├── log.py            # log/warn/err helpers (no internal deps)
│   ├── models.py         # model discovery, downloads, search
│   ├── picker.py         # interactive model/agent selection
│   ├── results.py        # CPU info, token estimates, summarization
│   ├── runner.py         # subprocess streaming, benchmark loop
│   └── state.py          # .bench-state.json persistence
├── bench.config.json     # saved config (written by picker or edited manually)
├── bench.config.full.json  # example multi-model, multi-backend config
├── bench.smoke.json      # minimal config for smoke testing
├── viewer.html           # static results viewer (no server needed)
└── tests/                # pytest test suite
```

Results live under `~/ai-bench/` so `git pull` or re-cloning doesn't clobber your history. Override with `$AGENT_BENCH_RESULTS_DIR`.

## License

MIT.
