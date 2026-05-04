# AI-Bench

End-to-end benchmark harness for **local AI coding agents** running against **local LLM backends**.

For any matrix of `agents × backends × models`, `bench.py` runs the same prompt N times per combination, captures wall time, time-to-first-token, output tokens, and the raw model output, then writes a `results.json` and a per-iteration text file you can open to verify the model actually did the task.

A standalone `viewer.html` reads any `results.json` and renders a sortable, color-coded report with column tooltips and bar charts.

## What's measured

For each iteration of each combination:

- **wall time** — total elapsed seconds, launch to last byte of output.
- **time-to-first-token** — seconds from launch to the first non-whitespace stdout byte.
- **estimated output tokens** — `chars / 4` heuristic.
- **throughput tok/s** — tokens ÷ wall time. End-to-end, includes all agent and model overhead.
- **streamed?** — whether the agent emitted output incrementally or buffered it all to the end.
- **valid HTML / has buttons** — quick task-completion checks (configurable via the prompt).

Each iteration's full stdout is saved to disk so you can open the generated artifact (e.g. an HTML page) and confirm it actually works, not just that it parses.

## Requirements

- macOS (primary) or Linux (Ollama-only; LM Studio install is not automated on Linux).
- Python 3.9+.
- Internet access for first-time agent/model downloads (subsequent runs work offline).
- LM Studio support requires Apple Silicon — the script auto-detects and skips on Intel.

The script will offer to install missing pieces (`pi`, `opencode`, `ollama`, LM Studio via Homebrew) and pull missing models. At the end it asks whether to uninstall everything it added.

## Quick start

```bash
git clone https://github.com/burghr/AI-Bench.git
cd AI-Bench
python3 bench.py
```

Skip auto-install if you already have the agents and models:

```bash
python3 bench.py --skip-install
```

Use the smoke config (1 iteration, fastest model) to validate your setup:

```bash
python3 bench.py --config bench.smoke.json --skip-install
```

Uninstall apps and pulled models that the script added:

```bash
python3 bench.py --cleanup-only
```

## Configuration

Everything lives in `bench.config.json`:

```json
{
  "models": [
    { "id": "qwen3-1.7b", "ollama": "qwen3:1.7b", "lmstudio": "qwen/qwen3-1.7b" },
    { "id": "gemma4",     "ollama": "gemma4",     "lmstudio": "google/gemma-4-it" }
  ],
  "agents":   ["pi", "opencode"],
  "backends": ["ollama", "lmstudio"],
  "iterations": 3,
  "warmup": 1,
  "prompt": "Build a single-page website..."
}
```

| field | meaning |
|---|---|
| `models` | List of models to test. Each entry has a human-readable `id` and per-backend aliases (since Ollama and LM Studio name the same model differently). A model is skipped for backends it has no alias for. |
| `agents` | Which agents to invoke. Currently `pi` and `opencode` are registered. |
| `backends` | Which local LLM servers to use. Currently `ollama` and `lmstudio`. |
| `iterations` | Timed runs per combination. |
| `warmup` | Untimed runs per combination, executed first. Hides cold model-load latency from the timed numbers. |
| `prompt` | What to send to every run. Change this to benchmark different use cases (HTML, C++, refactor tasks, etc.). One prompt per run; change between runs as needed. |

The script expands the cartesian product `agents × backends × models` and skips combinations that aren't installable (e.g. an agent that doesn't support a given backend, or a model with no alias for that backend) with a warning.

## The viewer

```bash
open viewer.html
```

Drop a `results/<timestamp>/results.json` into the page (or click to choose). The viewer renders:

- A summary table with `agent / backend / model` columns and ★ winners per metric.
- Per-iteration detail tables.
- The exact prompt and pre-flight inventory for that run.
- Tooltips on every column header explaining what the value means.

The viewer is fully static — no server, no dependencies, just open the file. It works against any `results.json` from any run, current or historical.

## Adding a new agent

Each agent declares its CLI invocation and supported backends in the `AGENTS` registry near the top of `bench.py`. To add an agent:

```python
def _myagent_cmd(model_alias, backend, prompt):
    # backend is "ollama" or "lmstudio"; model_alias is the per-backend model name.
    return (
        ["myagent", "--model", f"{backend}/{model_alias}", "--prompt", prompt],
        {},   # extra environment variables, if any
    )

AGENTS["myagent"] = {
    "supports_backends": ["ollama", "lmstudio"],
    "build_cmd": _myagent_cmd,
    "is_installed": lambda: bool(which("myagent")),
}
```

Then add `"myagent"` to `agents` in the config. The matrix expansion picks it up automatically.

## Troubleshooting

- **pi hangs forever on the first run, with low CPU usage.** Some pi extensions hold stdout open and prevent the process from exiting after `-p` mode finishes. Run `pi list` to see installed extensions, then `pi remove <name>` for any non-default ones, and re-run.
- **`pi --list-models` shows no `ollama/...` entries.** Pi's ollama integration extension isn't installed yet. Run `ollama launch pi --model gemma4 --yes -- --list-models` once — it'll install/update `@ollama/pi-coding-agent` and register the local ollama models. After it exits, plain `pi --list-models` should show them.
- **LM Studio model not found.** LM Studio's API model IDs depend on what's actually loaded — they don't always match HuggingFace paths. Run `curl -s http://127.0.0.1:1234/v1/models` to see exact IDs the API accepts, then update `lmstudio` aliases in `bench.config.json` to match.

## Notes from real-world testing

Findings while building this. The benchmark exists partly *to* surface these:

- **`pi -nt -p`** — pi must run with `-nt` (no tools) for benchmarking, otherwise it sometimes uses its `write` tool to save HTML to disk and only outputs a status message instead of the code.
- **`opencode --pure --agent notools`** — opencode's default `build` agent advertises file-editing tools; small/local models (gemma4, qwen3:1.7b) get confused, emit invalid tool calls, and fall back to greetings or schema-error messages. The bench writes a temporary `~/.config/opencode/agent/notools.md` that disables all tools, then restores the original config at end of run.
- **buffered vs streamed output** — pi's print mode buffers the entire response to the end, so per-iteration time-to-first-token equals wall time. The viewer surfaces this as `streamed: no` so you can interpret the ttft column accordingly.
- **LM Studio on Intel Macs** — not supported by Apple. The script detects architecture and skips LM Studio combinations on Intel hardware automatically.
- **Token counts are estimates** (`chars / 4`). Within a model family the relative comparison is meaningful, but the absolute numbers are not authoritative.

## Project layout

```
AI-Bench/
├── bench.py            # main script (matrix runner, agent registry, install/cleanup)
├── bench.config.json   # default matrix
├── bench.smoke.json    # quick-validation matrix (1 iter, smallest model)
├── viewer.html         # static results viewer
├── results/            # per-run output (gitignored)
└── README.md
```

## License

MIT.
