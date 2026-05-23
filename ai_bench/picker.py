"""Interactive model picker utilities."""

import json
import subprocess
import sys

from ai_bench.agents import AGENTS
from ai_bench.backends import which
from ai_bench.config import _DEFAULT_PROMPT
from ai_bench.installers import have_omlx
from ai_bench.models import (
    _list_ollama_installed, _search_ollama,
    _list_lmstudio_installed, _search_lmstudio_online,
    _list_omlx_installed, _search_hf_mlx,
    _fmt_size,
)

_LABEL_SUFFIXES = (
    "-mlx-bf16", "-mlx-4bit", "-mlx-8bit", "-mlx",
    "-gguf", "-q4_k_m", "-q4_k_s", "-q5_k_m", "-q5_k_s", "-q8_0",
    "-4bit", "-8bit", "-bf16", "-fp16",
)


def _normalize_label(model_id):
    s = model_id.split("/")[-1].lower()
    for sep in (":", "_", " "):
        s = s.replace(sep, "-")
    changed = True
    while changed:
        changed = False
        for suf in _LABEL_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return s


def _fzf_available():
    return bool(which("fzf"))


def _fzf_select(items, header=""):
    """Use fzf to interactively filter and multi-select from items.
    Returns list of chosen items. Tab selects, Enter confirms."""
    lines = []
    for m in items:
        size = f"\t{m['size']}" if m.get("size") else ""
        desc = f"  {m.get('desc', '')}" if m.get("desc") else ""
        lines.append(f"{m['id']}{size}{desc}")
    fzf_input = "\n".join(lines)
    cmd = [
        "fzf",
        "--multi",
        "--prompt", "  filter> ",
        "--height", "~40%",
        "--border", "rounded",
        "--info", "inline",
        "--bind", "space:toggle+down",
        "--header", header or "Space to select, Enter to confirm",
    ]
    r = subprocess.run(cmd, input=fzf_input, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    selected_ids = {line.split("\t")[0].strip() for line in r.stdout.strip().splitlines()}
    return [m for m in items if m["id"] in selected_ids]


def _display_list(items):
    for i, m in enumerate(items, 1):
        size = f"  {m['size']}" if m.get("size") else ""
        desc = f"  — {m.get('desc', '')}" if m.get("desc") else ""
        print(f"  {i:2}. {m['id']}{size}{desc}")


def _pick_indices(items, prompt="  Add (numbers, blank to skip): "):
    """Numbered fallback picker when fzf is not available."""
    if not items:
        return []
    raw = input(prompt).strip()
    chosen = []
    for tok in raw.split():
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(items):
                chosen.append(items[idx])
    return chosen


def _select_from(items, header=""):
    """Select one or more items from a list. Uses fzf if available, numbered list otherwise."""
    if not items:
        return []
    if _fzf_available():
        return _fzf_select(items, header)
    _display_list(items)
    return _pick_indices(items)


def _pick_backend_models(backend_label, search_fn=None, list_fn=None):
    """Interactive picker for one backend. Returns list of {id, [hf_id], size} dicts.

    list_fn  — callable returning [{id, size, ...}] of locally installed models
    search_fn — callable(term) returning [{id, size, ...}] from remote search
    Either or both may be provided. LM Studio has only list_fn; oMLX has only
    search_fn; Ollama has both (installed first, then optional remote search).
    """
    print(f"\n── {backend_label} {'─' * max(0, 50 - len(backend_label))}")
    if not _fzf_available():
        print("  (install fzf for live filtering: brew install fzf)")

    selected = []

    # Show installed models first (Ollama + LM Studio)
    if list_fn:
        installed = list_fn()
        if installed:
            chosen = _select_from(installed, f"{backend_label} — installed models (Space=multi-select, Enter=confirm)")
            selected.extend(chosen)
        else:
            print("  No models installed locally.")

    if search_fn and not selected:
        while True:
            term = input("  Search online (blank to skip): ").strip()
            if not term:
                break
            print("  Fetching…", end="", flush=True)
            results = search_fn(term)
            print()
            if not results:
                print("  No results — try a different term.")
                continue
            chosen = _select_from(results, f"{backend_label} — {term} (Space=multi-select, Enter=confirm)")
            selected.extend(chosen)
            if chosen and input("  Search again? [y/N] ").strip().lower() not in ("y", "yes"):
                break

    return selected


def model_picker(cfg_path, force=False):
    """Interactive model picker. Returns a config dict and saves it to cfg_path."""
    print()

    # Offer to reuse existing config (unless --configure forced the picker open)
    if not force and cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
            if existing.get("models"):
                ids = [m["id"] for m in existing["models"]]
                n = existing.get("iterations", 1)
                w = existing.get("warmup", 0)
                print(f"Last config: {', '.join(ids)}")
                print(f"  backends: {', '.join(existing.get('backends', []))}  |  {n} iter(s), {w} warmup")
                print("  (run with --configure to change models)")
                ans = input("Reuse last config? [Y/n] ").strip().lower()
                if ans not in ("n", "no"):
                    return existing
        except Exception:
            pass

    print("\nPick models to benchmark per backend.")
    print("Models with the same label are compared side-by-side in the viewer.\n")

    # Collect selections per backend
    raw = {}  # backend -> list of {id, [hf_id], size}

    raw["ollama"]   = _pick_backend_models("Ollama",             search_fn=_search_ollama,   list_fn=_list_ollama_installed)
    raw["lmstudio"] = _pick_backend_models("LM Studio",          search_fn=_search_lmstudio_online, list_fn=_list_lmstudio_installed)
    raw["omlx"]     = _pick_backend_models("oMLX / HuggingFace", search_fn=_search_hf_mlx, list_fn=_list_omlx_installed)

    # Build model entries. Each selected item gets a label; same label = grouped.
    # We accumulate into a dict keyed by label.
    model_map = {}  # label -> entry dict

    for backend, items in raw.items():
        if not items:
            continue
        for item in items:
            label = _normalize_label(item["id"])
            if label not in model_map:
                model_map[label] = {"id": label}
            entry = model_map[label]
            if backend == "omlx":
                entry["omlx"]    = item["id"]
                entry["omlx_hf"] = item.get("hf_id", f"mlx-community/{item['id']}")
            else:
                entry[backend] = item["id"]

    models = list(model_map.values())

    if not models:
        print("\nNo models selected.")
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())
        sys.exit(1)

    print("\n── Config summary " + "─" * 34)
    for m in models:
        parts = [f"{k}={v}" for k, v in m.items() if k not in ("id", "omlx_hf")]
        print(f"  {m['id']}: {', '.join(parts)}")

    # Other settings
    print()
    print("\n── Agents " + "─" * 41)
    agent_items = [{"id": name, "desc": meta["desc"]} for name, meta in AGENTS.items()]
    chosen_agents = _select_from(agent_items, "Agents to benchmark (Space=multi-select, Enter=confirm)")
    agents = [a["id"] for a in chosen_agents] if chosen_agents else ["pi"]

    n_str = input("Iterations per combo [3]: ").strip()
    n_iter = int(n_str) if n_str.isdigit() else 3

    w_str = input("Warmup runs [1]: ").strip()
    warmup = int(w_str) if w_str.isdigit() else 1

    all_backends = {k for m in models for k in m if k not in ("id", "omlx_hf")}
    backends = [b for b in ("ollama", "lmstudio", "omlx") if b in all_backends]

    prompt = _DEFAULT_PROMPT
    if cfg_path.exists():
        try:
            prompt = json.loads(cfg_path.read_text()).get("prompt", _DEFAULT_PROMPT)
        except Exception:
            pass

    cfg = {
        "models": models,
        "agents": agents,
        "backends": backends,
        "iterations": n_iter,
        "warmup": warmup,
        "prompt": prompt,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"\nSaved → {cfg_path.name}")
    return cfg
