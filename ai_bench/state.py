"""State management — loading and persisting the .bench-state.json file."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / ".bench-state.json"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "installed_by_us": {},
        "model_pulled_by_us": {"ollama": [], "lmstudio": []},
    }


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2))
