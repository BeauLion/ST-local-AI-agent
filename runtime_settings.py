"""
runtime_settings.py — hot-reloadable settings, layered on top of config.py.

config.py stays the source of truth for DEFAULTS and for anything that
requires a process restart to change (model file, ports, Docker image,
launch flags). This module owns the smaller set of values that are safe
to change while the server is running - sampling params, similarity
thresholds, and feature toggles - and makes them:

  1. Overridable at runtime via set_many() (called by the /settings API)
  2. Persisted to runtime_settings.json, so overrides survive a restart
  3. Read fresh on every call to get() - NOT imported once at module load

HOW TO ADD A NEW HOT-RELOADABLE SETTING
----------------------------------------
1. Add one entry to SCHEMA below (default pulled from config.py).
2. Wherever the value is used, replace the old top-of-file
       from config import THE_CONST
   with a call at the point of use:
       from runtime_settings import get as rs_get
       ... rs_get("THE_CONST") ...
   Importing it once at module load bakes in the value at startup, which
   defeats the whole point - always call get() where the value is
   actually used, not where the module is imported.

Anything NOT in SCHEMA is intentionally untouched by this module - it
still comes straight from config.py and still needs a restart to change
(that's correct for e.g. LLAMA_SERVER_PORT or DOCKER_IMAGE).
"""

import json
import threading
from pathlib import Path

import config

_LOCK = threading.Lock()
_OVERRIDES_PATH = Path(config.PROJECT_ROOT) / "runtime_settings.json"

# ─────────────────────────────────────────────────────────────
# Schema: name -> metadata. "default" is read from config.py so this file
# never hardcodes a value that could drift out of sync with config.py.
# type: "float" | "int" | "bool"
# ─────────────────────────────────────────────────────────────
SCHEMA = {
    # --- Sampling (sent per-request to llama-server, see main.py) ---
    "LLAMA_TEMP": {
        "type": "float", "min": 0.0, "max": 2.0,
        "default": config.LLAMA_TEMP, "group": "Sampling",
        "label": "Temperature",
        "help": "Higher = more random/creative. 0 = fully deterministic.",
    },
    "LLAMA_TOP_P": {
        "type": "float", "min": 0.0, "max": 1.0,
        "default": config.LLAMA_TOP_P, "group": "Sampling",
        "label": "Top-P",
        "help": "Nucleus sampling cutoff. 1.0 disables it.",
    },
    "LLAMA_TOP_K": {
        "type": "int", "min": 0, "max": 200,
        "default": config.LLAMA_TOP_K, "group": "Sampling",
        "label": "Top-K",
        "help": "Only consider the K most likely next tokens. 0 disables it.",
    },
    "LLAMA_MIN_P": {
        "type": "float", "min": 0.0, "max": 1.0,
        "default": config.LLAMA_MIN_P, "group": "Sampling",
        "label": "Min-P",
        "help": "Discards tokens below this fraction of the top token's probability.",
    },

    # --- Memory / RAG thresholds ---
    "MEMORY_SIMILARITY_THRESHOLD": {
        "type": "float", "min": 0.0, "max": 1.0,
        "default": config.MEMORY_SIMILARITY_THRESHOLD, "group": "Memory & RAG",
        "label": "Memory recall threshold",
        "help": "Lower = more permissive memory recall.",
    },
    "DOCUMENT_SIMILARITY_THRESHOLD": {
        "type": "float", "min": 0.0, "max": 1.0,
        "default": config.DOCUMENT_SIMILARITY_THRESHOLD, "group": "Memory & RAG",
        "label": "Document RAG threshold",
        "help": "Stricter threshold for search_documents results.",
    },
    "MEMORY_DEDUPE_SIMILARITY_THRESHOLD": {
        "type": "float", "min": 0.0, "max": 1.0,
        "default": config.MEMORY_DEDUPE_SIMILARITY_THRESHOLD, "group": "Memory & RAG",
        "label": "Memory dedupe threshold",
        "help": "How similar a new memory must be to an old one to flag as duplicate.",
    },
    "DURATION_CATEGORY_SIMILARITY_THRESHOLD": {
        "type": "float", "min": 0.0, "max": 1.0,
        "default": config.DURATION_CATEGORY_SIMILARITY_THRESHOLD, "group": "Memory & RAG",
        "label": "Task-duration category match threshold",
        "help": "Minimum similarity to match a task title to an existing category.",
    },

    # --- Tool selection ---
    "TOOL_SELECTION_ENABLED": {
        "type": "bool",
        "default": config.TOOL_SELECTION_ENABLED, "group": "Tool selection",
        "label": "Enable dynamic tool selection",
        "help": "Off = send every tool on every request (bigger prompts).",
    },
    "TOOL_SELECTION_MIN_SCORE": {
        "type": "float", "min": 0.0, "max": 1.0,
        "default": config.TOOL_SELECTION_MIN_SCORE, "group": "Tool selection",
        "label": "Tool selection min score",
        "help": "Minimum similarity for a tool group to be included.",
    },
    "TOOL_SELECTION_RESCUE_SCORE": {
        "type": "float", "min": 0.0, "max": 1.0,
        "default": config.TOOL_SELECTION_RESCUE_SCORE, "group": "Tool selection",
        "label": "Tool selection rescue score",
        "help": "Looser fallback threshold when nothing clears the min score.",
    },
    "TOOL_SELECTION_RESCUE_TOP_K": {
        "type": "int", "min": 1, "max": 20,
        "default": config.TOOL_SELECTION_RESCUE_TOP_K, "group": "Tool selection",
        "label": "Tool selection rescue top-K",
        "help": "Cap on how many tools the rescue tier can add.",
    },

    # --- Toggles ---
    "PROMPT_LOG_ENABLED": {
        "type": "bool",
        "default": config.PROMPT_LOG_ENABLED, "group": "Toggles",
        "label": "Prompt logging",
        "help": "Log every request body sent to llama-server (debug tool).",
    },
    "CALDAV_LOG_RAW_REQUESTS": {
        "type": "bool",
        "default": config.CALDAV_LOG_RAW_REQUESTS, "group": "Toggles",
        "label": "Log raw CalDAV requests",
        "help": "Include raw CalDAV HTTP traffic in the console log.",
    },
}

_overrides: dict = {}


def _coerce(name: str, raw_value):
    meta = SCHEMA[name]
    t = meta["type"]
    if t == "bool":
        value = bool(raw_value)
    elif t == "int":
        value = int(raw_value)
    elif t == "float":
        value = float(raw_value)
    else:
        raise ValueError(f"Unknown type for {name}")
    if "min" in meta and value < meta["min"]:
        raise ValueError(f"{name} must be >= {meta['min']}")
    if "max" in meta and value > meta["max"]:
        raise ValueError(f"{name} must be <= {meta['max']}")
    return value


def _load():
    global _overrides
    if _OVERRIDES_PATH.exists():
        try:
            raw = json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))
            # Silently drop keys no longer in SCHEMA (e.g. after a refactor)
            # rather than failing to boot over a stale settings file.
            _overrides = {k: v for k, v in raw.items() if k in SCHEMA}
        except (json.JSONDecodeError, OSError):
            _overrides = {}
    else:
        _overrides = {}


def _save():
    _OVERRIDES_PATH.write_text(json.dumps(_overrides, indent=2), encoding="utf-8")


_load()


def get(name: str):
    """Current effective value: override if set, else config.py's default."""
    if name not in SCHEMA:
        raise KeyError(f"{name} is not a registered hot-reloadable setting")
    with _LOCK:
        if name in _overrides:
            return _overrides[name]
        return SCHEMA[name]["default"]


def get_all() -> dict:
    """Everything the settings panel needs to render itself."""
    with _LOCK:
        return {
            name: {
                **{k: v for k, v in meta.items() if k != "default"},
                "default": meta["default"],
                "value": _overrides.get(name, meta["default"]),
                "overridden": name in _overrides,
            }
            for name, meta in SCHEMA.items()
        }


def set_many(updates: dict) -> dict:
    """
    Validate and apply multiple settings at once. Raises ValueError with a
    clear message on the FIRST bad value (all-or-nothing - never applies
    half a batch), then persists and returns the new effective values.
    """
    unknown = [k for k in updates if k not in SCHEMA]
    if unknown:
        raise ValueError(f"Unknown setting(s): {', '.join(unknown)}")
    coerced = {name: _coerce(name, value) for name, value in updates.items()}
    with _LOCK:
        _overrides.update(coerced)
        _save()
    return get_all()


def reset(name: str) -> dict:
    """Remove an override, falling back to config.py's default again."""
    with _LOCK:
        _overrides.pop(name, None)
        _save()
    return get_all()


def reset_all() -> dict:
    with _LOCK:
        _overrides.clear()
        _save()
    return get_all()
