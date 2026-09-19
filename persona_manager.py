"""
Persona manager: named system-prompt snippets the chat widget (web/
chat_widget.js) lets you switch between - e.g. "Concise", "Code reviewer".
The selected persona's content is sent as a system message alongside the
user's prompt on /webchat/completions.

Storage is a single JSON file (personas.json), same pattern as
project_manager.py/memory.py - plenty for personal-scale use (a handful of
personas).
"""

import json
import threading
import uuid
from pathlib import Path

from config import PERSONA_DATA_DIR

DATA_DIR = Path(PERSONA_DATA_DIR)
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "personas.json"

# Same single-process-lock reasoning as project_manager.py's _lock.
_lock = threading.Lock()

DEFAULT_STATE = {"personas": {}, "selected_persona_id": None}

MAX_PERSONA_NAME_LENGTH = 80
MAX_PERSONA_CONTENT_LENGTH = 8000


class PersonaManagerError(Exception):
    """Raised for any invalid operation - caught by main.py's route handlers
    and turned into a 400 response the widget can display."""


def _load() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = json.loads(json.dumps(DEFAULT_STATE))
    state.setdefault("personas", {})
    state.setdefault("selected_persona_id", None)
    return state


def _save(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def list_personas() -> dict:
    with _lock:
        state = _load()
        return {
            "personas": [
                {"id": pid, "name": p["name"], "content": p["content"]}
                for pid, p in state["personas"].items()
            ],
            "selected_persona_id": state["selected_persona_id"],
        }


def create_persona(name: str, content: str) -> dict:
    name = (name or "").strip()
    content = content or ""
    if not name:
        raise PersonaManagerError("Persona name cannot be empty.")
    if len(name) > MAX_PERSONA_NAME_LENGTH:
        raise PersonaManagerError(f"Persona name too long (max {MAX_PERSONA_NAME_LENGTH} chars).")
    if len(content) > MAX_PERSONA_CONTENT_LENGTH:
        raise PersonaManagerError(f"Persona content too long (max {MAX_PERSONA_CONTENT_LENGTH} chars).")
    with _lock:
        state = _load()
        pid = uuid.uuid4().hex[:8]
        state["personas"][pid] = {"name": name, "content": content}
        _save(state)
        return {"id": pid, "name": name, "content": content}


def update_persona(persona_id: str, name: str | None = None, content: str | None = None) -> dict:
    with _lock:
        state = _load()
        persona = state["personas"].get(persona_id)
        if not persona:
            raise PersonaManagerError(f"No persona with id '{persona_id}'.")
        if name is not None:
            name = name.strip()
            if not name:
                raise PersonaManagerError("Persona name cannot be empty.")
            if len(name) > MAX_PERSONA_NAME_LENGTH:
                raise PersonaManagerError(f"Persona name too long (max {MAX_PERSONA_NAME_LENGTH} chars).")
            persona["name"] = name
        if content is not None:
            if len(content) > MAX_PERSONA_CONTENT_LENGTH:
                raise PersonaManagerError(f"Persona content too long (max {MAX_PERSONA_CONTENT_LENGTH} chars).")
            persona["content"] = content
        _save(state)
        return {"id": persona_id, "name": persona["name"], "content": persona["content"]}


def delete_persona(persona_id: str) -> None:
    with _lock:
        state = _load()
        if persona_id not in state["personas"]:
            raise PersonaManagerError(f"No persona with id '{persona_id}'.")
        del state["personas"][persona_id]
        if state["selected_persona_id"] == persona_id:
            state["selected_persona_id"] = None
        _save(state)


def select_persona(persona_id: str | None) -> None:
    with _lock:
        state = _load()
        if persona_id is not None and persona_id not in state["personas"]:
            raise PersonaManagerError(f"No persona with id '{persona_id}'.")
        state["selected_persona_id"] = persona_id
        _save(state)


def get_selected_persona() -> dict | None:
    with _lock:
        state = _load()
        pid = state["selected_persona_id"]
        if not pid:
            return None
        persona = state["personas"].get(pid)
        if not persona:
            return None
        return {"id": pid, "name": persona["name"], "content": persona["content"]}
