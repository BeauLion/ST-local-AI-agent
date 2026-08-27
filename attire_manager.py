"""
Attire manager: tracks each character's current clothing/attire as a small
set of structured slots, persisted across conversations and independent of
SillyTavern's own chat history.

Design (see chat discussion for full rationale):
  - One record PER CHARACTER (global), not per chat/story.
  - Structured slots only: head, top, bottom, feet, accessories.
    "accessories" is the one multi-item slot (list of strings); everything
    else is a single string or None.
  - Current state only - no history/timeline is kept in v1.
  - Character identity is supplied by the model itself (character_name
    argument on every tool call), NOT parsed out of the SillyTavern
    character card - the card's format isn't reliable enough to regex
    a name out of consistently. Names are matched case/whitespace-
    insensitively; anything that doesn't normalize to an existing name
    becomes a NEW character record. This means nicknames/typos can
    fragment into separate records - a known, accepted limitation for v1,
    not silently patched around. If this becomes a real problem in
    practice, an explicit alias/merge tool is the natural follow-up
    (parking that rather than building it speculatively now).

Storage is a single JSON file (attire.json) - consistent with
project_manager.py/memory.py's approach.
"""

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import ATTIRE_DATA_DIR, ATTIRE_SLOTS, MAX_ATTIRE_ITEM_LENGTH, MAX_CHARACTER_NAME_LENGTH

DATA_DIR = Path(ATTIRE_DATA_DIR)
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "attire.json"

# Single-process agent server - a plain lock is enough to stop concurrent
# requests from corrupting the JSON file, same as project_manager.py.
_lock = threading.Lock()

DEFAULT_STATE = {"characters": {}}

# Slots that hold a single string-or-None value. "accessories" is handled
# separately everywhere below since it holds a list instead.
SINGLE_VALUE_SLOTS = [s for s in ATTIRE_SLOTS if s != "accessories"]


class AttireManagerError(Exception):
    """Raised for any invalid operation - caught by main.py's tool wrappers
    and turned into a plain error string the model can read and react to."""


# --------------------------- storage ---------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = json.loads(json.dumps(DEFAULT_STATE))
    state.setdefault("characters", {})
    return state


def _save(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------- small helpers ---------------------------

def _normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_key(value) -> str:
    return _normalize_text(value).lower()


def _clean_item(value) -> str:
    """One attire item's display text - trimmed, length-capped. Callers
    treat '' as 'this slot is explicitly empty', so this never raises for
    an empty string, only for something over-length."""
    clean = _normalize_text(value)
    if len(clean) > MAX_ATTIRE_ITEM_LENGTH:
        raise AttireManagerError(
            f"Attire item text must be {MAX_ATTIRE_ITEM_LENGTH} characters or fewer."
        )
    return clean


def _new_character_record(name: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "slots": {slot: (None if slot != "accessories" else []) for slot in ATTIRE_SLOTS},
        "updated_at": _now(),
    }


# --------------------------- character resolution ---------------------------

def resolve_character(state: dict, character_name: str, create_if_missing: bool = True) -> dict:
    """Finds a character record by case/whitespace-insensitive name match,
    creating a new one if create_if_missing and nothing matches. See the
    module docstring re: this NOT doing fuzzy/nickname matching - exact
    normalized string match only."""
    clean_name = _normalize_text(character_name)
    if not clean_name:
        raise AttireManagerError("character_name is required.")
    if len(clean_name) > MAX_CHARACTER_NAME_LENGTH:
        raise AttireManagerError(
            f"character_name must be {MAX_CHARACTER_NAME_LENGTH} characters or fewer."
        )
    key = _normalize_key(clean_name)
    for record in state["characters"].values():
        if _normalize_key(record["name"]) == key:
            return record
    if not create_if_missing:
        raise AttireManagerError(f"No attire record exists yet for \u201c{clean_name}\u201d.")
    record = _new_character_record(clean_name)
    state["characters"][record["id"]] = record
    return record


def find_character_names_in_text(state: dict, text: str) -> list:
    """Best-effort scan of arbitrary text (intended use: the SillyTavern
    character-card system message) for any KNOWN character's name as a
    whole-word, case-insensitive substring. Used only for automatic context
    injection - never for resolving a tool call, and never raises. Returns
    the list of matching character ids (usually 0 or 1; more than one is
    treated as ambiguous by the caller and skipped)."""
    if not text:
        return []
    matches = []
    for record in state["characters"].values():
        name = record["name"]
        if not name:
            continue
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(record["id"])
    return matches


# --------------------------- updates ---------------------------

def update_attire(
    character_name: str,
    head: str = None,
    top: str = None,
    bottom: str = None,
    feet: str = None,
    accessories: str = None,
) -> tuple:
    """Updates only the slots explicitly passed (non-None). An explicit ""
    means "this slot is now empty" - same not-passed-vs-empty-string
    convention calendar_manager.py already uses for location/description.
    accessories is a single comma-separated string in the tool interface
    (more forgiving for a local model to produce than a JSON array) but is
    stored internally as a list.

    Returns (record, changed_slots) - changed_slots is a list of slot names
    that actually changed, so the caller can report "no change needed" the
    same way project_manager does for a status update that's already set.
    """
    requested = {
        "head": head, "top": top, "bottom": bottom, "feet": feet, "accessories": accessories,
    }
    if all(value is None for value in requested.values()):
        raise AttireManagerError("At least one slot (head/top/bottom/feet/accessories) is required.")

    with _lock:
        state = _load()
        record = resolve_character(state, character_name, create_if_missing=True)
        changed = []

        for slot in SINGLE_VALUE_SLOTS:
            value = requested[slot]
            if value is None:
                continue
            clean = _clean_item(value)
            new_value = clean if clean else None
            if record["slots"].get(slot) != new_value:
                record["slots"][slot] = new_value
                changed.append(slot)

        if accessories is not None:
            if _normalize_text(accessories) == "":
                new_list = []
            else:
                new_list = [
                    _clean_item(item) for item in accessories.split(",") if _normalize_text(item)
                ]
            if record["slots"].get("accessories", []) != new_list:
                record["slots"]["accessories"] = new_list
                changed.append("accessories")

        if not changed:
            return record, []

        record["updated_at"] = _now()
        _save(state)
        return record, changed


# --------------------------- read-only views ---------------------------

def _format_slots(slots: dict) -> list:
    lines = []
    for slot in SINGLE_VALUE_SLOTS:
        value = slots.get(slot)
        lines.append(f"- {slot}: {value if value else '(none)'}")
    accessories = slots.get("accessories") or []
    lines.append(f"- accessories: {', '.join(accessories) if accessories else '(none)'}")
    return lines


def get_attire_text(character_name: str) -> str:
    """What the attire_get tool hands back to the model for one character."""
    state = _load()
    try:
        record = resolve_character(state, character_name, create_if_missing=False)
    except AttireManagerError as e:
        return str(e)
    lines = [f"Current attire for {record['name']}:"] + _format_slots(record["slots"])
    return "\n".join(lines)


def build_context_text_for_ids(state: dict, character_ids: list) -> str:
    """Compact [PERSISTENT ATTIRE STATE] block for injection into the
    system prompt - the attire equivalent of project_manager's
    build_context_text(). Called by main.py with whatever character ids
    find_character_names_in_text() matched against the leading character
    card. Returns '' if there's nothing to show (caller should skip
    injecting a message entirely in that case, same as project_manager)."""
    if not character_ids:
        return ""
    blocks = []
    for char_id in character_ids:
        record = state["characters"].get(char_id)
        if not record:
            continue
        blocks.append(f"{record['name']}:\n" + "\n".join(_format_slots(record["slots"])))
    if not blocks:
        return ""
    lines = ["[PERSISTENT ATTIRE STATE]"] + blocks + [
        "Treat this as the authoritative current outfit for each character listed. "
        "Call attire_manager_update the moment the narrative - yours or the user's - "
        "describes ANY change to what a character is wearing, not just full outfit "
        "changes. This includes subtle or partial changes that are easy to skip: "
        "loosening or removing a tie, unbuttoning a shirt, rolling up sleeves, taking "
        "off shoes or socks, removing a jacket or coat, taking off jewelry or an "
        "accessory, a garment getting torn/soaked/dirtied and no longer being worn "
        "properly, or one item being swapped for another. If you are narrating a scene "
        "and you write a sentence that touches any of the above, call the tool in that "
        "same turn - do not wait for the change to become large or for the scene to end. "
        "Update only the slot(s) that actually changed - do not restate slots that "
        "didn't change. An empty value means that slot is now bare/nothing. When in "
        "doubt about whether something counts, log it anyway; a spurious update is "
        "cheaper than a stale one."
    ]
    return "\n\n".join(lines)
