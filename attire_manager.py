"""
Attire manager: tracks each character's current clothing/attire as a small
set of structured slots, persisted across conversations and independent of
SillyTavern's own chat history.

Design (see chat discussion for full rationale; brainstorm-layered-clothing.md
for the history of how v2 below was arrived at):
  - One record PER CHARACTER (global), not per chat/story.
  - v2: every slot (head/top/bottom/feet/accessories) is a LIST of items.
    v1 had head/top/bottom/feet as a single string-or-None and only
    accessories as a list - that asymmetry was the root cause of a real
    silent-erasure bug: a model reasoning about only the part of a scene
    that changed ("shoes go on over socks") had no way to add an item
    without also being trusted to correctly restate everything else
    already in that slot, and in practice it didn't always. v2 removes
    the "restate the full value" trust requirement structurally: the
    tool interface only offers add_item (append), remove_item (best-
    effort single-item removal), and replace_slot (deliberate full wipe)
    - there is no way to add one item and accidentally erase another.
  - Current state only - no history/timeline is kept.
  - Character identity is supplied by the model itself (character_name
    argument on every tool call), NOT parsed out of the SillyTavern
    character card - the card's format isn't reliable enough to regex
    a name out of consistently. Names are matched case/whitespace-
    insensitively; anything that doesn't normalize to an existing name
    becomes a NEW character record. This means nicknames/typos can
    fragment into separate records - a known, accepted limitation,
    not silently patched around. If this becomes a real problem in
    practice, an explicit alias/merge tool is the natural follow-up
    (parking that rather than building it speculatively now).
  - Item-identity matching for remove_item (does "the jacket" match a
    stored "denim jacket"?) is the same class of problem as character-
    name matching above, and gets the same treatment: a plain, documented
    matching rule (exact, then case-insensitive substring either
    direction), fail-open (no change, logged by the caller) on anything
    ambiguous or unmatched, rather than an undocumented heuristic that
    might guess wrong and silently remove the wrong item.

Storage is a single JSON file (attire.json) - consistent with
project_manager.py/memory.py's approach. _load() transparently upgrades
any record still stored in the old v1 format (single-value slots) the
first time it's read - see _migrate_record() - so there is no separate
one-off migration script to run.
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


class AttireManagerError(Exception):
    """Raised for any invalid operation - caught by main.py's tool wrappers
    and turned into a plain error string the model can read and react to."""


# --------------------------- storage ---------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_record(record: dict) -> dict:
    """Upgrades one record from the old v1 format - head/top/bottom/feet
    as a string-or-None, only accessories as a list - into v2, where every
    slot is a list. Idempotent: a slot already stored as a list passes
    through untouched, so this is safe to run on every load regardless of
    which format is actually on disk. Called from _load(), not as a
    separate migration step - the file upgrades itself in place the next
    time anything about it is saved."""
    slots = record.setdefault("slots", {})
    for slot in ATTIRE_SLOTS:
        value = slots.get(slot)
        if isinstance(value, list):
            continue
        slots[slot] = [value] if value else []
    return record


def _load() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = json.loads(json.dumps(DEFAULT_STATE))
    state.setdefault("characters", {})
    for record in state["characters"].values():
        _migrate_record(record)
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
    treat '' as 'nothing to add/set', so this never raises for an empty
    string, only for something over-length."""
    clean = _normalize_text(value)
    if len(clean) > MAX_ATTIRE_ITEM_LENGTH:
        raise AttireManagerError(
            f"Attire item text must be {MAX_ATTIRE_ITEM_LENGTH} characters or fewer."
        )
    return clean


def _check_slot(slot: str) -> str:
    if slot not in ATTIRE_SLOTS:
        raise AttireManagerError(f"Unknown slot '{slot}'. Valid slots: {', '.join(ATTIRE_SLOTS)}.")
    return slot


def _new_character_record(name: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "slots": {slot: [] for slot in ATTIRE_SLOTS},
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

def add_item(character_name: str, slot: str, item: str) -> tuple:
    """Appends one item to a slot's list. Cannot remove or overwrite
    anything else already in that slot by construction - this is the
    operation a "something was put on / layered on" narrative moment
    should call, exactly because it structurally can't erase whatever
    else is already there (see module docstring).

    Returns (record, added) - added is False if the item (case/whitespace-
    insensitively) is already present, same "no change needed" signal
    project_manager gives for a status update that's already set.
    """
    _check_slot(slot)
    clean = _clean_item(item)
    if not clean:
        raise AttireManagerError("item is required and cannot be empty for add_item.")

    with _lock:
        state = _load()
        record = resolve_character(state, character_name, create_if_missing=True)
        current = record["slots"].setdefault(slot, [])
        if any(_normalize_key(existing) == _normalize_key(clean) for existing in current):
            return record, False
        current.append(clean)
        record["updated_at"] = _now()
        _save(state)
        return record, True


def remove_item(character_name: str, slot: str, item_hint: str) -> tuple:
    """Removes whichever item in `slot` matches `item_hint`: exact
    normalized match first, then case-insensitive substring match in
    either direction. Zero or more-than-one candidate match is treated as
    "not confident enough to guess" and changes nothing - fail open,
    caller logs it. A skipped removal leaves the record stale but honest
    (the item is still listed even though the narrative removed it),
    which is the deliberately-accepted lesser failure mode - see
    brainstorm-layered-clothing.md's cross-cutting notes on item-identity
    matching.

    Returns (record, removed) - removed is the matched item's stored text
    on success, or None if nothing was confidently matched/changed.
    """
    _check_slot(slot)
    hint_key = _normalize_key(item_hint)
    if not hint_key:
        raise AttireManagerError("item_hint is required and cannot be empty for remove_item.")

    with _lock:
        state = _load()
        record = resolve_character(state, character_name, create_if_missing=False)
        current = record["slots"].get(slot, [])

        exact = [it for it in current if _normalize_key(it) == hint_key]
        if len(exact) == 1:
            match = exact[0]
        else:
            substring = [
                it for it in current
                if hint_key in _normalize_key(it) or _normalize_key(it) in hint_key
            ]
            match = substring[0] if len(substring) == 1 else None

        if match is None:
            return record, None

        current.remove(match)
        record["updated_at"] = _now()
        _save(state)
        return record, match


def replace_slot(character_name: str, slot: str, items: str) -> tuple:
    """Deliberate full wipe-and-set for one slot - the ONLY operation
    allowed to discard everything currently in it. `items` is a comma-
    separated string (same convention accessories already used pre-v2).
    Intended for genuine full changes (a character changes into a whole
    different outfit) - never for describing a single item coming on or
    off, which should go through add_item/remove_item instead so nothing
    else in the slot is put at risk.

    Returns (record, changed) - changed is False if the resulting list is
    identical to what was already stored.
    """
    _check_slot(slot)
    if _normalize_text(items) == "":
        new_list = []
    else:
        new_list = [_clean_item(it) for it in items.split(",") if _normalize_text(it)]

    with _lock:
        state = _load()
        record = resolve_character(state, character_name, create_if_missing=True)
        if record["slots"].get(slot, []) == new_list:
            return record, False
        record["slots"][slot] = new_list
        record["updated_at"] = _now()
        _save(state)
        return record, True


# --------------------------- read-only views ---------------------------

def _format_slots(slots: dict) -> list:
    lines = []
    for slot in ATTIRE_SLOTS:
        items = slots.get(slot) or []
        lines.append(f"- {slot}: {', '.join(items) if items else '(none)'}")
    return lines


def get_attire_text(character_name: str) -> str:
    """What the attire_manager_get tool hands back to the model for one
    character."""
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
    injecting a message entirely in that case, same as project_manager).

    Purely informational: the main agent has no attire tools of its own
    (attire tracking is handled entirely by the post-turn
    attire_subagent.py pass) - this block exists so the main agent's
    narration stays consistent with the authoritative state, not so it
    calls anything back."""
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
        "Treat this as the authoritative current outfit for each character listed - "
        "keep your narration consistent with it. This is maintained automatically by "
        "a background process; you do not have a tool for updating it and do not need "
        "to log changes yourself."
    ]
    return "\n\n".join(lines)
