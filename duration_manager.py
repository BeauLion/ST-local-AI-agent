"""
Task duration tracking: learns the user's actual task-completion durations
over time, so future "how long will this take" estimates are grounded in
personal historical data instead of an LLM zero-shot guess.

Full design rationale lives in brainstorm-task-duration-tracking.md - this
module implements that design, not a new one. In short: capture the
active->done elapsed wall-clock time as a loose anchor only (never asserted
as fact), log it immediately (default-accept, correct-by-exception), and
compute a median/MAD per category once enough entries exist.

Storage is a single JSON file (durations.json) - same shape/lock pattern as
project_manager.py's projects.json and memory.py's memories.json.

duration_manager.py must not import project_manager.py (project_manager.py
imports this module instead, to hook task-status transitions).
"""

import json
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import numpy as np

import memory
from config import (
    DURATION_CANONICAL_CATEGORIES,
    DURATION_CATEGORY_ALIASES,
    DURATION_CATEGORY_SIMILARITY_THRESHOLD,
    DURATION_DATA_DIR,
    MIN_ENTRIES_FOR_CONFIDENT_ESTIMATE,
    MIN_ENTRIES_FOR_ESTIMATE,
)

DATA_DIR = Path(DURATION_DATA_DIR)
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "durations.json"

# Single-process agent server - same reasoning as project_manager.py's lock.
_lock = threading.Lock()

DEFAULT_STATE = {
    "entries": [],
    "active_windows": {},
    "recently_closed_windows": [],
    "custom_categories": [],
}

UNCATEGORIZED = "uncategorized"

# recently_closed_windows only needs to cover the overlap-detection window
# of "still open elsewhere" tasks - personal-scale, single-user tool, so no
# need to retain this forever.
CLOSED_WINDOW_RETENTION_HOURS = 24


class DurationError(Exception):
    """Raised for any invalid duration-tracking operation - caught by
    main.py's tool wrappers and turned into a plain error string."""


# --------------------------- storage ---------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = json.loads(json.dumps(DEFAULT_STATE))
    state.setdefault("entries", [])
    state.setdefault("active_windows", {})
    state.setdefault("recently_closed_windows", [])
    state.setdefault("custom_categories", [])

    cutoff = _now() - timedelta(hours=CLOSED_WINDOW_RETENTION_HOURS)
    state["recently_closed_windows"] = [
        w for w in state["recently_closed_windows"]
        if _parse_iso(w["ended_at"]) >= cutoff
    ]
    return state


def _save(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------- small helpers ---------------------------

def _normalize_key(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


# --------------------------- active/inactive/done hooks ---------------------------

def on_task_active(task_id: str, project_id: str) -> None:
    with _lock:
        state = _load()
        state["active_windows"][task_id] = {"project_id": project_id, "started_at": _now_iso()}
        _save(state)


def on_task_inactive(task_id: str) -> None:
    with _lock:
        state = _load()
        if task_id in state["active_windows"]:
            del state["active_windows"][task_id]
            _save(state)


def on_task_done(task_id: str, project_id: str, title: str) -> str | None:
    with _lock:
        state = _load()
        window = state["active_windows"].pop(task_id, None)
        if window is None:
            _save(state)
            return None

        started_at = _parse_iso(window["started_at"])
        now = _now()
        elapsed_anchor_minutes = (now - started_at).total_seconds() / 60

        overlap = False
        for other_id, other_window in state["active_windows"].items():
            if other_id == task_id:
                continue
            other_start = _parse_iso(other_window["started_at"])
            if _overlaps(started_at, now, other_start, now):
                overlap = True
                break
        if not overlap:
            for closed in state["recently_closed_windows"]:
                other_start = _parse_iso(closed["started_at"])
                other_end = _parse_iso(closed["ended_at"])
                if _overlaps(started_at, now, other_start, other_end):
                    overlap = True
                    break

        state["recently_closed_windows"].append({
            "task_id": task_id,
            "started_at": window["started_at"],
            "ended_at": now.isoformat(),
        })

        if overlap:
            _save(state)
            return f"Duration not logged for “{title}” — overlapped with another active task."

        category = resolve_category(title, state=state)
        rounded = round(elapsed_anchor_minutes)

        if category is None:
            entry_category = UNCATEGORIZED
            flag = (
                f"~{rounded} min logged for “{title}” ({UNCATEGORIZED}). "
                f"Want me to add a tracked category for tasks like this?"
            )
        else:
            entry_category = category
            flag = (
                f"~{rounded} min logged for “{title}” (category: {category}). "
                f"Reply if that's off."
            )

        entry = {
            "id": f"dur_{uuid.uuid4().hex}",
            "category": entry_category,
            "elapsed_anchor_minutes": elapsed_anchor_minutes,
            "logged_value_minutes": elapsed_anchor_minutes,
            "confirmation_state": "accepted",
            "task_id": task_id,
            "project_id": project_id,
            "title": title,
            "timestamp": now.isoformat(),
        }
        state["entries"].append(entry)
        _save(state)
        return flag


# --------------------------- categorization ---------------------------

_category_embedding_cache: dict[str, list] = {}


def _category_embedding(name: str) -> list:
    if name not in _category_embedding_cache:
        _category_embedding_cache[name] = memory.embed(name.replace("_", " "))
    return _category_embedding_cache[name]


def resolve_category(text: str, *, state: dict | None = None) -> str | None:
    """Resolve free text to a canonical/custom category name, or None if
    nothing matches closely enough. Never mutates state or creates
    categories - callers decide what to do with a None result."""
    norm = _normalize_key(text)
    if not norm:
        return None

    alias = DURATION_CATEGORY_ALIASES.get(norm)
    if alias:
        return alias

    owns_state = state is None
    if owns_state:
        state = _load()
    all_categories = list(DURATION_CANONICAL_CATEGORIES) + list(state.get("custom_categories", []))

    for category in all_categories:
        label = category.replace("_", " ")
        if norm == category or norm == label or label in norm:
            return category

    query_vec = np.array(memory.embed(text))
    best_category, best_score = None, -1.0
    for category in all_categories:
        score = float(np.array(_category_embedding(category)) @ query_vec)
        if score > best_score:
            best_category, best_score = category, score

    if best_category is not None and best_score >= DURATION_CATEGORY_SIMILARITY_THRESHOLD:
        return best_category
    return None


def confirm_new_category(name: str, reclassify_last: bool = True) -> str:
    clean = _normalize_key(name).replace(" ", "_")
    if not clean:
        raise DurationError("A category name is required.")

    with _lock:
        state = _load()
        existing = list(DURATION_CANONICAL_CATEGORIES) + list(state["custom_categories"])
        if clean not in existing:
            state["custom_categories"].append(clean)

        if reclassify_last:
            for entry in reversed(state["entries"]):
                if entry["category"] == UNCATEGORIZED:
                    entry["category"] = clean
                    break

        _save(state)
        return f"Added “{clean}” as a tracked duration category."


# --------------------------- estimates & corrections ---------------------------

def get_estimate(query: str) -> dict:
    category = resolve_category(query)
    if category is None:
        return {"resolved": False}

    state = _load()
    values = [
        e["logged_value_minutes"] for e in state["entries"]
        if e["category"] == category
    ]
    n = len(values)

    if n == 0:
        return {
            "resolved": True, "category": category, "n": 0,
            "confidence": "insufficient", "median_minutes": None, "mad_minutes": None,
        }

    if n < MIN_ENTRIES_FOR_ESTIMATE:
        confidence = "insufficient"
    elif n < MIN_ENTRIES_FOR_CONFIDENT_ESTIMATE:
        confidence = "rough"
    else:
        confidence = "confident"

    med = median(values)
    mad = median(abs(v - med) for v in values)

    return {
        "resolved": True, "category": category, "n": n, "confidence": confidence,
        "median_minutes": round(med, 1), "mad_minutes": round(mad, 1),
    }


_HOUR_UNITS = ("h", "hr", "hrs", "hour", "hours")
_MINUTE_UNITS = ("m", "min", "mins", "minute", "minutes")

_PHRASE_MINUTES = {
    "a quarter hour": 15, "quarter hour": 15, "a quarter of an hour": 15,
    "half an hour": 30, "half hour": 30, "a half hour": 30,
    "an hour": 60, "one hour": 60, "1 hour": 60,
    "an hour and a half": 90, "hour and a half": 90, "1.5 hours": 90,
    "a couple hours": 120, "couple of hours": 120, "two hours": 120,
}

# Compound "1h30m" / "1hr 30min" / "1 hour 30 minutes" - matched before the
# single-unit pattern below since e.g. "1h30m" would otherwise fail to
# match (it has two number+unit chunks, not one). Needed so that
# project_manager.py's note-tag round-trip (writing "dur: 1h30m", then
# re-parsing it later) never loses data.
_COMPOUND_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\s*(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)$"
)


def parse_duration_minutes(value_text: str) -> float | None:
    clean = _normalize_key(value_text)
    clean = clean.replace("~", "").replace("about", "").replace("approx", "").strip()
    if not clean:
        return None

    if clean in _PHRASE_MINUTES:
        return float(_PHRASE_MINUTES[clean])

    compound = _COMPOUND_RE.match(clean)
    if compound:
        return float(compound.group(1)) * 60 + float(compound.group(2))

    match = re.match(
        r"^(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)?$", clean
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit in _HOUR_UNITS:
        return value * 60
    return value


def correct_entry(task_or_category: str, value_text: str) -> str:
    minutes = parse_duration_minutes(value_text)
    if minutes is None:
        raise DurationError(f"Couldn't parse a duration from “{value_text}”. Try e.g. '20min', '1h', '90'.")

    with _lock:
        state = _load()
        entries = state["entries"]
        if not entries:
            raise DurationError("No logged durations exist yet.")

        query = _normalize_key(task_or_category)
        match = None

        if query:
            title_matches = [e for e in entries if query in _normalize_key(e["title"])]
            if title_matches:
                match = max(title_matches, key=lambda e: e["timestamp"])

        if match is None and query:
            category = resolve_category(task_or_category, state=state)
            if category:
                category_matches = [e for e in entries if e["category"] == category]
                if category_matches:
                    match = max(category_matches, key=lambda e: e["timestamp"])

        if match is None and not query:
            match = max(entries, key=lambda e: e["timestamp"])

        if match is None:
            raise DurationError(f"Couldn't find a logged duration matching “{task_or_category}”.")

        match["logged_value_minutes"] = minutes
        match["confirmation_state"] = "corrected"
        _save(state)
        return f"Updated “{match['title']}” (category: {match['category']}) to ~{round(minutes)} min."
