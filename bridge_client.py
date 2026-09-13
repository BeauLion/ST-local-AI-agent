"""Thin HTTP client for the iCloud Bridge (bridge/main.py, running on the
Mac). Used by calendar_manager.py as a fallback path when CalDAV itself is
unreachable - see the "BACKEND FALLBACK MODEL" note at the top of
calendar_manager.py for how the two fit together.

This module is deliberately dumb: it knows the bridge's REST shape (from
bridge/models.py) and nothing about CalDAV, staging, or the confirm
workflow. calendar_manager.py owns all of that; this just gets data in and
out of the bridge and raises BridgeError on any failure, connectivity or
otherwise. Unlike CalDAV's CalendarError/CalendarConnectivityError split,
there's no further fallback to decide once we're here - by the time
calendar_manager.py calls this module, CalDAV has already been established
as the one that's down, so a BridgeError just means "both paths are
unavailable right now."
"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from config import (
    BRIDGE_DEFAULT_CALENDAR_NAME,
    BRIDGE_TIMEOUT_SECONDS,
    BRIDGE_TOKEN_ENV_VAR,
    BRIDGE_URL_ENV_VAR,
    PROJECT_ROOT,
)

# Same pattern as calendar_manager.py: harmless no-op if .env doesn't
# exist yet, since _base_url()/_token() below give a clear error the
# first time this is actually used without one.
load_dotenv(Path(PROJECT_ROOT) / ".env")


class BridgeError(Exception):
    """Raised for any iCloud-bridge problem - connection failure, a
    non-2xx response, or the bridge simply not being configured yet.
    Always caught and wrapped into a plain CalendarError by
    calendar_manager.py before it reaches the model, same contract as
    CalendarError itself."""


_client_lock = threading.Lock()
_cached_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _cached_client
    with _client_lock:
        if _cached_client is None:
            _cached_client = httpx.Client(timeout=BRIDGE_TIMEOUT_SECONDS)
        return _cached_client


def _base_url() -> str:
    url = os.environ.get(BRIDGE_URL_ENV_VAR)
    if not url:
        raise BridgeError(
            f"iCloud bridge is not configured - set {BRIDGE_URL_ENV_VAR} in "
            f".env to the Mac's Tailscale address, e.g. http://100.x.x.x:8787 "
            f"(or the MagicDNS name), then restart the agent server."
        )
    return url.rstrip("/")


def _token() -> str:
    token = os.environ.get(BRIDGE_TOKEN_ENV_VAR)
    if not token:
        raise BridgeError(
            f"iCloud bridge is not configured - set {BRIDGE_TOKEN_ENV_VAR} in "
            f".env to the same token generated on the Mac side, then restart "
            f"the agent server."
        )
    return token


def _request(method: str, path: str, **kwargs):
    url = f"{_base_url()}{path}"
    headers = {"Authorization": f"Bearer {_token()}"}
    try:
        response = _get_client().request(method, url, headers=headers, **kwargs)
    except httpx.RequestError as e:
        # Connection refused, DNS failure, timeout, Tailscale down, Mac
        # asleep, bridge process not running - all land here.
        raise BridgeError(f"Could not reach the iCloud bridge at {url}: {e}")

    if response.status_code >= 400:
        detail = response.text
        try:
            parsed = response.json()
            detail = parsed.get("detail", detail)
        except Exception:
            pass
        raise BridgeError(f"iCloud bridge returned {response.status_code}: {detail}")

    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def _iso(dt: datetime) -> str:
    """dt is expected to already be tz-aware (calendar_manager.py localizes
    before calling into this module) - isoformat() then carries an
    explicit offset the bridge's pydantic datetime fields parse fine."""
    return dt.isoformat()


def _parse_iso(value) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------

def list_calendars() -> list[dict]:
    """Raw CalendarOut dicts: id, title, allows_modifications, color, source."""
    return _request("GET", "/calendars") or []


def resolve_calendar(name: str = None) -> dict | None:
    """Match a bridge calendar by title - exact match, then unique partial
    match, mirroring calendar_manager._resolve_calendar's CalDAV logic so
    the two backends behave the same way from the user's perspective.

    Returns None when no name is given and BRIDGE_DEFAULT_CALENDAR_NAME
    either isn't set or doesn't match anything - callers should then omit
    calendar_id entirely and let the bridge's own
    defaultCalendarForNewEvents() pick, same "fall through silently"
    behavior CALENDAR_DEFAULT_NAME has on the CalDAV side.

    Raises BridgeError if an explicitly-given name matches nothing, or
    matches more than one calendar."""
    calendars = list_calendars()

    def _match(key: str):
        for cal in calendars:
            if (cal.get("title") or "").strip().lower() == key:
                return cal
        partial = [c for c in calendars if key in (c.get("title") or "").strip().lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            names = ", ".join(c.get("title") or "(unnamed)" for c in partial)
            raise BridgeError(f"Multiple bridge calendars match '{key}': {names}. Be more specific.")
        return None

    if name and name.strip():
        matched = _match(name.strip().lower())
        if matched is not None:
            return matched
        available = ", ".join(c.get("title") or "(unnamed)" for c in calendars)
        raise BridgeError(f"No bridge calendar named '{name}' found. Available: {available}")

    if BRIDGE_DEFAULT_CALENDAR_NAME and BRIDGE_DEFAULT_CALENDAR_NAME.strip():
        try:
            matched = _match(BRIDGE_DEFAULT_CALENDAR_NAME.strip().lower())
        except BridgeError:
            matched = None  # ambiguous default - no name was explicitly
                             # requested, so fall through instead of hard-failing
        if matched is not None:
            return matched

    return None


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def list_events(start_dt: datetime, end_dt: datetime, calendar_id: str = None) -> list[dict]:
    params = {"start": _iso(start_dt), "end": _iso(end_dt)}
    if calendar_id:
        params["calendar_id"] = calendar_id
    raw = _request("GET", "/events", params=params) or []
    return [_normalize_event(e) for e in raw]


def create_event(title: str, start_dt: datetime, end_dt: datetime, calendar_id: str = None,
                  location: str = "", description: str = "") -> dict:
    body = {"title": title, "start": _iso(start_dt), "end": _iso(end_dt)}
    if calendar_id:
        body["calendar_id"] = calendar_id
    if location:
        body["location"] = location
    if description:
        body["notes"] = description  # bridge/models.py calls this field "notes", not "description"
    raw = _request("POST", "/events", json=body)
    return _normalize_event(raw)


def update_event(raw_id: str, title: str = None, start_dt: datetime = None, end_dt: datetime = None,
                  location: str = None, description: str = None) -> dict:
    body = {}
    if title is not None:
        body["title"] = title
    if start_dt is not None:
        body["start"] = _iso(start_dt)
    if end_dt is not None:
        body["end"] = _iso(end_dt)
    if location is not None:
        body["location"] = location
    if description is not None:
        body["notes"] = description
    raw = _request("PUT", f"/events/{raw_id}", json=body)
    return _normalize_event(raw)


def delete_event(raw_id: str) -> None:
    _request("DELETE", f"/events/{raw_id}")


def _normalize_event(raw: dict) -> dict:
    """bridge/models.py's EventOut shape -> a plain dict with parsed
    datetimes, still in the bridge's own field names (id/notes/etc.).
    calendar_manager.py's _bridge_event_to_dict() does the final
    translation into this project's unified event-dict shape (uid/
    description/calendar/etc.) - kept as a separate step so this module
    stays backend-format-only and doesn't need to know calendar_manager's
    conventions."""
    return {
        "id": raw["id"],
        "title": raw.get("title") or "",
        "start": _parse_iso(raw.get("start")),
        "end": _parse_iso(raw.get("end")),
        "all_day": bool(raw.get("all_day")),
        "calendar_id": raw.get("calendar_id"),
        "calendar_title": raw.get("calendar_title"),
        "notes": raw.get("notes") or "",
        "location": raw.get("location") or "",
    }
