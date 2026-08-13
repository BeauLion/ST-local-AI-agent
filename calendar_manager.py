"""
Calendar manager: read/search/create/edit/delete events on the user's
iCloud calendar over CalDAV.

Apple doesn't offer a modern OAuth calendar API for third-party apps like
this one, but iCloud speaks CalDAV (an older, open protocol most calendar
apps use under the hood). Auth is done with an "app-specific password"
generated at appleid.apple.com - not the user's real Apple ID password -
so it can be revoked independently at any time.

SAFETY MODEL - staged changes, not immediate writes:
Every write operation (create/edit/delete) is a two-step process:
  1. calendar_create_event / calendar_edit_event / calendar_delete_event
     validates the request and STAGES it - nothing touches iCloud yet.
     It returns a plain-language description of exactly what would happen.
  2. calendar_confirm_pending is the ONLY function that actually writes to
     iCloud, and it only applies whatever is currently staged.
This mirrors the pattern in project_manager.py (ProjectManagerError ->
plain string) but goes further: unlike delete_file (which relies on the
model reliably following a system-prompt instruction to "only call this
when the user clearly asks"), a calendar write cannot happen at all until
a second, separate confirm step occurs. The model can misbehave and stage
things speculatively without any real-world consequence.

Only one change can be staged at a time (single-user, single-session use
case) - staging a new one silently replaces whatever was staged before.
"""

import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import caldav
import icalendar
from dotenv import load_dotenv

from config import (
    CALDAV_TIMEOUT_SECONDS,
    CALENDAR_BATCH_DEFAULT_DURATION_MINUTES,
    CALENDAR_BATCH_MAX_EVENTS,
    CALENDAR_DEFAULT_LOOKAHEAD_DAYS,
    CALENDAR_DEFAULT_NAME,
    CALENDAR_PENDING_CHANGE_TTL_MINUTES,
    CALENDAR_SEARCH_LOOKAHEAD_DAYS,
    CALENDAR_SEARCH_LOOKBACK_DAYS,
    CALENDAR_TIMEZONE,
    CALENDAR_UID_LOOKUP_LOOKAHEAD_DAYS,
    CALENDAR_UID_LOOKUP_LOOKBACK_DAYS,
    CALENDAR_WRITE_RETRIES,
    ICLOUD_APP_PASSWORD_ENV_VAR,
    ICLOUD_CALDAV_URL,
    ICLOUD_USERNAME_ENV_VAR,
    PROJECT_ROOT,
)

# Cached once - ZoneInfo() does a filesystem/tzdata lookup internally, no
# need to repeat it on every event write.
_LOCAL_TZ = ZoneInfo(CALENDAR_TIMEZONE)


def _localize(dt: datetime) -> datetime:
    """Attach CALENDAR_TIMEZONE to a naive datetime before it's written to
    iCloud. Only used at the point events are actually created/edited -
    NOT applied to read-path datetimes (list/search/date_search), which
    were already working correctly and use naive datetimes consistently
    with each other. Mixing that up would risk breaking working reads to
    fix a writes-only bug."""
    if dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=_LOCAL_TZ)

# Loads ICLOUD_USERNAME / ICLOUD_APP_PASSWORD from a .env file sitting next
# to config.py. If the file doesn't exist yet, this is a harmless no-op -
# _get_credentials() below gives a clear error the first time a calendar
# tool is actually used.
load_dotenv(Path(PROJECT_ROOT) / ".env")


class CalendarError(Exception):
    """Raised for any calendar problem - caught by main.py's tool wrappers
    and turned into a plain 'Error: ...' string the model can read, relay
    honestly, and react to (same pattern as ProjectManagerError)."""


# ---------------------------------------------------------------------------
# Connection (lazy + cached, same reconnect-on-failure pattern main.py uses
# for the Docker client in run_python)
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_cached_client = None
_cached_calendars = None  # list of caldav Calendar objects, cached per-process


def _get_credentials() -> tuple[str, str]:
    username = os.environ.get(ICLOUD_USERNAME_ENV_VAR)
    password = os.environ.get(ICLOUD_APP_PASSWORD_ENV_VAR)
    if not username or not password:
        raise CalendarError(
            f"iCloud credentials not found. Copy .env.example to .env in the "
            f"project root and fill in {ICLOUD_USERNAME_ENV_VAR} and "
            f"{ICLOUD_APP_PASSWORD_ENV_VAR}, then restart the agent server."
        )
    return username, password


def _get_client() -> caldav.DAVClient:
    global _cached_client
    with _client_lock:
        if _cached_client is None:
            username, password = _get_credentials()
            try:
                client = caldav.DAVClient(
                    url=ICLOUD_CALDAV_URL, username=username, password=password,
                    timeout=CALDAV_TIMEOUT_SECONDS,
                )
                client.principal()  # forces a round-trip now, not on first real use
            except Exception as e:
                raise CalendarError(
                    f"Could not connect to iCloud calendar: {e}. Double-check the "
                    f"Apple ID and app-specific password in .env, and that the "
                    f"app-specific password hasn't been revoked."
                )
            _cached_client = client
        return _cached_client


def _get_calendars(refresh: bool = False):
    global _cached_calendars
    if _cached_calendars is not None and not refresh:
        return _cached_calendars
    client = _get_client()
    try:
        calendars = client.principal().calendars()
    except Exception as e:
        # Connection may have gone stale (password revoked mid-session, etc).
        # Drop the cached client so the next call reconnects fresh.
        _reset_client()
        raise CalendarError(f"Could not list iCloud calendars: {e}")
    if not calendars:
        raise CalendarError("No calendars found on this iCloud account.")
    _cached_calendars = calendars
    return calendars


_CALENDAR_RETRY_DELAY_SECONDS = 2


def _date_search_with_retry(cal, start_dt, end_dt, error_prefix: str):
    try:
        return cal.date_search(start_dt, end_dt)
    except Exception as first_error:
        time.sleep(_CALENDAR_RETRY_DELAY_SECONDS)
        try:
            return cal.date_search(start_dt, end_dt)
        except Exception as second_error:
            raise CalendarError(f"{error_prefix}: {second_error}")


def _find_matching_event(cal, title: str, start_dt: datetime):
    """Look for an event with a matching title at the same start time in
    `cal`, in a narrow window around start_dt. Used as a duplicate guard
    before creating an event: a previous create attempt may have
    succeeded on iCloud's side even though the client never saw the
    response (e.g. after a read timeout), and cal.add_event() always
    mints a new UID - so blindly retrying a create can silently
    double-book. Returns the matching event, or None."""
    window_start = start_dt - timedelta(minutes=1)
    window_end = start_dt + timedelta(minutes=1)
    try:
        events = cal.date_search(window_start, window_end)
    except Exception:
        # Don't let a failed duplicate-check block the real create
        # attempt - fall through and let that attempt's own error
        # handling decide the outcome.
        return None
    target = title.strip().lower()
    for event in events:
        try:
            summary = _ical_field(event.icalendar_component, "summary", "")
        except Exception:
            continue
        if summary.strip().lower() == target:
            return event
    return None


def _reset_client():
    global _cached_client, _cached_calendars
    _cached_client = None
    _cached_calendars = None


def _resolve_calendar(name: str = None):
    """Match by exact name, then unique partial name. When no name is
    given, falls back to CALENDAR_DEFAULT_NAME (config.py) if it matches a
    real calendar; if that's unset or doesn't match anything on the
    account, falls back to whichever calendar iCloud happens to return
    first."""
    calendars = _get_calendars()

    def _match(key: str):
        for cal in calendars:
            if (cal.name or "").strip().lower() == key:
                return cal
        partial = [c for c in calendars if key in (c.name or "").strip().lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            names = ", ".join(c.name or "(unnamed)" for c in partial)
            raise CalendarError(f"Multiple calendars match '{key}': {names}. Be more specific.")
        return None

    if name and name.strip():
        matched = _match(name.strip().lower())
        if matched is not None:
            return matched
        available = ", ".join(c.name or "(unnamed)" for c in calendars)
        raise CalendarError(f"No calendar named '{name}' found. Available calendars: {available}")

    if CALENDAR_DEFAULT_NAME and CALENDAR_DEFAULT_NAME.strip():
        try:
            matched = _match(CALENDAR_DEFAULT_NAME.strip().lower())
        except CalendarError:
            matched = None  # CALENDAR_DEFAULT_NAME ambiguously partial-matched - no name
                             # was explicitly requested here, so fall through instead
                             # of hard-failing every calendar_name-less call.
        if matched is not None:
            return matched
        # Configured default doesn't exist on this account - fall through
        # silently rather than hard-failing every calendar_name-less call.

    return calendars[0]


# ---------------------------------------------------------------------------
# Datetime parsing / event serialization
# ---------------------------------------------------------------------------

def _parse_datetime(value: str) -> datetime:
    """Accepts 'YYYY-MM-DD HH:MM', 'YYYY-MM-DDTHH:MM', or a bare
    'YYYY-MM-DD' (treated as midnight). Naive local time throughout - this
    is a single-user local setup, not built for cross-timezone scheduling."""
    value = (value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise CalendarError(f"Could not parse date/time '{value}'. Use 'YYYY-MM-DD HH:MM'.")


def _ical_field(component, name: str, default: str = ""):
    """Read a field off an icalendar Event component. Date/time fields come
    back as vDDDTypes with a .dt attribute; plain text fields are returned
    directly by .get(). Both are coerced to plain strings for the model."""
    value = component.get(name)
    if value is None:
        return default
    if hasattr(value, "dt"):
        return str(value.dt)
    return str(value)


def _event_to_dict(event, calendar_label: str = "") -> dict:
    try:
        component = event.icalendar_component
    except Exception as e:
        raise CalendarError(f"Could not read an event's data: {e}")
    if component is None:
        raise CalendarError("Could not read an event's data: empty calendar object.")

    return {
        "uid": _ical_field(component, "uid"),
        "title": _ical_field(component, "summary", "(no title)"),
        "start": _ical_field(component, "dtstart"),
        "end": _ical_field(component, "dtend"),
        "location": _ical_field(component, "location"),
        "description": _ical_field(component, "description"),
        "calendar": calendar_label,
    }


def list_calendar_names() -> list[str]:
    """Every calendar name on the account - lets the user/model see what's
    available, e.g. to troubleshoot events not showing up or to target a
    specific calendar by name in the write tools."""
    calendars = _get_calendars()
    return [c.name or "(unnamed)" for c in calendars]


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _day_bounds(d) -> tuple:
    start = datetime.combine(d, datetime.min.time())
    return start, start + timedelta(days=1)


def resolve_when(when: str) -> tuple:
    if not when or not when.strip():
        raise CalendarError("No date phrase given to resolve.")
    text = when.strip().lower()
    now = datetime.now()
    today = now.date()

    if text in ("today", "tonight"):
        return _day_bounds(today)
    if text == "tomorrow":
        return _day_bounds(today + timedelta(days=1))
    if text == "yesterday":
        return _day_bounds(today - timedelta(days=1))
    if text == "day after tomorrow":
        return _day_bounds(today + timedelta(days=2))

    if text == "this week":
        start = today - timedelta(days=today.weekday())
        return datetime.combine(start, datetime.min.time()), datetime.combine(start + timedelta(days=7), datetime.min.time())
    if text == "next week":
        start = today - timedelta(days=today.weekday()) + timedelta(days=7)
        return datetime.combine(start, datetime.min.time()), datetime.combine(start + timedelta(days=7), datetime.min.time())
    if text in ("this weekend", "the weekend"):
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        return datetime.combine(saturday, datetime.min.time()), datetime.combine(saturday + timedelta(days=2), datetime.min.time())
    if text == "this month":
        start = today.replace(day=1)
        end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
        return datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time())

    for name, idx in _WEEKDAYS.items():
        if text in (name, f"this {name}", f"next {name}"):
            delta = (idx - today.weekday()) % 7
            if delta == 0 and text == f"next {name}":
                delta = 7
            return _day_bounds(today + timedelta(days=delta))

    import dateparser
    parsed = dateparser.parse(when, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now})
    if parsed is None:
        raise CalendarError(
            f"Could not understand the date phrase '{when}'. Try a specific "
            f"date like 'YYYY-MM-DD' instead."
        )
    return _day_bounds(parsed.date())


# ---------------------------------------------------------------------------
# Read operations - execute immediately, no staging needed
# ---------------------------------------------------------------------------

# Cache of the most recent list/search results (single slot, like
# _pending_change) - lets stage_edit_event/stage_delete_event recover when
# the model fabricates a UID instead of copying the real one it was just
# shown. See _find_event_with_fallback below.
_last_results_lock = threading.Lock()
_last_results: list[dict] = []


def list_events(start: str = None, end: str = None, calendar_name: str = None, when: str = None) -> list[dict]:
    # Default to searching EVERY calendar on the account - a named
    # calendar_name narrows to just one. Without this, events silently
    # wouldn't show up if they live on any calendar other than whichever
    # one iCloud happens to return first.
    calendars = [_resolve_calendar(calendar_name)] if calendar_name else _get_calendars()

    if when and not (start or end):
        start_dt, end_dt = resolve_when(when)
    elif start:
        start_dt = _parse_datetime(start)
        if end:
            end_dt = _parse_datetime(end)
            if end_dt <= start_dt:
                # A single day is naturally expressed as the same date for both
                # start and end (e.g. "tomorrow" -> start='2026-08-12',
                # end='2026-08-12'). Treat that as "the whole day" rather than
                # rejecting it as a zero-length range.
                end_dt = start_dt + timedelta(days=1)
        else:
            end_dt = start_dt + timedelta(days=CALENDAR_DEFAULT_LOOKAHEAD_DAYS)
    else:
        start_dt = datetime.now()
        end_dt = start_dt + timedelta(days=CALENDAR_DEFAULT_LOOKAHEAD_DAYS)

    results = []
    for cal in calendars:
        label = cal.name or "(unnamed)"
        events = _date_search_with_retry(cal, start_dt, end_dt, f"Could not fetch events from '{label}'")
        results.extend(_event_to_dict(e, label) for e in events)

    results.sort(key=lambda e: e["start"])
    with _last_results_lock:
        _last_results.clear()
        _last_results.extend(results)
    return results


def search_events(query: str, start: str = None, end: str = None, calendar_name: str = None, when: str = None) -> list[dict]:
    if not query or not query.strip():
        raise CalendarError("A search query is required.")

    calendars = [_resolve_calendar(calendar_name)] if calendar_name else _get_calendars()

    if when and not (start or end):
        start_dt, end_dt = resolve_when(when)
    else:
        start_dt = _parse_datetime(start) if start else datetime.now() - timedelta(days=CALENDAR_SEARCH_LOOKBACK_DAYS)
        end_dt = _parse_datetime(end) if end else datetime.now() + timedelta(days=CALENDAR_SEARCH_LOOKAHEAD_DAYS)

    key = query.strip().lower()
    matches = []
    for cal in calendars:
        label = cal.name or "(unnamed)"
        events = _date_search_with_retry(cal, start_dt, end_dt, f"Could not search '{label}'")
        for event in events:
            d = _event_to_dict(event, label)
            haystack = f"{d['title']} {d['description']} {d['location']}".lower()
            if key in haystack:
                matches.append(d)
    matches.sort(key=lambda e: e["start"])
    with _last_results_lock:
        _last_results.clear()
        _last_results.extend(matches)
    return matches


# ---------------------------------------------------------------------------
# Staged writes - validate + describe now, apply only on confirm
# ---------------------------------------------------------------------------

_pending_lock = threading.Lock()
_pending_change = None  # {"description": str, "apply": callable, "created_at": float}


def _stage(description: str, apply_fn) -> str:
    global _pending_change
    with _pending_lock:
        _pending_change = {"description": description, "apply": apply_fn, "created_at": time.time()}
    return (
        f"Staged (NOT yet applied to the calendar): {description}\n"
        f"Tell the user exactly what this will do and ask them to confirm. "
        f"Only call calendar_confirm_pending if their next message clearly "
        f"confirms; call calendar_cancel_pending if they decline or want changes."
    )


def _peek_pending():
    """Returns the pending change dict if one exists and hasn't expired,
    else None (and clears it if expired)."""
    global _pending_change
    if _pending_change is None:
        return None
    age_minutes = (time.time() - _pending_change["created_at"]) / 60
    if age_minutes > CALENDAR_PENDING_CHANGE_TTL_MINUTES:
        _pending_change = None
        return None
    return _pending_change


def stage_create_event(title: str, start: str, end: str = None, location: str = "",
                        description: str = "", calendar_name: str = None) -> str:
    if not title or not title.strip():
        raise CalendarError("An event title is required.")
    cal = _resolve_calendar(calendar_name)  # validated up front so a bad name fails before staging
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end) if end else start_dt + timedelta(hours=1)
    if end_dt <= start_dt:
        raise CalendarError("Event end time must be after the start time.")
    # Attach the configured local timezone now - iCloud otherwise has no
    # timezone info to go on and events land shifted (usually to UTC).
    start_dt = _localize(start_dt)
    end_dt = _localize(end_dt)

    def apply_fn() -> str:
        # Duplicate guard: a previous attempt may have succeeded on
        # iCloud's side even though the client never saw the response
        # (e.g. a timeout) - check before minting a new event.
        if _find_matching_event(cal, title, start_dt) is not None:
            return (f"'{title}' on {start_dt.strftime('%A %Y-%m-%d')} "
                     f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')} already exists - "
                     f"not creating a duplicate.")
        try:
            cal.add_event(
                dtstart=start_dt, dtend=end_dt, summary=title,
                location=location or None, description=description or None,
            )
        except Exception as e:
            raise CalendarError(f"Failed to create event: {e}")
        return f"Created '{title}' on {start_dt.strftime('%A %Y-%m-%d')} {start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}."

    desc = (f"CREATE '{title}' on {start_dt.strftime('%a %Y-%m-%d')} "
            f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            + (f" at {location}" if location else "")
            + f" (in calendar '{cal.name or '(unnamed)'}')")
    return _stage(desc, apply_fn)


def _get_busy_intervals(start_dt: datetime, end_dt: datetime) -> list:
    """Return (busy_start, busy_end, title) tuples, tz-aware, for every
    event across EVERY calendar on the account that overlaps
    [start_dt, end_dt). start_dt/end_dt are naive here, consistent with
    the other read paths in this module (list_events/search_events) -
    iCloud's date_search accepts naive datetimes fine, same as those.
    Used by stage_create_events_batch to check a proposed schedule
    against what's already on the calendar, on ANY calendar - a
    conflicting event elsewhere should still block scheduling here."""
    busy = []
    for cal in _get_calendars():
        events = _date_search_with_retry(
            cal, start_dt, end_dt, f"Could not check '{cal.name or '(unnamed)'}' for conflicts"
        )
        for event in events:
            try:
                component = event.icalendar_component
            except Exception:
                continue
            if component is None:
                continue
            dtstart = component.get("dtstart")
            if dtstart is None:
                continue
            dtend = component.get("dtend")
            s = dtstart.dt
            e = dtend.dt if dtend is not None else s
            # Normalize all-day (date-only) events to a full-day interval
            # in the configured local timezone so they still block a
            # proposed slot inside that day.
            if not isinstance(s, datetime):
                s = datetime.combine(s, datetime.min.time(), tzinfo=_LOCAL_TZ)
            elif s.tzinfo is None:
                s = s.replace(tzinfo=_LOCAL_TZ)
            if not isinstance(e, datetime):
                e = datetime.combine(e, datetime.min.time(), tzinfo=_LOCAL_TZ)
            elif e.tzinfo is None:
                e = e.replace(tzinfo=_LOCAL_TZ)
            busy.append((s, e, _ical_field(component, "summary", "(no title)")))
    return busy


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def stage_create_events_batch(events: list, calendar_name: str = None) -> str:
    """Like stage_create_event, but for MULTIPLE events staged as one
    pending change - e.g. a proposed morning schedule. Validates that no
    two proposed events overlap each other, and that none of them
    overlap anything already on the calendar, before staging anything.
    A single calendar_confirm_pending call then creates all of them."""
    if not isinstance(events, list) or len(events) < 1:
        raise CalendarError("At least one event is required.")
    if len(events) > CALENDAR_BATCH_MAX_EVENTS:
        raise CalendarError(f"A batch may contain at most {CALENDAR_BATCH_MAX_EVENTS} events.")

    cal = _resolve_calendar(calendar_name)  # validated up front, same as stage_create_event

    prepared = []
    for i, ev in enumerate(events):
        title = str(ev.get("title") or "").strip()
        if not title:
            raise CalendarError(f"Event {i + 1}: a title is required.")
        start_raw = ev.get("start")
        if not start_raw:
            raise CalendarError(f"Event {i + 1} ('{title}'): a start time is required.")
        start_dt = _parse_datetime(start_raw)
        end_raw = ev.get("end")
        end_dt = (
            _parse_datetime(end_raw) if end_raw
            else start_dt + timedelta(minutes=CALENDAR_BATCH_DEFAULT_DURATION_MINUTES)
        )
        if end_dt <= start_dt:
            raise CalendarError(f"Event {i + 1} ('{title}'): end time must be after the start time.")
        prepared.append({
            "title": title,
            "start_dt": _localize(start_dt),
            "end_dt": _localize(end_dt),
            "location": str(ev.get("location") or ""),
            "description": str(ev.get("description") or ""),
        })

    # No two proposed events may overlap each other.
    ordered = sorted(prepared, key=lambda p: p["start_dt"])
    for a, b in zip(ordered, ordered[1:]):
        if a["end_dt"] > b["start_dt"]:
            raise CalendarError(
                f"Proposed events overlap: '{a['title']}' ({a['start_dt'].strftime('%H:%M')}-"
                f"{a['end_dt'].strftime('%H:%M')}) and '{b['title']}' "
                f"({b['start_dt'].strftime('%H:%M')}-{b['end_dt'].strftime('%H:%M')}). "
                f"Adjust the times so no two proposed events overlap, then retry."
            )

    # None of the proposed events may collide with what's already on the
    # calendar - checked across every calendar on the account, not just
    # the target one for this batch.
    window_start = min(p["start_dt"] for p in prepared).replace(tzinfo=None)
    window_end = max(p["end_dt"] for p in prepared).replace(tzinfo=None)
    busy = _get_busy_intervals(window_start, window_end)  # fetched ONCE for the whole
                                                            # batch, not once per event -
                                                            # the earlier per-event fetch
                                                            # multiplied CalDAV round-trips
                                                            # (and timeout risk) by batch size
    for p in prepared:
        for busy_start, busy_end, busy_title in busy:
            if _overlaps(p["start_dt"], p["end_dt"], busy_start, busy_end):
                raise CalendarError(
                    f"'{p['title']}' ({p['start_dt'].strftime('%a %H:%M')}-{p['end_dt'].strftime('%H:%M')}) "
                    f"conflicts with an existing event '{busy_title}' ({busy_start.strftime('%H:%M')}-"
                    f"{busy_end.strftime('%H:%M')}). Adjust the proposed schedule around it, then retry."
                )

    def apply_fn() -> str:
        results = []
        for p in prepared:
            # Duplicate guard, same as stage_create_event - protects a
            # retry-after-failure from double-booking events that
            # actually succeeded on iCloud's side in an earlier attempt.
            if _find_matching_event(cal, p["title"], p["start_dt"]) is not None:
                results.append(f"'{p['title']}' already exists - skipped (not duplicated).")
                continue
            try:
                cal.add_event(
                    dtstart=p["start_dt"], dtend=p["end_dt"], summary=p["title"],
                    location=p["location"] or None, description=p["description"] or None,
                )
            except Exception as e:
                raise CalendarError(
                    f"Failed to create '{p['title']}': {e}. {len(results)} of {len(prepared)} "
                    f"event(s) in this batch were created before this failure - confirming "
                    f"again will safely skip those and only create the rest."
                )
            results.append(f"Created '{p['title']}' {p['start_dt'].strftime('%H:%M')}-{p['end_dt'].strftime('%H:%M')}.")
        return "\n".join(results)

    lines = [
        f"- {p['title']}: {p['start_dt'].strftime('%a %Y-%m-%d %H:%M')}-{p['end_dt'].strftime('%H:%M')}"
        + (f" at {p['location']}" if p["location"] else "")
        for p in ordered
    ]
    desc = f"CREATE {len(prepared)} events in calendar '{cal.name or '(unnamed)'}':\n" + "\n".join(lines)
    return _stage(desc, apply_fn)


def _search_calendars_for_uid(calendars, uid: str):
    """Scan a list of caldav Calendar objects for an event with the given
    UID, using date_search() + in-memory filtering (see
    _get_event_by_uid_via_search's docstring for why event_by_uid() isn't
    used). Returns (event, calendar) or (None, None)."""
    start_dt = datetime.now() - timedelta(days=CALENDAR_UID_LOOKUP_LOOKBACK_DAYS)
    end_dt = datetime.now() + timedelta(days=CALENDAR_UID_LOOKUP_LOOKAHEAD_DAYS)
    for cal in calendars:
        events = _date_search_with_retry(cal, start_dt, end_dt, f"Could not search '{cal.name or '(unnamed)'}' for the event")
        for event in events:
            try:
                component = event.icalendar_component
            except Exception:
                continue
            if component is not None and str(component.get("uid", "")) == uid:
                return event, cal
    return None, None


def _get_event_by_uid_via_search(calendar_name: str, uid: str):
    """Find an event by UID. If calendar_name is given, only that calendar
    is scanned; otherwise EVERY calendar on the account is scanned - this
    matters because list_events/search_events default to searching every
    calendar too, so an event found by search (with no calendar_name) can
    live on ANY of the user's calendars, not necessarily whichever one
    _resolve_calendar(None) would guess as "the default" (iCloud's first
    returned calendar). Scanning only the guessed default calendar was a
    real bug: a UID genuinely found seconds earlier by search_events could
    still fail to resolve here purely because it lives on a different
    calendar than the default. Returns (event, calendar) or (None, None)."""
    calendars = [_resolve_calendar(calendar_name)] if calendar_name else _get_calendars()
    return _search_calendars_for_uid(calendars, uid)


def _find_event_with_fallback(calendar_name: str, event_uid: str):
    """Resolve an event by UID, scanning all calendars unless calendar_name
    narrows it. Falls back to the sole most-recent list/search result if
    either the UID wasn't found OR calendar_name itself doesn't match any
    real calendar - the model has been observed fabricating BOTH a
    plausible-looking UID and a plausible-looking calendar name on the same
    call, not just the UID. A bad calendar_name must not hard-fail before
    this fallback gets a chance to run. Returns (event, calendar,
    resolved_uid, was_corrected)."""
    try:
        event, cal = _get_event_by_uid_via_search(calendar_name, event_uid)
    except CalendarError:
        # calendar_name didn't resolve to a real calendar - don't give up,
        # fall through to the last-result rescue below just like an
        # unresolved UID would.
        event, cal = None, None

    if event is not None:
        return event, cal, event_uid, False

    with _last_results_lock:
        candidates = list(_last_results)

    if calendar_name:
        try:
            target_cal = _resolve_calendar(calendar_name)
            pool = [c for c in candidates if c.get("calendar") == (target_cal.name or "(unnamed)")]
        except CalendarError:
            # calendar_name was fabricated too - don't filter by it, just
            # use whatever was in the last search/list result as-is.
            pool = candidates
    else:
        pool = candidates

    if len(pool) == 1:
        corrected_uid = pool[0]["uid"]
        corrected_cal_name = pool[0].get("calendar")
        corrected_event, corrected_cal = _get_event_by_uid_via_search(corrected_cal_name, corrected_uid)
        if corrected_event is not None:
            return corrected_event, corrected_cal, corrected_uid, corrected_uid != event_uid

    raise CalendarError(
        f"Could not find an event with uid '{event_uid}' in the last "
        f"{CALENDAR_UID_LOOKUP_LOOKBACK_DAYS}-{CALENDAR_UID_LOOKUP_LOOKAHEAD_DAYS} day "
        f"window. Call calendar_list_events or calendar_search_events again "
        f"to get the exact uid, then retry using that exact value."
    )


def stage_edit_event(event_uid: str, title: str = None, start: str = None, end: str = None,
                      location: str = None, description: str = None, calendar_name: str = None) -> str:
    if not event_uid or not event_uid.strip():
        raise CalendarError("event_uid is required - use calendar_list_events or "
                             "calendar_search_events to find it first. Never guess a UID.")
    event, cal, event_uid, corrected = _find_event_with_fallback(calendar_name, event_uid.strip())

    changes = []
    if title:
        changes.append(f"title -> '{title}'")
    if start:
        changes.append(f"start -> {start}")
    if end:
        changes.append(f"end -> {end}")
    if location is not None:
        changes.append(f"location -> '{location}'")
    if description is not None:
        changes.append("description updated")
    if not changes:
        raise CalendarError("No fields to change were provided.")

    def apply_fn() -> str:
        try:
            # caldav 2.0+ no longer bundles vobject by default - icalendar
            # is the properly-supported dependency, so edits go through
            # edit_icalendar_component() (caldav's own documented pattern
            # for this) instead of the older vobject-based API.
            with event.edit_icalendar_component() as comp:
                if title:
                    comp["SUMMARY"] = title
                if start:
                    comp["DTSTART"] = icalendar.vDDDTypes(_localize(_parse_datetime(start)))
                if end:
                    comp["DTEND"] = icalendar.vDDDTypes(_localize(_parse_datetime(end)))
                if location is not None:
                    comp["LOCATION"] = location
                if description is not None:
                    comp["DESCRIPTION"] = description
            event.save()
        except Exception as e:
            raise CalendarError(f"Failed to edit event: {e}")
        return f"Edited event {event_uid}: {', '.join(changes)}."

    desc = f"EDIT event {event_uid}: {', '.join(changes)}"
    if corrected:
        desc += " (auto-matched to the event just shown in your last search/list result)"
    return _stage(desc, apply_fn)


def stage_delete_event(event_uid: str, calendar_name: str = None) -> str:
    if not event_uid or not event_uid.strip():
        raise CalendarError("event_uid is required - use calendar_list_events or "
                             "calendar_search_events to find it first. Never guess a UID.")
    event, cal, event_uid, corrected = _find_event_with_fallback(calendar_name, event_uid.strip())
    try:
        title = _ical_field(event.icalendar_component, "summary", event_uid)
    except Exception as e:
        raise CalendarError(f"Could not read event {event_uid}: {e}")

    def apply_fn() -> str:
        try:
            event.delete()
        except Exception as e:
            raise CalendarError(f"Failed to delete event: {e}")
        return f"Deleted '{title}'."

    desc = f"DELETE event '{title}' (uid {event_uid})"
    if corrected:
        desc += " (auto-matched to the event just shown in your last search/list result)"
    return _stage(desc, apply_fn)


def has_pending_change() -> bool:
    """Whether a staged (unconfirmed, unexpired) calendar change currently
    exists. Used by main.py to detect when the model re-stages a change
    instead of calling calendar_confirm_pending after the user has already
    confirmed - see the matching guard in agent_loop, added after live
    testing showed the model looping: re-staging the identical change on
    every 'confirm' instead of ever actually applying it."""
    with _pending_lock:
        return _peek_pending() is not None


def confirm_pending() -> str:
    global _pending_change
    with _pending_lock:
        pending = _peek_pending()
        if pending is None:
            return "Nothing is staged to confirm. It may have expired - propose the change again."
        last_error = None
        for attempt in range(CALENDAR_WRITE_RETRIES + 1):
            try:
                result = pending["apply"]()
            except CalendarError as e:
                last_error = e
                if attempt < CALENDAR_WRITE_RETRIES:
                    time.sleep(_CALENDAR_RETRY_DELAY_SECONDS)
                continue
            _pending_change = None
            return result
        # All attempts failed - deliberately do NOT clear _pending_change
        # here (unlike before). That lets the user just say "confirm"
        # again later instead of the model re-describing/re-staging it
        # from scratch, and apply_fn's own duplicate guard (for creates)
        # makes that safe even if an earlier attempt actually succeeded
        # on iCloud's side despite the client-side error. The existing
        # TTL in _peek_pending() still expires it eventually.
        raise last_error


def cancel_pending() -> str:
    global _pending_change
    with _pending_lock:
        pending = _peek_pending()
        if pending is None:
            return "Nothing is staged to cancel."
        desc = pending["description"]
        _pending_change = None
        return f"Cancelled: {desc}"
