"""
Tests for calendar_manager.py's background context-cache machinery:
refresh_cache() (writes the on-disk cache), get_cached_context()
(formats it for the system-prompt injection), and
start_background_refresh()'s once-per-process guard.

Per the module's own comment (see the section header above refresh_cache
in calendar_manager.py): this cache is context-injection only. A bug
here can only make the ambient "here's what's coming up" text stale -
never cause a wrong tool result or a bad write. These tests treat it
with that same weight - correctness matters, but there's no staging/
confirm safety model to worry about breaking.
"""
import json

import pytest
from freezegun import freeze_time

import config
import calendar_manager as cm
from calendar_manager import CalendarError


FROZEN_NOW = "2026-08-26 10:00:00"


# ---------------------------------------------------------------------------
# refresh_cache
# ---------------------------------------------------------------------------

def test_refresh_writes_a_readable_cache_file(fake_calendars, tmp_cache_file):
    from datetime import datetime
    fake_calendars[0].seed_event("Standup", datetime(2026, 8, 27, 9, 0), datetime(2026, 8, 27, 9, 30))

    with freeze_time(FROZEN_NOW):
        cm.refresh_cache()

    assert tmp_cache_file.exists()
    payload = json.loads(tmp_cache_file.read_text(encoding="utf-8"))
    assert payload["events"][0]["title"] == "Standup"
    assert payload["events"][0]["start_iso"]  # machine-readable field for check_availability
    assert payload["refreshed_at"] == "2026-08-26T10:00:00"


def test_refresh_creates_missing_parent_directory(fake_calendars, tmp_path, monkeypatch):
    """_CACHE_FILE.parent.mkdir(exist_ok=True) should create the cache
    directory on a fresh install where it doesn't exist yet - this is
    NOT covered by the tmp_cache_file fixture, which always creates
    tmp_path itself; this test points at a nested path whose parent
    genuinely doesn't exist."""
    fresh_cache_path = tmp_path / "calendar_data" / "context_cache.json"
    monkeypatch.setattr(cm, "_CACHE_FILE", fresh_cache_path)
    assert not fresh_cache_path.parent.exists()

    cm.refresh_cache()

    assert fresh_cache_path.exists()


def test_refresh_sorts_events_by_start_time(fake_calendars, tmp_cache_file):
    from datetime import datetime
    fake_calendars[0].seed_event("Later", datetime(2026, 8, 27, 14, 0))
    fake_calendars[0].seed_event("Earlier", datetime(2026, 8, 27, 9, 0))

    cm.refresh_cache()

    payload = json.loads(tmp_cache_file.read_text(encoding="utf-8"))
    assert [e["title"] for e in payload["events"]] == ["Earlier", "Later"]


def test_refresh_leaves_no_leftover_temp_files(fake_calendars, tmp_cache_file):
    """refresh_cache writes via a temp file + os.replace() for an atomic
    swap. After a successful refresh, no stray .cache_tmp_* file should
    remain in the cache directory."""
    cm.refresh_cache()

    leftovers = list(tmp_cache_file.parent.glob(".cache_tmp_*"))
    assert leftovers == []


def test_refresh_keeps_previous_cache_when_fetch_fails(fake_calendars, tmp_cache_file, monkeypatch):
    """A failed refresh should degrade to 'slightly stale', never to 'no
    context at all' - the existing cache file must be left untouched."""
    tmp_cache_file.write_text(json.dumps({"refreshed_at": "old", "events": [{"title": "Old event"}]}), encoding="utf-8")

    def raise_error(refresh=False):
        raise CalendarError("simulated connection failure")

    monkeypatch.setattr(cm, "_get_calendars", raise_error)

    cm.refresh_cache()  # should not raise

    payload = json.loads(tmp_cache_file.read_text(encoding="utf-8"))
    assert payload["refreshed_at"] == "old"
    assert payload["events"][0]["title"] == "Old event"


def test_refresh_only_looks_ahead_configured_number_of_days(fake_calendars, tmp_cache_file):
    from datetime import datetime, timedelta
    with freeze_time(FROZEN_NOW):
        just_inside = datetime(2026, 8, 26) + timedelta(days=config.CALENDAR_CACHE_LOOKAHEAD_DAYS - 1)
        just_outside = datetime(2026, 8, 26) + timedelta(days=config.CALENDAR_CACHE_LOOKAHEAD_DAYS + 1)
        fake_calendars[0].seed_event("In range", just_inside)
        fake_calendars[0].seed_event("Out of range", just_outside)

        cm.refresh_cache()

    payload = json.loads(tmp_cache_file.read_text(encoding="utf-8"))
    titles = [e["title"] for e in payload["events"]]
    assert "In range" in titles
    assert "Out of range" not in titles


# ---------------------------------------------------------------------------
# get_cached_context
# ---------------------------------------------------------------------------

def test_returns_empty_string_when_no_cache_file_exists(tmp_cache_file):
    assert not tmp_cache_file.exists()
    assert cm.get_cached_context() == ""


def test_returns_empty_string_when_cache_file_is_corrupt(tmp_cache_file):
    tmp_cache_file.write_text("{not valid json", encoding="utf-8")
    assert cm.get_cached_context() == ""


def test_reports_none_upcoming_when_events_list_is_empty(tmp_cache_file):
    tmp_cache_file.write_text(json.dumps({"refreshed_at": "x", "events": []}), encoding="utf-8")

    result = cm.get_cached_context()

    assert "[UPCOMING CALENDAR EVENTS]" in result
    assert f"None in the next {config.CALENDAR_CACHE_LOOKAHEAD_DAYS} days" in result


def test_formats_events_with_location(tmp_cache_file):
    payload = {"refreshed_at": "x", "events": [
        {"title": "Standup", "start": "2026-08-27 09:00:00", "end": "2026-08-27 09:30:00", "location": "Office"},
    ]}
    tmp_cache_file.write_text(json.dumps(payload), encoding="utf-8")

    result = cm.get_cached_context()

    assert "2026-08-27 09:00:00 to 2026-08-27 09:30:00: Standup @ Office" in result


def test_formats_events_without_location_omits_the_at_sign(tmp_cache_file):
    payload = {"refreshed_at": "x", "events": [
        {"title": "Standup", "start": "2026-08-27 09:00:00", "end": "2026-08-27 09:30:00", "location": ""},
    ]}
    tmp_cache_file.write_text(json.dumps(payload), encoding="utf-8")

    result = cm.get_cached_context()

    assert "Standup" in result
    assert "@" not in result


def test_truncates_to_configured_max_events_in_context(tmp_cache_file):
    events = [
        {"title": f"Event {i}", "start": f"2026-08-{27 + i:02d} 09:00:00", "end": f"2026-08-{27+i:02d} 09:30:00", "location": ""}
        for i in range(config.CALENDAR_CACHE_MAX_EVENTS_IN_CONTEXT + 5)
    ]
    tmp_cache_file.write_text(json.dumps({"refreshed_at": "x", "events": events}), encoding="utf-8")

    result = cm.get_cached_context()

    shown = result.count("Event ")
    assert shown == config.CALENDAR_CACHE_MAX_EVENTS_IN_CONTEXT


# ---------------------------------------------------------------------------
# start_background_refresh - idempotency guard only (not the actual loop)
# ---------------------------------------------------------------------------

def test_start_background_refresh_only_starts_one_thread(monkeypatch):
    """Calling this more than once (e.g. a double app-startup call) must
    not spawn a second polling thread. The loop itself sleeps for
    CALENDAR_CACHE_REFRESH_MINUTES and runs forever, so this test never
    lets a real thread start - it only checks the guard flag's effect on
    how many Thread objects get created."""
    monkeypatch.setattr(cm, "_background_thread_started", False)
    started_threads = []

    class FakeThread:
        def __init__(self, *args, **kwargs):
            started_threads.append((args, kwargs))

        def start(self):
            pass  # deliberately never actually runs _loop()

    monkeypatch.setattr(cm.threading, "Thread", FakeThread)

    cm.start_background_refresh()
    cm.start_background_refresh()

    assert len(started_threads) == 1
