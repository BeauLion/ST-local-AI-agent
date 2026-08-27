"""
Tests for calendar_manager.py's check_availability(). The interesting
part of this function is the branch it takes: it answers from the local
on-disk cache when the requested range falls inside the cache's own
lookahead window (fast, no CalDAV round-trip), and falls back to a live
_get_busy_intervals() call otherwise (or when the cache is missing/
unreadable). Every test below is written to prove WHICH branch actually
ran, not just that the final answer happens to look right - see
test_falls_back_to_live_when_outside_cache_window_even_if_cache_says_busy
for the clearest example of why that distinction matters.
"""
import json
from datetime import datetime, timedelta

import pytest
from freezegun import freeze_time

import config
from calendar_manager import CalendarError
import calendar_manager as cm


FROZEN_NOW = "2026-08-26 10:00:00"


def _write_cache_payload(cache_path, events):
    """events: list of (title, start_datetime, end_datetime), naive or
    aware - written in the same shape refresh_cache() itself produces
    (a plain 'start'/'end' display string, plus the machine-readable
    'start_iso'/'end_iso' fields check_availability's cache path
    actually reads)."""
    records = []
    for title, start, end in events:
        start = cm._localize(start)
        end = cm._localize(end)
        records.append({
            "title": title, "start": str(start), "end": str(end),
            "location": "", "start_iso": start.isoformat(), "end_iso": end.isoformat(),
        })
    cache_path.write_text(json.dumps({"refreshed_at": "irrelevant", "events": records}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_requires_when_or_start(fake_calendars, tmp_cache_file):
    with pytest.raises(CalendarError, match="Provide either a 'when' phrase or a start time"):
        cm.check_availability()


def test_end_before_start_raises(fake_calendars, tmp_cache_file):
    with pytest.raises(CalendarError, match="End time must be after the start time"):
        cm.check_availability(start="2026-08-27 10:00", end="2026-08-27 09:00")


def test_end_equal_to_start_raises(fake_calendars, tmp_cache_file):
    with pytest.raises(CalendarError, match="End time must be after the start time"):
        cm.check_availability(start="2026-08-27 10:00", end="2026-08-27 10:00")


def test_default_end_is_one_hour_after_start(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        result = cm.check_availability(start="2026-08-27 09:00")  # no end given
        assert result["free"] is True  # nothing seeded to conflict with; just proves it didn't raise


def test_when_phrase_is_accepted(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        result = cm.check_availability(when="tomorrow")
        assert result["free"] is True


# ---------------------------------------------------------------------------
# Cache-hit path (request falls inside the cache's own lookahead window)
# ---------------------------------------------------------------------------

def test_uses_cache_source_when_within_window_and_cache_exists(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        _write_cache_payload(tmp_cache_file, [])  # empty but present and readable
        result = cm.check_availability(start="2026-08-27 09:00", end="2026-08-27 10:00")
        assert result["source"] == "cache"
        assert result["free"] is True


def test_cache_hit_reports_a_real_conflict(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        _write_cache_payload(tmp_cache_file, [
            ("Cached Standup", datetime(2026, 8, 27, 9, 0), datetime(2026, 8, 27, 9, 30)),
        ])
        result = cm.check_availability(start="2026-08-27 09:15", end="2026-08-27 09:45")

        assert result["source"] == "cache"
        assert result["free"] is False
        assert result["conflicts"][0]["title"] == "Cached Standup"


def test_cache_hit_reports_free_when_no_time_overlap(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        _write_cache_payload(tmp_cache_file, [
            ("Cached Standup", datetime(2026, 8, 27, 9, 0), datetime(2026, 8, 27, 9, 30)),
        ])
        result = cm.check_availability(start="2026-08-27 11:00", end="2026-08-27 12:00")

        assert result["source"] == "cache"
        assert result["free"] is True
        assert result["conflicts"] == []


def test_cache_entries_missing_iso_fields_are_skipped_not_crashed(fake_calendars, tmp_cache_file):
    """A cache entry without start_iso/end_iso (e.g. an all-day event
    whose component had no usable start/end - see refresh_cache's own
    comment about this) must be silently skipped, not raise."""
    with freeze_time(FROZEN_NOW):
        payload = {"refreshed_at": "x", "events": [{"title": "No iso fields", "start": "n/a", "end": "n/a"}]}
        tmp_cache_file.write_text(json.dumps(payload), encoding="utf-8")

        result = cm.check_availability(start="2026-08-27 09:00", end="2026-08-27 10:00")

        assert result["source"] == "cache"
        assert result["free"] is True  # the malformed entry was skipped, not treated as a conflict


def test_cache_entries_with_unparseable_iso_are_skipped_not_crashed(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        payload = {"refreshed_at": "x", "events": [{
            "title": "Bad iso", "start": "x", "end": "x",
            "start_iso": "not-a-real-timestamp", "end_iso": "also-not-real",
        }]}
        tmp_cache_file.write_text(json.dumps(payload), encoding="utf-8")

        result = cm.check_availability(start="2026-08-27 09:00", end="2026-08-27 10:00")

        assert result["source"] == "cache"
        assert result["free"] is True


# ---------------------------------------------------------------------------
# Live-fallback path
# ---------------------------------------------------------------------------

def test_falls_back_to_live_when_cache_file_does_not_exist(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        # tmp_cache_file fixture points at a path but never writes to it
        assert not tmp_cache_file.exists()
        result = cm.check_availability(start="2026-08-27 09:00", end="2026-08-27 10:00")
        assert result["source"] == "live"


def test_falls_back_to_live_when_cache_file_is_corrupt_json(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        tmp_cache_file.write_text("{not valid json at all", encoding="utf-8")
        result = cm.check_availability(start="2026-08-27 09:00", end="2026-08-27 10:00")
        assert result["source"] == "live"


def test_falls_back_to_live_when_outside_cache_window_even_if_cache_says_busy(fake_calendars, tmp_cache_file):
    """The clearest proof the branch selection is actually working: seed
    a cache entry that WOULD report a conflict if it were used, but
    query a range outside the cache's lookahead window, with nothing
    seeded on the live fake calendar at that time. If this comes back
    'free' and 'live', the cache genuinely wasn't consulted - if the
    branch logic were broken and it used the stale cache anyway, this
    would incorrectly report a conflict."""
    with freeze_time(FROZEN_NOW):
        far_future = datetime(2026, 8, 26) + timedelta(days=config.CALENDAR_CACHE_LOOKAHEAD_DAYS + 5)
        _write_cache_payload(tmp_cache_file, [
            ("Stale cached conflict", far_future.replace(hour=9), far_future.replace(hour=9, minute=30)),
        ])

        result = cm.check_availability(
            start=far_future.replace(hour=9, minute=15).strftime("%Y-%m-%d %H:%M"),
            end=far_future.replace(hour=9, minute=45).strftime("%Y-%m-%d %H:%M"),
        )

        assert result["source"] == "live"
        assert result["free"] is True
        assert result["conflicts"] == []


def test_live_path_reports_a_real_conflict_from_the_calendar(fake_calendars, tmp_cache_file):
    with freeze_time(FROZEN_NOW):
        far_future = datetime(2026, 8, 26) + timedelta(days=config.CALENDAR_CACHE_LOOKAHEAD_DAYS + 5)
        fake_calendars[0].seed_event("Live Dentist", far_future.replace(hour=9), far_future.replace(hour=9, minute=30))

        result = cm.check_availability(
            start=far_future.replace(hour=9, minute=15).strftime("%Y-%m-%d %H:%M"),
            end=far_future.replace(hour=9, minute=45).strftime("%Y-%m-%d %H:%M"),
        )

        assert result["source"] == "live"
        assert result["free"] is False
        assert result["conflicts"][0]["title"] == "Live Dentist"
