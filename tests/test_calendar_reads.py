"""
Tests for calendar_manager.py's read-only functions: list_calendar_names,
list_events, search_events.

Worth calling out up front (verified against the real module before
writing any assertions, not assumed from reading the source): when NO
calendar_name is given, list_events() only queries the single resolved
DEFAULT calendar (_resolve_calendar(None)), while search_events() scans
EVERY calendar on the account. This is a real, easy-to-miss asymmetry
between the two functions - not a bug, just a behavior worth a test
locking it down, since it means "list my events" and "search my events"
can have different blind spots depending on which calendar something
lives on. See test_list_events_only_scans_the_default_calendar and
test_search_events_scans_every_calendar_by_default below.
"""
from datetime import datetime, timedelta

import pytest
from freezegun import freeze_time

import config
from calendar_manager import CalendarError
import calendar_manager as cm
from tests.fakes.fake_caldav import FakeCalendar


FROZEN_NOW = "2026-08-26 10:00:00"


# ---------------------------------------------------------------------------
# list_calendar_names
# ---------------------------------------------------------------------------

def test_list_calendar_names_returns_all_calendar_names(fake_calendars):
    fake_calendars.append(FakeCalendar(name="Work"))
    assert cm.list_calendar_names() == ["Home", "Work"]


def test_list_calendar_names_labels_unnamed_calendar(fake_calendars):
    fake_calendars[0].name = None
    assert cm.list_calendar_names() == ["(unnamed)"]


# ---------------------------------------------------------------------------
# list_events - date-range resolution
# ---------------------------------------------------------------------------

def test_list_events_with_no_args_defaults_to_now_through_lookahead_days(fake_calendars):
    with freeze_time(FROZEN_NOW):
        fake_calendars[0].seed_event("Within range", datetime(2026, 9, 5, 9, 0))   # +10 days
        fake_calendars[0].seed_event("Out of range", datetime(2026, 9, 20, 9, 0))  # +25 days, past the 14-day default

        events = cm.list_events()

        titles = [e["title"] for e in events]
        assert "Within range" in titles
        assert "Out of range" not in titles


def test_list_events_with_when_phrase_resolves_via_resolve_when(fake_calendars):
    with freeze_time(FROZEN_NOW):
        fake_calendars[0].seed_event("Tomorrow's event", datetime(2026, 8, 27, 9, 0))
        events = cm.list_events(when="tomorrow")
        assert [e["title"] for e in events] == ["Tomorrow's event"]


def test_list_events_start_without_end_defaults_to_lookahead_window(fake_calendars):
    fake_calendars[0].seed_event("Just inside", datetime(2026, 9, 1) + timedelta(days=config.CALENDAR_DEFAULT_LOOKAHEAD_DAYS - 1))
    fake_calendars[0].seed_event("Just outside", datetime(2026, 9, 1) + timedelta(days=config.CALENDAR_DEFAULT_LOOKAHEAD_DAYS + 1))

    events = cm.list_events(start="2026-09-01")

    titles = [e["title"] for e in events]
    assert "Just inside" in titles
    assert "Just outside" not in titles


def test_list_events_same_start_and_end_date_is_treated_as_the_whole_day(fake_calendars):
    """A single day is naturally expressed as the same date for both start
    and end (e.g. 'tomorrow' -> start='2026-09-01', end='2026-09-01').
    This should cover the whole day, not be rejected as a zero-length
    range."""
    fake_calendars[0].seed_event("All day standup", datetime(2026, 9, 1, 9, 0), datetime(2026, 9, 1, 9, 30))

    events = cm.list_events(start="2026-09-01", end="2026-09-01")

    assert [e["title"] for e in events] == ["All day standup"]


def test_list_events_normal_start_end_range(fake_calendars):
    fake_calendars[0].seed_event("In range", datetime(2026, 9, 5, 9, 0))
    fake_calendars[0].seed_event("Before range", datetime(2026, 8, 1, 9, 0))
    fake_calendars[0].seed_event("After range", datetime(2026, 10, 1, 9, 0))

    events = cm.list_events(start="2026-09-01", end="2026-09-30")

    assert [e["title"] for e in events] == ["In range"]


def test_list_events_returns_empty_list_when_nothing_found(fake_calendars):
    assert cm.list_events(start="2026-09-01", end="2026-09-02") == []


def test_list_events_sorts_results_by_start_time(fake_calendars):
    fake_calendars[0].seed_event("Later", datetime(2026, 9, 1, 14, 0))
    fake_calendars[0].seed_event("Earlier", datetime(2026, 9, 1, 9, 0))

    events = cm.list_events(start="2026-09-01", end="2026-09-02")

    assert [e["title"] for e in events] == ["Earlier", "Later"]


# ---------------------------------------------------------------------------
# list_events - calendar scope (the single-calendar side of the asymmetry)
# ---------------------------------------------------------------------------

def test_list_events_with_explicit_calendar_name_only_queries_that_calendar(fake_calendars):
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    fake_calendars[0].seed_event("Home event", datetime(2026, 9, 1, 9, 0))
    work.seed_event("Work event", datetime(2026, 9, 1, 9, 0))

    events = cm.list_events(calendar_name="Work", start="2026-09-01", end="2026-09-02")

    assert [e["title"] for e in events] == ["Work event"]


def test_list_events_only_scans_the_default_calendar_not_every_calendar(fake_calendars):
    """DOCUMENTED ASYMMETRY: with no calendar_name given, list_events only
    looks at the resolved default calendar (here, 'Home', matching
    config.CALENDAR_DEFAULT_NAME) - an event that lives only on a
    non-default calendar will NOT show up, even though it exists on the
    account. Contrast with test_search_events_scans_every_calendar_by_default
    below."""
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    fake_calendars[0].seed_event("Home event", datetime(2026, 9, 1, 9, 0))
    work.seed_event("Work event", datetime(2026, 9, 1, 9, 0))

    events = cm.list_events(start="2026-09-01", end="2026-09-02")  # no calendar_name

    assert [e["title"] for e in events] == ["Home event"]


# ---------------------------------------------------------------------------
# list_events - _last_results bookkeeping
# ---------------------------------------------------------------------------

def test_list_events_replaces_last_results_rather_than_accumulating(fake_calendars):
    fake_calendars[0].seed_event("First call", datetime(2026, 9, 1, 9, 0))
    cm.list_events(start="2026-09-01", end="2026-09-02")

    fake_calendars[0].events.clear()
    fake_calendars[0].seed_event("Second call", datetime(2026, 9, 5, 9, 0))
    cm.list_events(start="2026-09-05", end="2026-09-06")

    with cm._last_results_lock:
        titles = [r["title"] for r in cm._last_results]
    assert titles == ["Second call"]  # "First call" must not linger


# ---------------------------------------------------------------------------
# search_events - validation
# ---------------------------------------------------------------------------

def test_search_requires_a_non_empty_query(fake_calendars):
    with pytest.raises(CalendarError, match="A search query is required"):
        cm.search_events(query="")


def test_search_requires_non_whitespace_query(fake_calendars):
    with pytest.raises(CalendarError, match="A search query is required"):
        cm.search_events(query="   ")


# ---------------------------------------------------------------------------
# search_events - matching behavior
# ---------------------------------------------------------------------------

def test_search_matches_title_case_insensitively(fake_calendars):
    fake_calendars[0].seed_event("Team Standup", datetime(2026, 9, 1, 9, 0))
    results = cm.search_events(query="standup", start="2026-09-01", end="2026-09-02")
    assert [r["title"] for r in results] == ["Team Standup"]


def test_search_matches_against_location(fake_calendars):
    fake_calendars[0].seed_event(
        "1:1", datetime(2026, 9, 1, 9, 0), location="Conference Room B",
    )
    results = cm.search_events(query="conference room", start="2026-09-01", end="2026-09-02")
    assert [r["title"] for r in results] == ["1:1"]


def test_search_matches_against_description(fake_calendars):
    fake_calendars[0].seed_event(
        "Planning", datetime(2026, 9, 1, 9, 0), description="Discuss Q4 roadmap",
    )
    results = cm.search_events(query="roadmap", start="2026-09-01", end="2026-09-02")
    assert [r["title"] for r in results] == ["Planning"]


def test_search_with_no_match_returns_empty_list(fake_calendars):
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    assert cm.search_events(query="nonexistent", start="2026-09-01", end="2026-09-02") == []


def test_search_with_when_phrase_resolves_via_resolve_when(fake_calendars):
    with freeze_time(FROZEN_NOW):
        fake_calendars[0].seed_event("Standup", datetime(2026, 8, 27, 9, 0))
        results = cm.search_events(query="standup", when="tomorrow")
        assert len(results) == 1


def test_search_default_date_range_uses_lookback_and_lookahead_config(fake_calendars):
    with freeze_time(FROZEN_NOW):
        just_inside_lookback = datetime(2026, 8, 26) - timedelta(days=config.CALENDAR_SEARCH_LOOKBACK_DAYS - 1)
        just_outside_lookback = datetime(2026, 8, 26) - timedelta(days=config.CALENDAR_SEARCH_LOOKBACK_DAYS + 1)
        fake_calendars[0].seed_event("In lookback", just_inside_lookback)
        fake_calendars[0].seed_event("Before lookback", just_outside_lookback)

        results = cm.search_events(query="lookback")

        titles = [r["title"] for r in results]
        assert "In lookback" in titles
        assert "Before lookback" not in titles


def test_search_sorts_results_by_start_time_across_calendars(fake_calendars):
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    work.seed_event("Later meeting", datetime(2026, 9, 1, 14, 0))
    fake_calendars[0].seed_event("Earlier meeting", datetime(2026, 9, 1, 9, 0))

    results = cm.search_events(query="meeting", start="2026-09-01", end="2026-09-02")

    assert [r["title"] for r in results] == ["Earlier meeting", "Later meeting"]


# ---------------------------------------------------------------------------
# search_events - calendar scope (the all-calendars side of the asymmetry)
# ---------------------------------------------------------------------------

def test_search_events_scans_every_calendar_by_default(fake_calendars):
    """Contrast with test_list_events_only_scans_the_default_calendar_not_every_calendar
    above - search_events with no calendar_name checks EVERY calendar on
    the account, not just the default one."""
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    fake_calendars[0].seed_event("Home event", datetime(2026, 9, 1, 9, 0))
    work.seed_event("Work event", datetime(2026, 9, 1, 9, 0))

    results = cm.search_events(query="event", start="2026-09-01", end="2026-09-02")

    assert {r["title"] for r in results} == {"Home event", "Work event"}


def test_search_events_with_explicit_calendar_name_only_scans_that_one(fake_calendars):
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    fake_calendars[0].seed_event("Home event", datetime(2026, 9, 1, 9, 0))
    work.seed_event("Work event", datetime(2026, 9, 1, 9, 0))

    results = cm.search_events(query="event", calendar_name="Work", start="2026-09-01", end="2026-09-02")

    assert [r["title"] for r in results] == ["Work event"]
