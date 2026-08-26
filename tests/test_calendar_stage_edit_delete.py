"""
Tests for calendar_manager.py's stage_edit_event / stage_delete_event and
the UID-resolution logic behind both of them (_find_event_with_fallback,
_get_event_by_uid_via_search, _search_calendars_for_uid).

This is the most convoluted state in the module, so the tests are
organized around the actual decision tree in _find_event_with_fallback:

  1. Direct UID search succeeds -> use it, not "corrected".
  2. Direct search fails (bad uid, OR bad calendar_name) -> fall back to
     _last_results (populated by the most recent list/search call).
     2a. Exactly one candidate in the (possibly calendar_name-filtered)
         pool -> re-resolve using that candidate's real uid+calendar.
     2b. Zero or multiple candidates -> nothing safe to guess -> raise.

The "corrected" flag that comes back specifically means "the uid we
ended up using differs from the uid we were given" - NOT "any fallback
logic ran at all". The last test in this file exists specifically to
pin down that distinction, since it's easy to misread from the source
alone.
"""
from datetime import datetime

import pytest

from calendar_manager import CalendarError
import calendar_manager as cm
from tests.fakes.fake_caldav import FakeCalendar


# ---------------------------------------------------------------------------
# Input validation - before any UID resolution happens
# ---------------------------------------------------------------------------

def test_edit_requires_event_uid(fake_calendars):
    with pytest.raises(CalendarError, match="event_uid is required"):
        cm.stage_edit_event(event_uid="", title="New title")


def test_delete_requires_event_uid(fake_calendars):
    with pytest.raises(CalendarError, match="event_uid is required"):
        cm.stage_delete_event(event_uid="   ")


def test_edit_requires_at_least_one_field_to_change(fake_calendars):
    ev = fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    real_uid = str(ev.icalendar_component.get("uid"))

    with pytest.raises(CalendarError, match="No fields to change"):
        cm.stage_edit_event(event_uid=real_uid)


def test_edit_rejects_unparseable_date_before_staging(fake_calendars):
    ev = fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    real_uid = str(ev.icalendar_component.get("uid"))

    with pytest.raises(CalendarError, match="Could not parse date/time"):
        cm.stage_edit_event(event_uid=real_uid, start="not a date")
    assert cm.has_pending_change() is False


# ---------------------------------------------------------------------------
# Direct UID match - the simple, non-fallback path
# ---------------------------------------------------------------------------

def test_edit_by_real_uid_updates_only_given_fields(fake_calendars):
    ev = fake_calendars[0].seed_event(
        "Standup", datetime(2026, 9, 1, 9, 0), datetime(2026, 9, 1, 9, 30),
        location="Office",
    )
    real_uid = str(ev.icalendar_component.get("uid"))

    result = cm.stage_edit_event(event_uid=real_uid, title="Renamed")
    assert "auto-matched" not in result  # direct match, no fallback needed

    cm.confirm_pending()

    comp = ev.icalendar_component
    assert str(comp.get("summary")) == "Renamed"
    assert str(comp.get("location")) == "Office"  # untouched field preserved


def test_edit_can_update_multiple_fields_at_once(fake_calendars):
    ev = fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0), datetime(2026, 9, 1, 9, 30))
    real_uid = str(ev.icalendar_component.get("uid"))

    cm.stage_edit_event(
        event_uid=real_uid, title="Renamed", location="New Office",
        start="2026-09-01 10:00", end="2026-09-01 10:30",
    )
    cm.confirm_pending()

    comp = ev.icalendar_component
    assert str(comp.get("summary")) == "Renamed"
    assert str(comp.get("location")) == "New Office"
    assert comp.get("dtstart").dt.replace(tzinfo=None) == datetime(2026, 9, 1, 10, 0)


def test_delete_by_real_uid_removes_event(fake_calendars):
    ev = fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    real_uid = str(ev.icalendar_component.get("uid"))

    result = cm.stage_delete_event(event_uid=real_uid)
    assert "auto-matched" not in result
    cm.confirm_pending()

    assert len(fake_calendars[0].events) == 0


# ---------------------------------------------------------------------------
# UID fallback - fabricated UID, resolved via the last list/search result
# ---------------------------------------------------------------------------

def test_edit_with_fabricated_uid_falls_back_when_exactly_one_recent_result(fake_calendars):
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    cm.list_events(start="2026-09-01", end="2026-09-02")  # populates _last_results

    result = cm.stage_edit_event(event_uid="fabricated-not-a-real-uid", title="Renamed")

    assert "auto-matched to the event just shown" in result
    cm.confirm_pending()
    assert str(fake_calendars[0].events[0].icalendar_component.get("summary")) == "Renamed"


def test_delete_with_fabricated_uid_falls_back_when_exactly_one_recent_result(fake_calendars):
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    cm.search_events(query="Standup")  # populates _last_results

    result = cm.stage_delete_event(event_uid="fabricated-not-a-real-uid")

    assert "auto-matched to the event just shown" in result
    cm.confirm_pending()
    assert len(fake_calendars[0].events) == 0


def test_fabricated_uid_with_no_prior_search_raises(fake_calendars):
    """No list/search has been called yet this session - _last_results is
    empty, so there's nothing to fall back to."""
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))

    with pytest.raises(CalendarError, match="Could not find an event with uid"):
        cm.stage_edit_event(event_uid="fabricated-not-a-real-uid", title="Renamed")


def test_fabricated_uid_with_multiple_recent_results_raises(fake_calendars):
    """Two-plus candidates in _last_results means guessing which one the
    model meant would be unsafe - must hard-fail instead."""
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    fake_calendars[0].seed_event("Dentist", datetime(2026, 9, 1, 14, 0))
    cm.list_events(start="2026-09-01", end="2026-09-02")  # 2 results

    with pytest.raises(CalendarError, match="Could not find an event with uid"):
        cm.stage_edit_event(event_uid="fabricated-not-a-real-uid", title="Renamed")


# ---------------------------------------------------------------------------
# UID fallback interacting with calendar_name, across multiple calendars
# ---------------------------------------------------------------------------

def test_fallback_narrows_by_valid_calendar_name_across_multiple_calendars(fake_calendars):
    """Two real calendars, both with a same-named event in _last_results.
    Passing the correct calendar_name should narrow the fallback pool
    down to exactly the one in that calendar, not fail as ambiguous."""
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    work.seed_event("Standup", datetime(2026, 9, 1, 9, 0))

    # search_events with no calendar_name scans every calendar, so both
    # same-named events end up in _last_results with different labels.
    cm.search_events(query="Standup")

    result = cm.stage_edit_event(
        event_uid="fabricated-not-a-real-uid", calendar_name="Work", title="Renamed",
    )
    assert "auto-matched to the event just shown" in result
    cm.confirm_pending()

    assert str(work.events[0].icalendar_component.get("summary")) == "Renamed"
    assert str(fake_calendars[0].events[0].icalendar_component.get("summary")) == "Standup"  # untouched


def test_fallback_pool_empty_after_filtering_by_valid_calendar_name_raises(fake_calendars):
    """The only _last_results candidate is on 'Home', but the caller
    asked to scope the fallback to 'Work' - filtering leaves zero
    candidates, which must fail rather than silently ignoring the
    requested calendar_name."""
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    cm.list_events(calendar_name="Home", start="2026-09-01", end="2026-09-02")

    with pytest.raises(CalendarError, match="Could not find an event with uid"):
        cm.stage_edit_event(event_uid="fabricated-not-a-real-uid", calendar_name="Work", title="Renamed")


def test_fallback_ignores_a_fabricated_calendar_name_and_uses_last_results_anyway(fake_calendars):
    """calendar_name itself can be fabricated too (module docstring: 'the
    model has been observed fabricating BOTH a plausible-looking UID and
    a plausible-looking calendar name on the same call'). A calendar_name
    that matches no real calendar should NOT be treated as a filter that
    zeroes out the candidate pool - it should be ignored so the single
    remaining candidate can still be used."""
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    cm.list_events(start="2026-09-01", end="2026-09-02")

    result = cm.stage_edit_event(
        event_uid="fabricated-uid", calendar_name="NotARealCalendar", title="Renamed",
    )
    assert "auto-matched to the event just shown" in result
    cm.confirm_pending()
    assert str(fake_calendars[0].events[0].icalendar_component.get("summary")) == "Renamed"


# ---------------------------------------------------------------------------
# UID search scope
# ---------------------------------------------------------------------------

def test_uid_search_scans_every_calendar_when_no_calendar_name_given(fake_calendars):
    """A real uid should resolve even if it lives on a calendar other
    than the one _resolve_calendar(None) would guess as the default -
    this is the exact bug described in _get_event_by_uid_via_search's
    docstring."""
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    ev = work.seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    real_uid = str(ev.icalendar_component.get("uid"))

    # No calendar_name given at all - must still find it on "Work".
    cm.stage_edit_event(event_uid=real_uid, title="Renamed")
    cm.confirm_pending()

    assert str(work.events[0].icalendar_component.get("summary")) == "Renamed"


# ---------------------------------------------------------------------------
# The "corrected" flag's precise meaning
# ---------------------------------------------------------------------------

def test_fabricated_calendar_name_with_real_uid_is_not_reported_as_corrected(fake_calendars):
    """Subtle case: the UID given is REAL, but the calendar_name given is
    FABRICATED. Direct search fails (because _resolve_calendar rejects
    the fake calendar_name before the uid is even checked), so this
    falls through to the _last_results path just like a fabricated UID
    would. But since the fallback ultimately resolves to the SAME uid
    that was given, 'corrected' must be False - it only means 'the uid
    changed', not 'any fallback logic ran'. Getting this wrong would
    make stage_edit_event tell the user 'auto-matched to the event just
    shown' for a UID they actually specified correctly, which would be
    confusing."""
    work = FakeCalendar(name="Work")
    fake_calendars.append(work)
    ev = work.seed_event("Standup", datetime(2026, 9, 1, 9, 0))
    real_uid = str(ev.icalendar_component.get("uid"))

    cm.search_events(query="Standup")  # populates _last_results with the Work event

    result = cm.stage_edit_event(
        event_uid=real_uid, calendar_name="NotARealCalendar", title="Renamed",
    )

    assert "auto-matched" not in result  # uid didn't change - not "corrected"
    cm.confirm_pending()
    assert str(work.events[0].icalendar_component.get("summary")) == "Renamed"
