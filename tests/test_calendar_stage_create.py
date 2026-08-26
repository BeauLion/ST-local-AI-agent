"""
Tests for calendar_manager.py's create-staging functions:
stage_create_event (single event) and stage_create_events_batch
(multiple events staged as one pending change).

Covers three kinds of behavior:
  1. Validation that happens BEFORE anything is staged (bad title, bad
     dates, bad calendar name, batch size limits).
  2. Conflict detection that's specific to batch creation (proposed
     events overlapping each other, or overlapping something already on
     the calendar).
  3. The duplicate-write guard in each apply_fn - protects against a
     retry-after-timeout double-booking an event that actually succeeded
     on iCloud's side on an earlier attempt.
"""
from datetime import datetime

import pytest

import config
from calendar_manager import CalendarError
import calendar_manager as cm


# ---------------------------------------------------------------------------
# stage_create_event - validation
# ---------------------------------------------------------------------------

def test_missing_title_raises(fake_calendars):
    with pytest.raises(CalendarError, match="title is required"):
        cm.stage_create_event(title="", start="2026-09-01 09:00")


def test_title_that_is_only_emoji_raises_specific_message(fake_calendars):
    """_strip_unsafe_text removes emoji before validation - a title of
    JUST emoji becomes an empty string, which needs its own error
    message (per calendar_manager.py's comment) rather than reusing the
    generic 'title is required' message, since the user did type
    something."""
    with pytest.raises(CalendarError, match="emoji/symbols only"):
        cm.stage_create_event(title="🎉🎉", start="2026-09-01 09:00")


def test_unparseable_start_raises(fake_calendars):
    with pytest.raises(CalendarError, match="Could not parse date/time"):
        cm.stage_create_event(title="Standup", start="not a date")


def test_end_before_start_raises(fake_calendars):
    with pytest.raises(CalendarError, match="end time must be after the start time"):
        cm.stage_create_event(title="Standup", start="2026-09-01 09:00", end="2026-09-01 08:00")


def test_end_defaults_to_one_hour_after_start(fake_calendars):
    cm.stage_create_event(title="Standup", start="2026-09-01 09:00")
    cm.confirm_pending()

    comp = fake_calendars[0].events[0].icalendar_component
    assert comp.get("dtstart").dt.replace(tzinfo=None) == datetime(2026, 9, 1, 9, 0)
    assert comp.get("dtend").dt.replace(tzinfo=None) == datetime(2026, 9, 1, 10, 0)


def test_unknown_calendar_name_raises_before_staging(fake_calendars):
    with pytest.raises(CalendarError, match="No calendar named 'Work' found"):
        cm.stage_create_event(title="Standup", start="2026-09-01 09:00", calendar_name="Work")
    assert cm.has_pending_change() is False  # validation failed - nothing staged


def test_emoji_stripped_from_location_and_description(fake_calendars):
    cm.stage_create_event(
        title="Standup", start="2026-09-01 09:00",
        location="Office 🏢", description="Weekly sync 🎉",
    )
    cm.confirm_pending()

    comp = fake_calendars[0].events[0].icalendar_component
    assert str(comp.get("location")) == "Office"
    assert str(comp.get("description")) == "Weekly sync"


# ---------------------------------------------------------------------------
# stage_create_event - duplicate-write guard
# ---------------------------------------------------------------------------

def test_confirm_skips_creating_duplicate_of_existing_event(fake_calendars):
    """Simulates the scenario the guard exists for: a previous create
    attempt already succeeded on iCloud's side (event with matching
    title+start already exists), but the client is retrying anyway.
    apply_fn should recognize it and NOT add a second copy."""
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0), datetime(2026, 9, 1, 9, 30))

    cm.stage_create_event(title="Standup", start="2026-09-01 09:00")
    result = cm.confirm_pending()

    assert "already exists" in result
    assert len(fake_calendars[0].events) == 1  # still just the one


def test_title_matching_is_case_insensitive_for_duplicate_guard(fake_calendars):
    fake_calendars[0].seed_event("standup", datetime(2026, 9, 1, 9, 0), datetime(2026, 9, 1, 9, 30))

    cm.stage_create_event(title="STANDUP", start="2026-09-01 09:00")
    cm.confirm_pending()

    assert len(fake_calendars[0].events) == 1


def test_different_title_at_same_time_is_not_a_duplicate(fake_calendars):
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0), datetime(2026, 9, 1, 9, 30))

    cm.stage_create_event(title="Dentist", start="2026-09-01 09:00")
    cm.confirm_pending()

    assert len(fake_calendars[0].events) == 2


# ---------------------------------------------------------------------------
# stage_create_events_batch - validation
# ---------------------------------------------------------------------------

def test_batch_requires_at_least_one_event(fake_calendars):
    with pytest.raises(CalendarError, match="At least one event is required"):
        cm.stage_create_events_batch(events=[])


def test_batch_rejects_non_list(fake_calendars):
    with pytest.raises(CalendarError, match="At least one event is required"):
        cm.stage_create_events_batch(events="not a list")


def test_batch_enforces_max_size(fake_calendars):
    too_many = [
        {"title": f"Event {i}", "start": f"2026-09-01 {9 + i:02d}:00"}
        for i in range(config.CALENDAR_BATCH_MAX_EVENTS + 1)
    ]
    with pytest.raises(CalendarError, match=f"at most {config.CALENDAR_BATCH_MAX_EVENTS} events"):
        cm.stage_create_events_batch(events=too_many)


def test_batch_reports_which_event_is_missing_a_title(fake_calendars):
    events = [
        {"title": "Standup", "start": "2026-09-01 09:00"},
        {"title": "", "start": "2026-09-01 10:00"},
    ]
    with pytest.raises(CalendarError, match=r"Event 2: a title is required"):
        cm.stage_create_events_batch(events=events)


def test_batch_reports_which_event_is_missing_a_start(fake_calendars):
    events = [{"title": "Standup", "start": ""}]
    with pytest.raises(CalendarError, match=r"Event 1 \('Standup'\): a start time is required"):
        cm.stage_create_events_batch(events=events)


def test_batch_end_defaults_to_configured_duration(fake_calendars):
    cm.stage_create_events_batch(events=[{"title": "Standup", "start": "2026-09-01 09:00"}])
    cm.confirm_pending()

    comp = fake_calendars[0].events[0].icalendar_component
    start = comp.get("dtstart").dt.replace(tzinfo=None)
    end = comp.get("dtend").dt.replace(tzinfo=None)
    assert (end - start).total_seconds() / 60 == config.CALENDAR_BATCH_DEFAULT_DURATION_MINUTES


def test_batch_rejects_end_before_start_for_one_event(fake_calendars):
    events = [{"title": "Standup", "start": "2026-09-01 09:00", "end": "2026-09-01 08:00"}]
    with pytest.raises(CalendarError, match="end time must be after the start time"):
        cm.stage_create_events_batch(events=events)


def test_batch_unknown_calendar_name_raises_before_any_validation(fake_calendars):
    with pytest.raises(CalendarError, match="No calendar named 'Work' found"):
        cm.stage_create_events_batch(
            events=[{"title": "Standup", "start": "2026-09-01 09:00"}], calendar_name="Work",
        )


# ---------------------------------------------------------------------------
# stage_create_events_batch - conflict detection
# ---------------------------------------------------------------------------

def test_batch_rejects_two_proposed_events_that_overlap_each_other(fake_calendars):
    events = [
        {"title": "Standup", "start": "2026-09-01 09:00", "end": "2026-09-01 10:00"},
        {"title": "1:1", "start": "2026-09-01 09:30", "end": "2026-09-01 10:30"},
    ]
    with pytest.raises(CalendarError, match="Proposed events overlap"):
        cm.stage_create_events_batch(events=events)


def test_batch_allows_back_to_back_proposed_events(fake_calendars):
    """End of one exactly equal to start of the next is adjacent, not
    overlapping - should be allowed."""
    events = [
        {"title": "Standup", "start": "2026-09-01 09:00", "end": "2026-09-01 09:30"},
        {"title": "1:1", "start": "2026-09-01 09:30", "end": "2026-09-01 10:00"},
    ]
    cm.stage_create_events_batch(events=events)
    cm.confirm_pending()
    assert len(fake_calendars[0].events) == 2


def test_batch_rejects_proposed_event_conflicting_with_existing_calendar_event(fake_calendars):
    fake_calendars[0].seed_event("Dentist", datetime(2026, 9, 1, 9, 0), datetime(2026, 9, 1, 9, 30))

    events = [{"title": "Standup", "start": "2026-09-01 09:15", "end": "2026-09-01 09:45"}]
    with pytest.raises(CalendarError, match="conflicts with an existing event 'Dentist'"):
        cm.stage_create_events_batch(events=events)

    assert len(fake_calendars[0].events) == 1  # nothing was staged/created


def test_batch_staged_description_lists_events_in_time_order_not_input_order(fake_calendars):
    events = [
        {"title": "Later", "start": "2026-09-01 14:00", "end": "2026-09-01 15:00"},
        {"title": "Earlier", "start": "2026-09-01 09:00", "end": "2026-09-01 10:00"},
    ]
    result = cm.stage_create_events_batch(events=events)

    assert result.index("Earlier") < result.index("Later")


# ---------------------------------------------------------------------------
# stage_create_events_batch - duplicate guard and partial-failure behavior
# ---------------------------------------------------------------------------

def test_batch_treats_an_exact_duplicate_of_an_existing_event_as_a_conflict(fake_calendars):
    """DISCOVERED ASYMMETRY (found by writing this test, not a pre-known
    fact): stage_create_event's duplicate guard runs at CONFIRM time,
    inside apply_fn (_find_matching_event), and quietly skips an exact
    retry-duplicate. stage_create_events_batch's conflict check instead
    runs at STAGE time via _get_busy_intervals, which has no concept of
    'this proposed event IS the thing that already exists' - it only
    sees a time-slot collision. So re-proposing an already-existing
    event inside a batch is rejected as a conflict and blocks staging
    the ENTIRE batch (including unrelated events in it), rather than
    being silently skipped the way the single-event path would handle
    the same retry scenario. Documenting current behavior here, not
    asserting it's correct - worth a decision on whether batch should
    match single-event behavior."""
    fake_calendars[0].seed_event("Standup", datetime(2026, 9, 1, 9, 0), datetime(2026, 9, 1, 9, 30))

    events = [
        {"title": "Standup", "start": "2026-09-01 09:00", "end": "2026-09-01 09:30"},
        {"title": "Dentist", "start": "2026-09-01 11:00", "end": "2026-09-01 11:30"},
    ]
    with pytest.raises(CalendarError, match="conflicts with an existing event 'Standup'"):
        cm.stage_create_events_batch(events=events)

    assert len(fake_calendars[0].events) == 1  # nothing staged or created - not even Dentist


def test_batch_partial_failure_reports_progress_and_is_safe_to_retry(fake_calendars, monkeypatch):
    """If one event in a batch fails to create, the error message should
    say how many succeeded before the failure, and confirming again
    (once the underlying problem is gone) should not re-create the ones
    that already went through.

    Note this needs the failure to persist across confirm_pending()'s
    OWN internal retry loop (CALENDAR_WRITE_RETRIES + 1 attempts) - a
    failure that clears up after just one retry gets silently absorbed
    by that loop and never reaches the caller at all, which is a nice
    self-healing property but means a single-call flaky mock (as in an
    earlier draft of this test) doesn't actually exercise the
    'reports progress' error path - it has to keep failing on 'Second'
    every attempt to prove that path."""
    events = [
        {"title": "First", "start": "2026-09-01 09:00", "end": "2026-09-01 09:30"},
        {"title": "Second", "start": "2026-09-01 10:00", "end": "2026-09-01 10:30"},
        {"title": "Third", "start": "2026-09-01 11:00", "end": "2026-09-01 11:30"},
    ]
    cm.stage_create_events_batch(events=events)
    monkeypatch.setattr(cm.time, "sleep", lambda seconds: None)  # skip real retry delays

    real_add_event = fake_calendars[0].add_event

    def blocks_second(*args, **kwargs):
        if kwargs.get("summary") == "Second":
            raise RuntimeError("simulated network drop")
        return real_add_event(*args, **kwargs)

    monkeypatch.setattr(fake_calendars[0], "add_event", blocks_second)

    with pytest.raises(CalendarError, match=r"1 of 3 event\(s\) in this batch were created"):
        cm.confirm_pending()

    assert len(fake_calendars[0].events) == 1  # only "First" made it, on the very first attempt
    assert cm.has_pending_change() is True  # still staged - safe to confirm again

    # Un-break add_event and confirm for real: "First" should be skipped
    # via the duplicate guard, "Second" and "Third" should be created.
    monkeypatch.setattr(fake_calendars[0], "add_event", real_add_event)
    result = cm.confirm_pending()

    assert "already exists - skipped" in result
    assert "Created 'Second'" in result
    assert "Created 'Third'" in result
    assert len(fake_calendars[0].events) == 3
