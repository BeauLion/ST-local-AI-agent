"""
Tests for calendar_manager.py's staged-write lifecycle: _stage(),
has_pending_change(), confirm_pending(), cancel_pending(), and the TTL
expiry inside _peek_pending().

These deliberately test the *lifecycle* machinery (stage -> confirm,
stage -> cancel, stage -> expire, confirm -> retry) using
stage_create_event as a convenient real staging call to drive it - the
correctness of event creation itself belongs in
test_calendar_stage_create.py, not here.
"""
from datetime import timedelta

import pytest
from freezegun import freeze_time

import calendar_manager as cm
import config
from calendar_manager import CalendarError


def test_no_pending_change_initially(fake_calendars):
    assert cm.has_pending_change() is False


def test_stage_sets_pending_and_does_not_write_yet(fake_calendars):
    result = cm.stage_create_event(title="Standup", start="2026-09-01 09:00")

    assert cm.has_pending_change() is True
    assert "Staged (NOT yet applied" in result
    assert "confirm" in result.lower()
    assert len(fake_calendars[0].events) == 0  # nothing written to the calendar yet


def test_confirm_applies_exactly_once_and_clears(fake_calendars):
    cm.stage_create_event(title="Standup", start="2026-09-01 09:00")

    result = cm.confirm_pending()

    assert "Created 'Standup'" in result
    assert len(fake_calendars[0].events) == 1
    assert cm.has_pending_change() is False


def test_confirm_with_nothing_staged_does_not_reapply(fake_calendars):
    cm.stage_create_event(title="Standup", start="2026-09-01 09:00")
    cm.confirm_pending()

    # A second confirm with nothing staged must not create it again.
    second = cm.confirm_pending()

    assert "Nothing is staged" in second
    assert len(fake_calendars[0].events) == 1


def test_cancel_clears_without_writing(fake_calendars):
    cm.stage_create_event(title="Standup", start="2026-09-01 09:00")

    result = cm.cancel_pending()

    assert "Cancelled" in result
    assert cm.has_pending_change() is False
    assert len(fake_calendars[0].events) == 0


def test_cancel_with_nothing_staged(fake_calendars):
    result = cm.cancel_pending()
    assert result == "Nothing is staged to cancel."


def test_staging_a_new_change_replaces_the_old_one(fake_calendars):
    """Module docstring: 'Only one change can be staged at a time -
    staging a new one silently replaces whatever was staged before.'"""
    cm.stage_create_event(title="First", start="2026-09-01 09:00")
    cm.stage_create_event(title="Second", start="2026-09-01 10:00")

    result = cm.confirm_pending()

    assert "Created 'Second'" in result
    assert "First" not in result
    assert len(fake_calendars[0].events) == 1
    assert str(fake_calendars[0].events[0].icalendar_component.get("summary")) == "Second"


def test_pending_change_expires_after_ttl(fake_calendars):
    with freeze_time("2026-09-01 08:00:00") as frozen:
        cm.stage_create_event(title="Standup", start="2026-09-01 09:00")
        assert cm.has_pending_change() is True

        frozen.tick(timedelta(minutes=config.CALENDAR_PENDING_CHANGE_TTL_MINUTES + 1))

        assert cm.has_pending_change() is False
        result = cm.confirm_pending()
        assert "Nothing is staged" in result
        assert len(fake_calendars[0].events) == 0  # expired change was never applied


def test_pending_change_not_yet_expired_within_ttl(fake_calendars):
    with freeze_time("2026-09-01 08:00:00") as frozen:
        cm.stage_create_event(title="Standup", start="2026-09-01 09:00")

        frozen.tick(timedelta(minutes=config.CALENDAR_PENDING_CHANGE_TTL_MINUTES - 1))

        assert cm.has_pending_change() is True


def test_confirm_retries_then_raises_but_keeps_pending_on_persistent_failure(fake_calendars, monkeypatch):
    """If apply_fn keeps failing, confirm_pending should retry
    CALENDAR_WRITE_RETRIES times and then raise - and, per the comment
    in confirm_pending's source, deliberately NOT clear the pending
    change on total failure. That lets the user just say 'confirm' again
    later instead of the model re-describing/re-staging from scratch."""
    cm.stage_create_event(title="Standup", start="2026-09-01 09:00")
    monkeypatch.setattr(cm.time, "sleep", lambda seconds: None)  # skip real retry delays

    call_count = {"n": 0}

    def always_fails():
        call_count["n"] += 1
        raise CalendarError("simulated iCloud failure")

    cm._pending_change["apply"] = always_fails

    with pytest.raises(CalendarError, match="simulated iCloud failure"):
        cm.confirm_pending()

    assert call_count["n"] == config.CALENDAR_WRITE_RETRIES + 1
    assert cm.has_pending_change() is True  # still staged, not cleared


def test_confirm_succeeds_after_a_transient_failure(fake_calendars, monkeypatch):
    """Confirms the loop actually retries mid-flight, not just that it
    eventually gives up - first attempt fails, second succeeds."""
    cm.stage_create_event(title="Standup", start="2026-09-01 09:00")
    monkeypatch.setattr(cm.time, "sleep", lambda seconds: None)

    real_apply = cm._pending_change["apply"]
    attempts = {"n": 0}

    def flaky_apply():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise CalendarError("transient failure")
        return real_apply()

    cm._pending_change["apply"] = flaky_apply

    result = cm.confirm_pending()

    assert "Created 'Standup'" in result
    assert attempts["n"] == 2
    assert cm.has_pending_change() is False
