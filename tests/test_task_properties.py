"""
Tests for project_manager.py's first-class task scheduling field
validators: _parse_task_deadline, _parse_task_duration, _parse_task_effort,
_parse_task_when, and the compact _format_task_properties_inline renderer
used by build_context_text.

duration_manager.parse_duration_minutes is faked (fake_duration_module
fixture, matching the fake_duration fixture's controller but used here as a
direct module swap since these are plain functions, not project_manager
mutations that need state) - see fake_duration_manager.py for exactly which
input shapes the fake understands ("Xm", "Xh", "XhYm", bare numbers).
"""
import pytest

import project_manager as pm
from project_manager import ProjectManagerError
from tests.fakes.fake_duration_manager import FakeDurationModule


@pytest.fixture
def fake_duration_module(monkeypatch):
    fake_module = FakeDurationModule()
    monkeypatch.setattr(pm, "duration_manager", fake_module)
    return fake_module.controller


# ---------------------------------------------------------------------------
# _parse_task_deadline
# ---------------------------------------------------------------------------

def test_deadline_empty_and_none_are_not_set():
    assert pm._parse_task_deadline(None) == ""
    assert pm._parse_task_deadline("") == ""


def test_deadline_accepts_date_only():
    assert pm._parse_task_deadline("2026-09-10") == "2026-09-10"


def test_deadline_accepts_date_and_time():
    assert pm._parse_task_deadline("2026-09-10T14:30") == "2026-09-10T14:30"


def test_deadline_rejects_time_only():
    with pytest.raises(ProjectManagerError, match="YYYY-MM-DD"):
        pm._parse_task_deadline("14:30")


def test_deadline_rejects_malformed_string():
    with pytest.raises(ProjectManagerError, match="YYYY-MM-DD"):
        pm._parse_task_deadline("09/10/2026")


def test_deadline_rejects_invalid_calendar_date():
    with pytest.raises(ProjectManagerError, match="not a valid date"):
        pm._parse_task_deadline("2026-13-40")


def test_deadline_rejects_invalid_calendar_datetime():
    with pytest.raises(ProjectManagerError, match="not a valid date/time"):
        pm._parse_task_deadline("2026-09-10T25:99")


def test_deadline_error_message_uses_given_label():
    with pytest.raises(ProjectManagerError, match="^Due date"):
        pm._parse_task_deadline("garbage", "Due date")


# ---------------------------------------------------------------------------
# _parse_task_duration
# ---------------------------------------------------------------------------

def test_duration_empty_and_none_are_not_set(fake_duration_module):
    assert pm._parse_task_duration(None) is None
    assert pm._parse_task_duration("") is None


def test_duration_parses_recognized_shapes(fake_duration_module):
    assert pm._parse_task_duration("45m") == 45.0
    assert pm._parse_task_duration("1h") == 60.0
    assert pm._parse_task_duration("1h30m") == 90.0
    assert pm._parse_task_duration("90") == 90.0


def test_duration_rejects_unparseable_text(fake_duration_module):
    with pytest.raises(ProjectManagerError, match="Duration must be"):
        pm._parse_task_duration("banana")


def test_duration_error_message_uses_given_label(fake_duration_module):
    with pytest.raises(ProjectManagerError, match="^Estimate"):
        pm._parse_task_duration("banana", "Estimate")


# ---------------------------------------------------------------------------
# _parse_task_effort
# ---------------------------------------------------------------------------

def test_effort_empty_and_none_are_not_set():
    assert pm._parse_task_effort(None) == ""
    assert pm._parse_task_effort("") == ""


def test_effort_resolves_recognized_aliases():
    assert pm._parse_task_effort("low") == "low"
    assert pm._parse_task_effort("lo") == "low"
    assert pm._parse_task_effort("med") == "medium"
    assert pm._parse_task_effort("normal") == "medium"
    assert pm._parse_task_effort("hi") == "high"


def test_effort_is_case_insensitive():
    assert pm._parse_task_effort("HIGH") == "high"


def test_effort_rejects_unrecognized_value():
    with pytest.raises(ProjectManagerError, match="Effort must be"):
        pm._parse_task_effort("urgent")


# ---------------------------------------------------------------------------
# _parse_task_when
# ---------------------------------------------------------------------------

def test_when_empty_and_none_are_not_set():
    assert pm._parse_task_when(None) == ""
    assert pm._parse_task_when("") == ""


def test_when_accepts_time_word_only():
    assert pm._parse_task_when("afternoon") == "afternoon"


def test_when_accepts_time_and_modifier():
    assert pm._parse_task_when("afternoon weekend") == "afternoon weekend"


def test_when_rejects_unrecognized_time_word():
    with pytest.raises(ProjectManagerError, match="When must be"):
        pm._parse_task_when("someday")


def test_when_rejects_too_many_words():
    with pytest.raises(ProjectManagerError, match="When must be"):
        pm._parse_task_when("morning weekday extra")


def test_when_rejects_modifier_only():
    with pytest.raises(ProjectManagerError, match="When must be"):
        pm._parse_task_when("weekend")


# ---------------------------------------------------------------------------
# _format_duration_tag
# ---------------------------------------------------------------------------

def test_format_duration_tag_hours_and_minutes():
    assert pm._format_duration_tag(90) == "1h30m"


def test_format_duration_tag_whole_hours_only():
    assert pm._format_duration_tag(120) == "2h"


def test_format_duration_tag_minutes_only():
    assert pm._format_duration_tag(45) == "45m"


def test_format_duration_tag_rounds_to_nearest_minute():
    assert pm._format_duration_tag(45.6) == "46m"


# ---------------------------------------------------------------------------
# _format_task_properties_inline
# ---------------------------------------------------------------------------

def test_format_properties_inline_empty_task_is_empty_string():
    assert pm._format_task_properties_inline({}) == ""


def test_format_properties_inline_all_fields():
    task = {"duration_minutes": 45.0, "effort": "medium", "when": "afternoon weekend"}
    assert pm._format_task_properties_inline(task) == "~45m · medium effort · afternoon weekend"


def test_format_properties_inline_partial_fields():
    task = {"duration_minutes": None, "effort": "high", "when": ""}
    assert pm._format_task_properties_inline(task) == "high effort"


def test_format_properties_inline_missing_keys_are_treated_as_unset():
    assert pm._format_task_properties_inline({"title": "No properties here"}) == ""
