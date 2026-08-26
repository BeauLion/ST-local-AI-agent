"""
Tests for calendar_manager.py's datetime helpers. These are pure
functions (aside from resolve_when's dependence on "now") - no CalDAV,
no fake_calendars fixture needed, just calendar_manager imported
directly.

Dependency note found while writing these tests (not something these
tests check for, just worth knowing): resolve_when() does `import
dateparser` internally, but "dateparser" is NOT listed in
requirements.txt. It's presumably already installed in your real
environment (otherwise resolve_when would already be broken), but a
fresh `pip install -r requirements.txt` on a clean machine would leave
it missing until someone hits a date phrase that needs the fallback
path. Worth adding to requirements.txt.
"""
from datetime import datetime, timezone

import pytest
from freezegun import freeze_time

from calendar_manager import CalendarError
import calendar_manager as cm


# ---------------------------------------------------------------------------
# _parse_datetime
# ---------------------------------------------------------------------------

def test_parses_space_separated_format():
    assert cm._parse_datetime("2026-08-26 10:00") == datetime(2026, 8, 26, 10, 0)


def test_parses_t_separated_format():
    assert cm._parse_datetime("2026-08-26T10:00") == datetime(2026, 8, 26, 10, 0)


def test_parses_bare_date_as_midnight():
    assert cm._parse_datetime("2026-08-26") == datetime(2026, 8, 26, 0, 0)


def test_parses_iso_with_seconds():
    assert cm._parse_datetime("2026-08-26T10:00:05") == datetime(2026, 8, 26, 10, 0, 5)


def test_strips_surrounding_whitespace():
    assert cm._parse_datetime("  2026-08-26 10:00  ") == datetime(2026, 8, 26, 10, 0)


def test_iso_offset_matching_local_summer_offset_is_unchanged():
    """Europe/Amsterdam is UTC+2 in August (CEST). An incoming +02:00
    offset should map to the same local wall-clock time, not shift it."""
    result = cm._parse_datetime("2026-08-26T10:00:00+02:00")
    assert result == datetime(2026, 8, 26, 10, 0, 0)
    assert result.tzinfo is None  # converted then stripped to naive, per _parse_datetime's docstring


def test_utc_z_suffix_is_converted_to_local_time():
    """Z means UTC (offset 0). Amsterdam is UTC+2 in August, so 10:00 UTC
    should become 12:00 local - this is the one case in this file where
    the input and output clock times genuinely differ."""
    result = cm._parse_datetime("2026-08-26T10:00:00Z")
    assert result == datetime(2026, 8, 26, 12, 0, 0)


def test_rejects_unparseable_garbage():
    with pytest.raises(CalendarError, match="Could not parse date/time 'not a date'"):
        cm._parse_datetime("not a date")


def test_rejects_empty_string():
    with pytest.raises(CalendarError, match="Could not parse date/time"):
        cm._parse_datetime("")


def test_rejects_whitespace_only_string():
    with pytest.raises(CalendarError, match="Could not parse date/time"):
        cm._parse_datetime("   ")


# ---------------------------------------------------------------------------
# _localize
# ---------------------------------------------------------------------------

def test_localize_attaches_configured_timezone_to_naive_datetime():
    result = cm._localize(datetime(2026, 8, 26, 10, 0))
    assert result.tzinfo is not None
    assert result.tzinfo.key == "Europe/Amsterdam"  # CALENDAR_TIMEZONE
    assert result.replace(tzinfo=None) == datetime(2026, 8, 26, 10, 0)  # wall-clock time unchanged


def test_localize_leaves_already_aware_datetime_untouched():
    """An already-aware datetime is returned as-is, even if it's in a
    different timezone than CALENDAR_TIMEZONE - _localize does not
    re-localize/convert, only attaches a timezone when there isn't one."""
    aware = datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc)
    result = cm._localize(aware)
    assert result is aware
    assert result.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# resolve_when - keyword phrases (frozen on a known Wednesday)
# ---------------------------------------------------------------------------
# 2026-08-26 is a Wednesday. All expectations below were independently
# verified against the real resolve_when() with this same frozen clock
# before being written down here, not derived from the source by hand.

FROZEN_NOW = "2026-08-26 10:00:00"


def test_empty_phrase_raises():
    with pytest.raises(CalendarError, match="No date phrase given"):
        cm.resolve_when("")


def test_today_and_tonight():
    with freeze_time(FROZEN_NOW):
        expected = (datetime(2026, 8, 26), datetime(2026, 8, 27))
        assert cm.resolve_when("today") == expected
        assert cm.resolve_when("tonight") == expected


def test_tomorrow():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("tomorrow") == (datetime(2026, 8, 27), datetime(2026, 8, 28))


def test_yesterday():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("yesterday") == (datetime(2026, 8, 25), datetime(2026, 8, 26))


def test_day_after_tomorrow():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("day after tomorrow") == (datetime(2026, 8, 28), datetime(2026, 8, 29))


def test_this_week_spans_monday_to_next_monday():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("this week") == (datetime(2026, 8, 24), datetime(2026, 8, 31))


def test_next_week_is_the_following_seven_days():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("next week") == (datetime(2026, 8, 31), datetime(2026, 9, 7))


def test_this_weekend_is_the_upcoming_saturday_and_sunday():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("this weekend") == (datetime(2026, 8, 29), datetime(2026, 8, 31))
        assert cm.resolve_when("the weekend") == (datetime(2026, 8, 29), datetime(2026, 8, 31))


def test_this_weekend_when_today_is_already_saturday_starts_today():
    """Edge case in the (5 - today.weekday()) % 7 formula: when today IS
    Saturday, that expression is 0, so the weekend should start today,
    not jump a full week ahead."""
    with freeze_time("2026-08-29 10:00:00"):  # a Saturday
        assert cm.resolve_when("this weekend") == (datetime(2026, 8, 29), datetime(2026, 8, 31))


def test_this_month_spans_full_calendar_month():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("this month") == (datetime(2026, 8, 1), datetime(2026, 9, 1))


def test_this_month_handles_december_to_january_year_rollover():
    with freeze_time("2026-12-15 10:00:00"):
        assert cm.resolve_when("this month") == (datetime(2026, 12, 1), datetime(2027, 1, 1))


def test_bare_weekday_name_meaning_today_resolves_to_today():
    with freeze_time(FROZEN_NOW):  # today is Wednesday
        assert cm.resolve_when("wednesday") == (datetime(2026, 8, 26), datetime(2026, 8, 27))
        assert cm.resolve_when("this wednesday") == (datetime(2026, 8, 26), datetime(2026, 8, 27))


def test_next_weekday_when_today_already_matches_skips_to_following_week():
    """'next wednesday' when today already IS Wednesday should mean next
    week's Wednesday, not today - this is the delta==0 override in
    resolve_when's weekday-matching loop."""
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("next wednesday") == (datetime(2026, 9, 2), datetime(2026, 9, 3))


def test_bare_weekday_name_for_a_future_day_finds_the_upcoming_one():
    with freeze_time(FROZEN_NOW):  # today is Wednesday; Monday is 5 days out
        assert cm.resolve_when("monday") == (datetime(2026, 8, 31), datetime(2026, 9, 1))


def test_next_weekday_for_a_day_other_than_today_behaves_like_bare_name():
    """'next monday' when today is NOT Monday should give the same answer
    as bare 'monday' - the delta==0 override only applies when today
    already matches the requested day."""
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("next monday") == cm.resolve_when("monday")


def test_phrase_is_case_and_whitespace_insensitive():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("  TOMORROW  ") == cm.resolve_when("tomorrow")


# ---------------------------------------------------------------------------
# resolve_when - dateparser fallback, for phrases not in the keyword list
# ---------------------------------------------------------------------------

def test_dateparser_fallback_handles_relative_phrase():
    with freeze_time(FROZEN_NOW):
        assert cm.resolve_when("in 3 days") == (datetime(2026, 8, 29), datetime(2026, 8, 30))


def test_dateparser_fallback_rejects_truly_unparseable_phrase():
    with freeze_time(FROZEN_NOW):
        with pytest.raises(CalendarError, match="Could not understand the date phrase"):
            cm.resolve_when("asdkfjhasdkfjh")
