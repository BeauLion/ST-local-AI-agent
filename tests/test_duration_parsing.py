"""
Tests for duration_manager.py's parse_duration_minutes() - a pure
function (no state, no embeddings), used both by
project_manager.py's "dur:" note-tag parsing and by correct_entry()'s
free-text value parsing.
"""
import duration_manager as dm


# ---------------------------------------------------------------------------
# Phrase table (_PHRASE_MINUTES)
# ---------------------------------------------------------------------------

def test_phrase_table_hits():
    assert dm.parse_duration_minutes("half an hour") == 30.0
    assert dm.parse_duration_minutes("an hour") == 60.0
    assert dm.parse_duration_minutes("an hour and a half") == 90.0
    assert dm.parse_duration_minutes("a couple hours") == 120.0
    assert dm.parse_duration_minutes("a quarter hour") == 15.0


def test_phrase_table_is_case_and_whitespace_insensitive():
    assert dm.parse_duration_minutes("  HALF AN   HOUR  ") == 30.0


# ---------------------------------------------------------------------------
# Compound "1h30m" style (_COMPOUND_RE) - must be tried before the
# single-unit pattern, since it has two number+unit chunks
# ---------------------------------------------------------------------------

def test_compound_hours_and_minutes_compact_form():
    assert dm.parse_duration_minutes("1h30m") == 90.0


def test_compound_hours_and_minutes_spaced_form():
    assert dm.parse_duration_minutes("1hr 30min") == 90.0


def test_compound_hours_and_minutes_spelled_out():
    assert dm.parse_duration_minutes("1 hour 30 minutes") == 90.0


def test_compound_round_trips_with_the_format_helper_in_project_manager():
    """project_manager.py's _format_duration_tag writes exactly "1h30m"
    style back to notes for a mixed hours+minutes value - this is the
    exact string shape that round-trip has to survive re-parsing."""
    assert dm.parse_duration_minutes("2h15m") == 135.0


# ---------------------------------------------------------------------------
# Single unit: bare number, or number + hour/minute unit
# ---------------------------------------------------------------------------

def test_bare_number_with_no_unit_is_treated_as_minutes():
    assert dm.parse_duration_minutes("45") == 45.0


def test_number_with_minute_unit_variants():
    for unit in ("m", "min", "mins", "minute", "minutes"):
        assert dm.parse_duration_minutes(f"20{unit}") == 20.0
        assert dm.parse_duration_minutes(f"20 {unit}") == 20.0


def test_number_with_hour_unit_variants():
    for unit in ("h", "hr", "hrs", "hour", "hours"):
        assert dm.parse_duration_minutes(f"2{unit}") == 120.0
        assert dm.parse_duration_minutes(f"2 {unit}") == 120.0


def test_decimal_hours():
    assert dm.parse_duration_minutes("1.5h") == 90.0


# ---------------------------------------------------------------------------
# Filler word stripping ("~", "about", "approx")
# ---------------------------------------------------------------------------

def test_tilde_prefix_is_stripped():
    assert dm.parse_duration_minutes("~20min") == 20.0


def test_about_and_approx_are_stripped():
    assert dm.parse_duration_minutes("about 20 min") == 20.0
    assert dm.parse_duration_minutes("approx 20 min") == 20.0


# ---------------------------------------------------------------------------
# Invalid / unparseable input
# ---------------------------------------------------------------------------

def test_empty_string_returns_none():
    assert dm.parse_duration_minutes("") is None


def test_whitespace_only_returns_none():
    assert dm.parse_duration_minutes("   ") is None


def test_filler_words_alone_with_nothing_left_returns_none():
    assert dm.parse_duration_minutes("about") is None


def test_garbage_text_returns_none():
    assert dm.parse_duration_minutes("banana") is None


def test_unit_with_no_number_returns_none():
    assert dm.parse_duration_minutes("hours") is None