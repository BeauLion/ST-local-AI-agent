"""
Tests for project_manager.py's lightweight "key: value" note-tag syntax:
_parse_note_tags, _format_tags_block/_format_tags_inline, and
_compute_next_notes (the merge logic shared by update_task_notes and
the batch-update path).

_parse_note_tags only recognizes tag lines AT THE VERY TOP of a note,
and stops at the first line that isn't a recognized, validly-formatted
tag - everything from that line onward becomes prose, even if it looks
tag-like further down. That "stops at first miss" behavior is the thing
most worth pinning down precisely here.

duration_manager.parse_duration_minutes is faked (fake_duration fixture)
for these tests - see fake_duration_manager.py for why, and for exactly
which input shapes the fake understands ("Xm", "Xh", "XhYm", bare
numbers).
"""
import project_manager as pm


# ---------------------------------------------------------------------------
# _parse_note_tags - basic recognition
# ---------------------------------------------------------------------------

def test_empty_notes_returns_no_tags_and_empty_prose():
    assert pm._parse_note_tags("") == ({}, "")


def test_none_notes_returns_no_tags_and_empty_prose():
    assert pm._parse_note_tags(None) == ({}, "")


def test_plain_prose_with_no_tag_lines_is_returned_unchanged(fake_duration):
    tags, prose = pm._parse_note_tags("Just a plain note.\nSecond line.")
    assert tags == {}
    assert prose == "Just a plain note.\nSecond line."


def test_dur_tag_is_parsed_via_duration_manager(fake_duration):
    tags, prose = pm._parse_note_tags("dur: 30m\nRest of the note.")
    assert tags == {"dur": 30.0}
    assert prose == "Rest of the note."


def test_effort_tag_resolves_via_alias_table(fake_duration):
    tags, _ = pm._parse_note_tags("effort: hi\nSome prose.")
    assert tags == {"effort": "high"}


def test_when_tag_time_word_only(fake_duration):
    tags, _ = pm._parse_note_tags("when: afternoon\nProse.")
    assert tags == {"when": "afternoon"}


def test_when_tag_time_and_modifier(fake_duration):
    tags, _ = pm._parse_note_tags("when: afternoon weekend\nProse.")
    assert tags == {"when": "afternoon weekend"}


def test_multiple_tags_parsed_together_in_any_written_order(fake_duration):
    notes = "dur: 45m\neffort: low\nwhen: morning weekday\nActual notes here."
    tags, prose = pm._parse_note_tags(notes)
    assert tags == {"dur": 45.0, "effort": "low", "when": "morning weekday"}
    assert prose == "Actual notes here."


# ---------------------------------------------------------------------------
# _parse_note_tags - "stops at first miss" behavior
# ---------------------------------------------------------------------------

def test_notes_with_no_tag_lines_at_all_are_pure_prose(fake_duration):
    """A note written before this feature existed, or that just never
    used tag syntax - every line becomes prose, tags={}."""
    tags, prose = pm._parse_note_tags("key: value\nBut not a recognized key")
    assert tags == {}
    assert prose == "key: value\nBut not a recognized key"


def test_invalid_tag_value_stops_parsing_at_that_line(fake_duration):
    """"dur: not-a-duration" fails duration_manager.parse_duration_minutes
    (returns None) - the tag block stops THERE, and that line (plus
    everything after) becomes prose, even though a valid "effort:" line
    follows it."""
    notes = "dur: not-a-duration\neffort: high\nProse."
    tags, prose = pm._parse_note_tags(notes)
    assert tags == {}
    assert prose == "dur: not-a-duration\neffort: high\nProse."


def test_tag_recognized_only_at_the_very_top_not_mid_note(fake_duration):
    """A tag-shaped line appearing AFTER prose has already started must
    NOT be retroactively recognized - front-matter only."""
    notes = "dur: 20m\nSome notes.\neffort: high"
    tags, prose = pm._parse_note_tags(notes)
    assert tags == {"dur": 20.0}
    assert prose == "Some notes.\neffort: high"


def test_unrecognized_tag_key_stops_parsing(fake_duration):
    notes = "priority: urgent\ndur: 20m\nProse."
    tags, prose = pm._parse_note_tags(notes)
    assert tags == {}
    assert prose == "priority: urgent\ndur: 20m\nProse."


def test_malformed_when_phrase_stops_parsing():
    """"when: someday" isn't a recognized TASK_NOTE_WHEN_TIMES word -
    _parse_when_tag returns None, so the when line itself becomes prose."""
    tags, prose = pm._parse_note_tags("when: someday\nProse.")
    assert tags == {}
    assert prose == "when: someday\nProse."


def test_when_phrase_with_too_many_words_is_invalid():
    tags, prose = pm._parse_note_tags("when: morning weekday extra\nProse.")
    assert tags == {}
    assert prose == "when: morning weekday extra\nProse."


# ---------------------------------------------------------------------------
# _format_tags_block / _format_tags_inline
# ---------------------------------------------------------------------------

def test_format_tags_block_fixed_order_regardless_of_dict_order():
    tags = {"when": "morning", "dur": 45, "effort": "low"}
    assert pm._format_tags_block(tags) == "dur: 45m\neffort: low\nwhen: morning"


def test_format_tags_block_with_only_some_tags_present():
    assert pm._format_tags_block({"effort": "high"}) == "effort: high"


def test_format_tags_block_empty_dict_is_empty_string():
    assert pm._format_tags_block({}) == ""


def test_format_tags_inline_uses_middot_separator():
    tags = {"dur": 45, "effort": "medium", "when": "afternoon weekend"}
    assert pm._format_tags_inline(tags) == "~45m \u00b7 medium effort \u00b7 afternoon weekend"


def test_format_tags_inline_empty_dict_is_empty_string():
    assert pm._format_tags_inline({}) == ""


def test_format_duration_tag_hours_and_minutes():
    assert pm._format_duration_tag(90) == "1h30m"


def test_format_duration_tag_whole_hours_only():
    assert pm._format_duration_tag(120) == "2h"


def test_format_duration_tag_minutes_only():
    assert pm._format_duration_tag(45) == "45m"


def test_format_duration_tag_rounds_to_nearest_minute():
    assert pm._format_duration_tag(45.6) == "46m"


# ---------------------------------------------------------------------------
# _compute_next_notes - replace / clear / append modes
# ---------------------------------------------------------------------------

def test_clear_mode_always_returns_empty_string(fake_duration):
    assert pm._compute_next_notes("dur: 30m\nOld notes.", "clear", "ignored") == ""


def test_replace_mode_ignores_current_and_uses_text_verbatim(fake_duration):
    assert pm._compute_next_notes("Old notes.", "replace", "New notes.") == "New notes."


def test_replace_mode_with_none_text_becomes_empty_string(fake_duration):
    assert pm._compute_next_notes("Old notes.", "replace", None) == ""


def test_append_mode_with_no_tags_on_either_side_is_plain_string_join(fake_duration):
    result = pm._compute_next_notes("First line.", "append", "Second line.")
    assert result == "First line.\nSecond line."


def test_append_mode_merges_tags_new_values_win(fake_duration):
    current = "dur: 30m\neffort: low\nOld prose."
    addition = "dur: 60m\nMore prose."
    result = pm._compute_next_notes(current, "append", addition)
    # new "dur" (60m, re-formatted as "1h" by _format_duration_tag) wins
    # over old (30m); "effort" (low) survives untouched
    assert result == "dur: 1h\neffort: low\nOld prose.\nMore prose."


def test_append_mode_adds_a_new_tag_not_present_before(fake_duration):
    current = "dur: 30m\nOld prose."
    addition = "effort: high\nMore prose."
    result = pm._compute_next_notes(current, "append", addition)
    assert result == "dur: 30m\neffort: high\nOld prose.\nMore prose."


def test_append_mode_with_only_new_side_having_tags(fake_duration):
    current = "Just old prose, no tags."
    addition = "dur: 15m\nNew prose."
    result = pm._compute_next_notes(current, "append", addition)
    assert result == "dur: 15m\nJust old prose, no tags.\nNew prose."


def test_append_mode_when_new_text_has_no_prose_after_its_tags(fake_duration):
    current = "Old prose."
    addition = "dur: 15m"  # tag only, no prose line after it
    result = pm._compute_next_notes(current, "append", addition)
    assert result == "dur: 15m\nOld prose."
