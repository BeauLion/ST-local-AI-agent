"""
Tests for console_log.py - a tiny shared buffer so per-request console
prints can be captured alongside the terminal output they already
produce, then attached to the matching prompt-log entry.

No Pydantic migration for this module: alog()/flush() have no input
validation at all (alog accepts any string, flush takes no arguments),
so there's no field-shape logic to move into a model - this module is
tests-only.
"""
import console_log


def test_alog_prints_exactly_as_a_normal_print_would(capsys):
    console_log.alog("Hello world")
    captured = capsys.readouterr()
    assert captured.out == "Hello world\n"


def test_alog_appends_the_message_to_the_buffer():
    console_log.alog("First message")
    assert console_log._lines == ["First message"]


def test_multiple_alog_calls_accumulate_in_order():
    console_log.alog("First")
    console_log.alog("Second")
    console_log.alog("Third")
    assert console_log._lines == ["First", "Second", "Third"]


def test_flush_returns_everything_buffered_since_the_last_flush():
    console_log.alog("One")
    console_log.alog("Two")
    result = console_log.flush()
    assert result == ["One", "Two"]


def test_flush_clears_the_buffer():
    console_log.alog("One")
    console_log.flush()
    assert console_log._lines == []


def test_flush_with_nothing_buffered_returns_empty_list():
    assert console_log.flush() == []


def test_flush_after_flush_is_empty():
    console_log.alog("One")
    console_log.flush()
    assert console_log.flush() == []


def test_lines_logged_after_a_flush_start_a_fresh_buffer():
    console_log.alog("Before flush")
    console_log.flush()
    console_log.alog("After flush")
    assert console_log.flush() == ["After flush"]


def test_flush_returns_a_new_list_not_a_live_reference_to_the_internal_buffer():
    """flush() does `lines, _lines = _lines, []` - the caller's returned
    list must be independent of the module's internal buffer from that
    point on, so mutating the returned list (or logging more afterward)
    can't retroactively change what was already flushed."""
    console_log.alog("One")
    result = console_log.flush()
    console_log.alog("Two")
    assert result == ["One"]  # unaffected by the alog() call after flush
