"""
Tests for prompt_log_engine.py's log_prompt/log_console - the two
functions main.py's agent_loop calls every iteration to write JSONL
session log entries.

Both are deliberately best-effort: every exception is caught and
printed, never raised (per their own docstrings) - so these tests focus
on "does it write the right JSON shape" and "does it correctly skip
writing when appropriate", not on error-rejection, since there's no
input these functions are designed to reject in the first place.
"""
import json

import prompt_log_engine as pe
import console_log


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# log_prompt
# ---------------------------------------------------------------------------

def test_log_prompt_writes_one_jsonl_entry(tmp_log_dir, prompt_log_enabled):
    body = {"model": "test-model", "messages": [{"role": "system", "content": "You are helpful."}]}
    pe.log_prompt(body, iteration=1, section_labels=["system"])

    entries = _read_lines(pe.SESSION_LOG_PATH)
    assert len(entries) == 1
    assert entries[0]["type"] == "prompt"
    assert entries[0]["iteration"] == 1
    assert entries[0]["model"] == "test-model"
    assert entries[0]["chunks"][0] == {"role": "system", "section": "system", "content": "You are helpful."}


def test_log_prompt_does_nothing_when_disabled(tmp_log_dir, prompt_log_disabled):
    body = {"model": "test-model", "messages": [{"role": "system", "content": "Hi"}]}
    pe.log_prompt(body, iteration=1, section_labels=["system"])

    assert not pe.SESSION_LOG_PATH.exists()


def test_log_prompt_labels_messages_beyond_section_labels_as_conversation(tmp_log_dir, prompt_log_enabled):
    body = {"model": "m", "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]}
    pe.log_prompt(body, iteration=1, section_labels=["system"])  # only one label given

    entries = _read_lines(pe.SESSION_LOG_PATH)
    sections = [c["section"] for c in entries[0]["chunks"]]
    assert sections == ["system", "conversation", "conversation"]


def test_log_prompt_serializes_tool_calls_as_json_content_when_content_is_none(tmp_log_dir, prompt_log_enabled):
    body = {"model": "m", "messages": [
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "get_weather"}}]},
    ]}
    pe.log_prompt(body, iteration=1, section_labels=[])

    entries = _read_lines(pe.SESSION_LOG_PATH)
    assert "get_weather" in entries[0]["chunks"][0]["content"]


def test_log_prompt_appends_a_tools_schema_chunk_when_tools_are_present(tmp_log_dir, prompt_log_enabled):
    body = {"model": "m", "messages": [], "tools": [{"function": {"name": "search"}}]}
    pe.log_prompt(body, iteration=1, section_labels=[])

    entries = _read_lines(pe.SESSION_LOG_PATH)
    tool_chunks = [c for c in entries[0]["chunks"] if c["section"] == "tools_schema"]
    assert len(tool_chunks) == 1


def test_log_prompt_includes_tool_selection_debug_chunk_when_scores_given(tmp_log_dir, prompt_log_enabled):
    body = {"model": "m", "messages": [], "tools": [{"function": {"name": "search"}}]}
    pe.log_prompt(body, iteration=1, section_labels=[], tool_scores={"search": 0.9, "other_tool": 0.2})

    entries = _read_lines(pe.SESSION_LOG_PATH)
    debug_chunks = [c for c in entries[0]["chunks"] if c["section"] == "tool_selection_debug"]
    assert len(debug_chunks) == 1
    content = json.loads(debug_chunks[0]["content"])
    assert content["search"]["included"] is True
    assert content["other_tool"]["included"] is False


def test_log_prompt_includes_tier_in_debug_chunk_when_given(tmp_log_dir, prompt_log_enabled):
    body = {"model": "m", "messages": [], "tools": []}
    pe.log_prompt(body, iteration=1, section_labels=[], tool_scores={"search": 0.9}, tool_tier="direct")

    entries = _read_lines(pe.SESSION_LOG_PATH)
    debug_chunk = next(c for c in entries[0]["chunks"] if c["section"] == "tool_selection_debug")
    assert debug_chunk["tier"] == "direct"


def test_log_prompt_appends_to_the_same_file_across_multiple_calls(tmp_log_dir, prompt_log_enabled):
    body = {"model": "m", "messages": []}
    pe.log_prompt(body, iteration=1, section_labels=[])
    pe.log_prompt(body, iteration=2, section_labels=[])

    entries = _read_lines(pe.SESSION_LOG_PATH)
    assert [e["iteration"] for e in entries] == [1, 2]


def test_log_prompt_never_raises_even_on_a_malformed_body(tmp_log_dir, prompt_log_enabled, capsys):
    """Best-effort per the docstring - a body missing "messages" entirely
    should be caught and printed, not propagate up into agent_loop."""
    pe.log_prompt({"model": "m"}, iteration=1, section_labels=[])  # no "messages" key at all
    captured = capsys.readouterr()
    assert "Prompt log write failed" in captured.out


# ---------------------------------------------------------------------------
# log_console
# ---------------------------------------------------------------------------

def test_log_console_writes_buffered_lines(tmp_log_dir, prompt_log_enabled):
    console_log.alog("line one")
    console_log.alog("line two")

    pe.log_console(iteration=1)

    entries = _read_lines(pe.SESSION_LOG_PATH)
    assert entries[0]["type"] == "console"
    assert entries[0]["lines"] == ["line one", "line two"]


def test_log_console_drains_the_buffer_even_when_disabled(tmp_log_dir, prompt_log_disabled):
    """Per the docstring: even when disabled, flush_console() must still
    run so buffered lines can't leak into a LATER (possibly enabled)
    request."""
    console_log.alog("should be drained")
    pe.log_console(iteration=1)

    assert console_log._lines == []
    assert not pe.SESSION_LOG_PATH.exists()  # nothing written though - disabled


def test_log_console_writes_nothing_for_a_quiet_iteration(tmp_log_dir, prompt_log_enabled):
    """No buffered lines, no response, no thinking - writing nothing here
    keeps quiet iterations from cluttering the log, per the docstring."""
    pe.log_console(iteration=1)
    assert not pe.SESSION_LOG_PATH.exists()


def test_log_console_writes_an_entry_for_a_response_with_no_console_lines(tmp_log_dir, prompt_log_enabled):
    pe.log_console(iteration=1, response={"kind": "text", "text": "Hello!"})

    entries = _read_lines(pe.SESSION_LOG_PATH)
    assert entries[0]["lines"] == []
    assert entries[0]["response"] == {"kind": "text", "text": "Hello!"}


def test_log_console_includes_thinking_when_given(tmp_log_dir, prompt_log_enabled):
    pe.log_console(iteration=1, thinking="Reasoning trace...")

    entries = _read_lines(pe.SESSION_LOG_PATH)
    assert entries[0]["thinking"] == "Reasoning trace..."


def test_log_console_omits_response_and_thinking_keys_when_not_given(tmp_log_dir, prompt_log_enabled):
    console_log.alog("just a line")
    pe.log_console(iteration=1)

    entries = _read_lines(pe.SESSION_LOG_PATH)
    assert "response" not in entries[0]
    assert "thinking" not in entries[0]


def test_log_console_appears_immediately_after_its_matching_prompt_entry(tmp_log_dir, prompt_log_enabled):
    pe.log_prompt({"model": "m", "messages": []}, iteration=1, section_labels=[])
    console_log.alog("some output")
    pe.log_console(iteration=1)

    entries = _read_lines(pe.SESSION_LOG_PATH)
    assert entries[0]["type"] == "prompt"
    assert entries[1]["type"] == "console"
