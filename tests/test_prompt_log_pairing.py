"""
Tests for prompt_log_engine.py's _entry_type/_build_groups/_group_chunks/
_group_ref - the Python port of prompt_log_viewer.js's buildGroups()/
groupChunks()/groupRef(). This is the highest-value part of the module to
test: it's hand-duplicated logic that has to agree with a separate JS
implementation the module's own docstring warns about drifting out of
sync, and /notes/search's ability to resolve a note back to the right
iteration/chunk depends entirely on it staying correct.
"""
import prompt_log_engine as pe


def _prompt_entry(iteration=1, chunks=None):
    return {"type": "prompt", "iteration": iteration, "chunks": chunks or []}


def _console_entry(iteration=1, lines=None, response=None, thinking=None):
    entry = {"type": "console", "iteration": iteration, "lines": lines or []}
    if response:
        entry["response"] = response
    if thinking:
        entry["thinking"] = thinking
    return entry


# ---------------------------------------------------------------------------
# _entry_type
# ---------------------------------------------------------------------------

def test_entry_type_reads_the_explicit_type_field():
    assert pe._entry_type({"type": "console"}) == "console"
    assert pe._entry_type({"type": "prompt"}) == "prompt"


def test_entry_type_falls_back_to_chunks_presence_for_untyped_entries():
    """Older log entries (written before the "type" field existed) are
    inferred from whether "chunks" is present - a prompt entry always has
    a "chunks" key, a console entry never does."""
    assert pe._entry_type({"chunks": []}) == "prompt"
    assert pe._entry_type({"lines": []}) == "console"


# ---------------------------------------------------------------------------
# _build_groups - normal pairing
# ---------------------------------------------------------------------------

def test_build_groups_pairs_a_prompt_with_its_following_console_entry():
    entries = [_prompt_entry(1), _console_entry(1)]
    groups = pe._build_groups(entries)

    assert len(groups) == 1
    assert groups[0]["prompt"] is entries[0]
    assert groups[0]["console"] is entries[1]


def test_build_groups_handles_multiple_iterations_in_sequence():
    entries = [_prompt_entry(1), _console_entry(1), _prompt_entry(2), _console_entry(2)]
    groups = pe._build_groups(entries)

    assert len(groups) == 2
    assert groups[0]["prompt"]["iteration"] == 1
    assert groups[1]["prompt"]["iteration"] == 2


# ---------------------------------------------------------------------------
# _build_groups - the edge cases the module's docstring specifically calls out
# ---------------------------------------------------------------------------

def test_build_groups_handles_a_prompt_with_no_following_console_entry():
    """A prompt entry as the very last line (server crashed/restarted
    before its console flush was written) must still form its own group,
    with console=None - not get silently dropped or merged incorrectly."""
    entries = [_prompt_entry(1)]
    groups = pe._build_groups(entries)

    assert len(groups) == 1
    assert groups[0]["prompt"] is entries[0]
    assert groups[0]["console"] is None


def test_build_groups_handles_an_orphaned_console_entry_with_no_preceding_prompt():
    """An orphaned console entry (e.g. a manual flush with no matching
    prompt log, per the module docstring's "orphaned-console edge case")
    must become its own group with prompt=None, consuming only itself -
    not get incorrectly attached to something else or advance the index
    by 2."""
    entries = [_console_entry(1), _prompt_entry(2), _console_entry(2)]
    groups = pe._build_groups(entries)

    assert len(groups) == 2
    assert groups[0]["prompt"] is None
    assert groups[0]["console"] is entries[0]
    assert groups[1]["prompt"] is entries[1]
    assert groups[1]["console"] is entries[2]


def test_build_groups_two_consecutive_prompts_with_no_console_between_them():
    """Two prompt entries back to back (no console line in between at
    all) must form two SEPARATE single-entry groups, not accidentally
    pair the second prompt in as if it were the first's console."""
    entries = [_prompt_entry(1), _prompt_entry(2)]
    groups = pe._build_groups(entries)

    assert len(groups) == 2
    assert groups[0]["prompt"] is entries[0]
    assert groups[0]["console"] is None
    assert groups[1]["prompt"] is entries[1]
    assert groups[1]["console"] is None


def test_build_groups_empty_entries_list_returns_empty_groups():
    assert pe._build_groups([]) == []


def test_build_groups_preserves_original_order_and_position_indices():
    """/notes/search resolves a note's saved key back to groups[index] -
    group POSITION must be stable and match insertion order exactly."""
    entries = [
        _console_entry(1),          # group 0: orphaned console
        _prompt_entry(2), _console_entry(2),  # group 1
        _prompt_entry(3),           # group 2: prompt with no console
    ]
    groups = pe._build_groups(entries)
    assert len(groups) == 3
    assert groups[0]["console"] is entries[0]
    assert groups[1]["prompt"] is entries[1] and groups[1]["console"] is entries[2]
    assert groups[2]["prompt"] is entries[3]


# ---------------------------------------------------------------------------
# _group_chunks
# ---------------------------------------------------------------------------

def test_group_chunks_includes_the_prompt_chunks_first():
    prompt_chunks = [{"role": "system", "section": "system", "content": "You are..."}]
    group = {"prompt": _prompt_entry(1, chunks=prompt_chunks), "console": None}

    result = pe._group_chunks(group)
    assert result == prompt_chunks


def test_group_chunks_appends_console_output_as_its_own_chunk():
    group = {"prompt": _prompt_entry(1), "console": _console_entry(1, lines=["print A", "print B"])}

    result = pe._group_chunks(group)
    assert result[-1] == {"role": "console", "section": "console_output", "content": "print A\nprint B"}


def test_group_chunks_omits_console_output_chunk_when_there_are_no_lines():
    group = {"prompt": _prompt_entry(1), "console": _console_entry(1, lines=[])}
    result = pe._group_chunks(group)
    assert not any(c["section"] == "console_output" for c in result)


def test_group_chunks_appends_thinking_chunk_when_present():
    group = {"prompt": _prompt_entry(1), "console": _console_entry(1, thinking="Let me consider...")}
    result = pe._group_chunks(group)
    assert {"role": "thinking", "section": "reasoning", "content": "Let me consider..."} in result


def test_group_chunks_appends_response_chunk_with_its_kind_as_section():
    group = {"prompt": _prompt_entry(1), "console": _console_entry(1, response={"kind": "tool_call", "text": "..."})}
    result = pe._group_chunks(group)
    assert {"role": "response", "section": "tool_call", "content": "..."} in result


def test_group_chunks_response_without_kind_falls_back_to_response_section():
    group = {"prompt": _prompt_entry(1), "console": _console_entry(1, response={"text": "Hello"})}
    result = pe._group_chunks(group)
    assert {"role": "response", "section": "response", "content": "Hello"} in result


def test_group_chunks_with_no_prompt_and_no_console_returns_empty_list():
    group = {"prompt": None, "console": None}
    assert pe._group_chunks(group) == []


def test_group_chunks_with_only_an_orphaned_console_entry():
    group = {"prompt": None, "console": _console_entry(1, lines=["orphaned output"])}
    result = pe._group_chunks(group)
    assert result == [{"role": "console", "section": "console_output", "content": "orphaned output"}]


def test_group_chunks_preserves_chunk_order_prompt_then_console_then_thinking_then_response():
    group = {
        "prompt": _prompt_entry(1, chunks=[{"role": "system", "section": "system", "content": "sys"}]),
        "console": _console_entry(1, lines=["out"], thinking="thought", response={"kind": "text", "text": "resp"}),
    }
    result = pe._group_chunks(group)
    sections = [c["section"] for c in result]
    assert sections == ["system", "console_output", "reasoning", "text"]


# ---------------------------------------------------------------------------
# _group_ref
# ---------------------------------------------------------------------------

def test_group_ref_prefers_the_prompt_entry():
    prompt = _prompt_entry(1)
    console = _console_entry(1)
    assert pe._group_ref({"prompt": prompt, "console": console}) is prompt


def test_group_ref_falls_back_to_console_when_no_prompt():
    console = _console_entry(1)
    assert pe._group_ref({"prompt": None, "console": console}) is console
