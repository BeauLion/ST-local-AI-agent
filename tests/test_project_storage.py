"""
Tests for project_manager.py's low-level building blocks: _load/_save
(and the focused_project_id auto-clear when that project no longer
exists), the small text-normalization helpers, and the project short
code derivation/allocation logic used by create_project.
"""
import json

import pytest

import project_manager as pm
from project_manager import ProjectManagerError


# ---------------------------------------------------------------------------
# _load / _save
# ---------------------------------------------------------------------------

def test_load_returns_default_state_when_no_file_exists(tmp_project_file):
    state = pm._load()
    assert state == {"projects": {}, "focused_project_id": None}


def test_load_clears_focused_project_id_if_that_project_no_longer_exists(tmp_project_file):
    tmp_project_file.write_text(json.dumps({
        "projects": {}, "focused_project_id": "project_ghost",
    }), encoding="utf-8")

    state = pm._load()

    assert state["focused_project_id"] is None


def test_load_keeps_focused_project_id_when_project_exists(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    state = pm._load()
    assert state["focused_project_id"] == project["id"]


def test_save_writes_readable_json(tmp_project_file):
    pm._save({"projects": {}, "focused_project_id": None})
    assert tmp_project_file.exists()
    assert json.loads(tmp_project_file.read_text(encoding="utf-8")) == {
        "projects": {}, "focused_project_id": None,
    }


# ---------------------------------------------------------------------------
# _normalize_text / _normalize_key
# ---------------------------------------------------------------------------

def test_normalize_text_collapses_internal_whitespace():
    assert pm._normalize_text("Thesis   Chapter   3") == "Thesis Chapter 3"


def test_normalize_text_strips_leading_and_trailing_whitespace():
    assert pm._normalize_text("  Thesis  ") == "Thesis"


def test_normalize_text_handles_none_and_non_string_input():
    assert pm._normalize_text(None) == ""
    assert pm._normalize_text(123) == "123"


def test_normalize_key_lowercases_after_normalizing():
    assert pm._normalize_key("  Thesis   Chapter  ") == "thesis chapter"


# ---------------------------------------------------------------------------
# _require_length
# ---------------------------------------------------------------------------

def test_require_length_raises_on_empty():
    with pytest.raises(ProjectManagerError, match="Task title is required"):
        pm._require_length("", "Task title", 100)


def test_require_length_raises_on_whitespace_only():
    with pytest.raises(ProjectManagerError, match="Task title is required"):
        pm._require_length("   ", "Task title", 100)


def test_require_length_raises_when_over_maximum():
    with pytest.raises(ProjectManagerError, match="must be 5 characters or fewer"):
        pm._require_length("Too Long", "Task title", 5)


def test_require_length_returns_normalized_value_when_valid():
    assert pm._require_length("  Reasonable Title  ", "Task title", 100) == "Reasonable Title"


# ---------------------------------------------------------------------------
# _derive_project_code
# ---------------------------------------------------------------------------

def test_derive_code_single_word_uses_first_three_letters():
    assert pm._derive_project_code("THESIS") == "THE"


def test_derive_code_multiple_words_uses_initials():
    assert pm._derive_project_code("MACHINE LEARNING RESEARCH") == "MLR"


def test_derive_code_caps_at_four_words_and_max_length():
    # 5 words -> only first 4 initials used, then truncated to MAX_PROJECT_CODE_LENGTH (6)
    assert pm._derive_project_code("ALPHA BETA GAMMA DELTA EPSILON") == "ABGD"


def test_derive_code_with_no_words_falls_back_to_prj():
    assert pm._derive_project_code("!!! ???") == "PRJ"


def test_derive_code_treats_digit_runs_as_their_own_word():
    assert pm._derive_project_code("PROJECT 2026") == "P2"


# ---------------------------------------------------------------------------
# _allocate_project_code - collision handling
# ---------------------------------------------------------------------------

def _state_with_codes(*codes):
    return {"projects": {f"p{i}": {"short_code": c} for i, c in enumerate(codes)}, "focused_project_id": None}


def test_allocate_code_uses_derived_code_when_available():
    state = _state_with_codes()
    assert pm._allocate_project_code(state, "Thesis") == "THE"


def test_allocate_code_appends_suffix_on_collision():
    state = _state_with_codes("THE")
    assert pm._allocate_project_code(state, "Thesis") == "THE2"


def test_allocate_code_increments_suffix_past_multiple_collisions():
    state = _state_with_codes("THE", "THE2", "THE3")
    assert pm._allocate_project_code(state, "Thesis") == "THE4"


def test_allocate_code_truncates_base_to_fit_suffix_within_max_length():
    """MAX_PROJECT_CODE_LENGTH is 6 - a base code plus a multi-digit
    suffix must still fit within that, truncating the base as needed."""
    state = _state_with_codes("ABGDEF")  # 6-char base already taken
    result = pm._allocate_project_code(state, "ALPHA BETA GAMMA DELTA EPSILON FOO")
    assert len(result) <= 6
    assert result != "ABGDEF"
