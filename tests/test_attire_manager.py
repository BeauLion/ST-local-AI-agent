"""
Tests for attire_manager.py: character resolution, the update_attire/
get_attire_text/build_context_text_for_ids read/write surface, and
find_character_names_in_text (the best-effort scan used by main.py for
both context injection and force-including the attire tool group - see
that function's docstring for why it deliberately does whole-word,
case-insensitive matching only, no fuzzy/nickname matching).

Unlike calendar_manager, there's no staging/confirm safety model here and
no live network seam to fake - every write is immediate and local, so
these tests exercise attire_manager directly against the tmp_attire_file
fixture rather than needing anything like fake_calendars.
"""
import pytest
from freezegun import freeze_time

import attire_manager as am
from attire_manager import AttireManagerError


# ---------------------------------------------------------------------------
# resolve_character
# ---------------------------------------------------------------------------

def test_resolve_creates_a_new_character_when_none_matches(tmp_attire_file):
    state = am._load()
    record = am.resolve_character(state, "Aria")

    assert record["name"] == "Aria"
    assert record["slots"] == {"head": None, "top": None, "bottom": None, "feet": None, "accessories": []}
    assert record["id"] in state["characters"]


def test_resolve_matches_existing_character_case_and_whitespace_insensitively(tmp_attire_file):
    state = am._load()
    original = am.resolve_character(state, "Aria")

    found = am.resolve_character(state, "  aRIA  ")

    assert found["id"] == original["id"]
    assert len(state["characters"]) == 1  # no duplicate created


def test_resolve_a_genuinely_different_name_creates_a_separate_record(tmp_attire_file):
    """Documented limitation (see module docstring): exact normalized
    match only - a nickname or alternate spelling is NOT recognized as
    the same character and fragments into its own record."""
    state = am._load()
    am.resolve_character(state, "Aria Blackwood")
    am.resolve_character(state, "Aria")

    assert len(state["characters"]) == 2


def test_resolve_strips_and_normalizes_internal_whitespace_in_stored_name(tmp_attire_file):
    state = am._load()
    record = am.resolve_character(state, "  Aria   Blackwood  ")
    assert record["name"] == "Aria Blackwood"


def test_resolve_raises_on_empty_name(tmp_attire_file):
    state = am._load()
    with pytest.raises(AttireManagerError, match="character_name is required"):
        am.resolve_character(state, "")


def test_resolve_raises_on_whitespace_only_name(tmp_attire_file):
    state = am._load()
    with pytest.raises(AttireManagerError, match="character_name is required"):
        am.resolve_character(state, "   ")


def test_resolve_raises_on_name_over_max_length(tmp_attire_file, monkeypatch):
    monkeypatch.setattr(am, "MAX_CHARACTER_NAME_LENGTH", 10)
    state = am._load()
    with pytest.raises(AttireManagerError, match="100 characters or fewer|10 characters or fewer"):
        am.resolve_character(state, "A" * 11)


def test_resolve_with_create_if_missing_false_raises_when_not_found(tmp_attire_file):
    state = am._load()
    with pytest.raises(AttireManagerError, match="No attire record exists yet"):
        am.resolve_character(state, "Aria", create_if_missing=False)


def test_resolve_with_create_if_missing_false_still_finds_an_existing_match(tmp_attire_file):
    state = am._load()
    original = am.resolve_character(state, "Aria")

    found = am.resolve_character(state, "aria", create_if_missing=False)

    assert found["id"] == original["id"]


# ---------------------------------------------------------------------------
# find_character_names_in_text
# ---------------------------------------------------------------------------

def test_find_returns_empty_list_for_empty_text(tmp_attire_file):
    state = am._load()
    am.resolve_character(state, "Aria")
    assert am.find_character_names_in_text(state, "") == []


def test_find_returns_empty_list_when_no_known_character_mentioned(tmp_attire_file):
    state = am._load()
    am.resolve_character(state, "Aria")
    assert am.find_character_names_in_text(state, "A story about someone else entirely.") == []


def test_find_matches_a_known_name_case_insensitively(tmp_attire_file):
    state = am._load()
    record = am.resolve_character(state, "Aria")

    matches = am.find_character_names_in_text(state, "ARIA walks into the room.")

    assert matches == [record["id"]]


def test_find_respects_whole_word_boundaries(tmp_attire_file):
    """'Ana' must not match inside 'Anastasia' - a substring match without
    word boundaries would incorrectly flag this as a mention of Ana."""
    state = am._load()
    am.resolve_character(state, "Ana")

    matches = am.find_character_names_in_text(state, "Anastasia walks into the room.")

    assert matches == []


def test_find_returns_multiple_ids_when_multiple_known_characters_mentioned(tmp_attire_file):
    state = am._load()
    aria = am.resolve_character(state, "Aria")
    kai = am.resolve_character(state, "Kai")

    matches = am.find_character_names_in_text(state, "Aria and Kai stood at the door.")

    assert set(matches) == {aria["id"], kai["id"]}


def test_find_never_raises_on_characters_with_empty_names(tmp_attire_file):
    """Defensive case - a record with a falsy name (shouldn't normally
    happen given resolve_character's validation, but the function is
    documented as never raising) must be skipped, not crash the scan."""
    state = am._load()
    state["characters"]["fake-id"] = {"id": "fake-id", "name": "", "slots": {}, "updated_at": "x"}
    assert am.find_character_names_in_text(state, "Some text.") == []


# ---------------------------------------------------------------------------
# update_attire - validation
# ---------------------------------------------------------------------------

def test_update_requires_at_least_one_slot(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="At least one slot"):
        am.update_attire("Aria")


def test_update_requires_character_name(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="character_name is required"):
        am.update_attire("", top="jacket")


def test_update_rejects_item_text_over_max_length(tmp_attire_file, monkeypatch):
    monkeypatch.setattr(am, "MAX_ATTIRE_ITEM_LENGTH", 5)
    with pytest.raises(AttireManagerError, match="5 characters or fewer"):
        am.update_attire("Aria", top="way too long")


# ---------------------------------------------------------------------------
# update_attire - creating and updating
# ---------------------------------------------------------------------------

def test_update_creates_a_new_character_on_first_call(tmp_attire_file):
    record, changed = am.update_attire("Aria", top="leather jacket")

    assert record["name"] == "Aria"
    assert record["slots"]["top"] == "leather jacket"
    assert changed == ["top"]


def test_update_only_touches_the_slots_passed(tmp_attire_file):
    am.update_attire("Aria", top="leather jacket", feet="boots")

    record, changed = am.update_attire("Aria", top="denim jacket")

    assert record["slots"]["top"] == "denim jacket"
    assert record["slots"]["feet"] == "boots"  # untouched
    assert changed == ["top"]


def test_update_can_change_multiple_slots_in_one_call(tmp_attire_file):
    record, changed = am.update_attire("Aria", top="jacket", bottom="jeans", feet="boots")

    assert set(changed) == {"top", "bottom", "feet"}
    assert record["slots"]["top"] == "jacket"
    assert record["slots"]["bottom"] == "jeans"
    assert record["slots"]["feet"] == "boots"


def test_update_with_empty_string_clears_a_slot(tmp_attire_file):
    am.update_attire("Aria", feet="boots")

    record, changed = am.update_attire("Aria", feet="")

    assert record["slots"]["feet"] is None
    assert changed == ["feet"]


def test_update_with_omitted_slot_leaves_it_unchanged_not_cleared(tmp_attire_file):
    """The not-passed-vs-empty-string distinction, mirroring
    calendar_manager's location/description convention: None means 'no
    change requested', not 'clear this'."""
    am.update_attire("Aria", feet="boots")

    record, changed = am.update_attire("Aria", top="jacket")

    assert record["slots"]["feet"] == "boots"
    assert "feet" not in changed


def test_update_reports_no_change_when_value_already_matches(tmp_attire_file):
    am.update_attire("Aria", top="jacket")

    record, changed = am.update_attire("Aria", top="jacket")

    assert changed == []


def test_update_reports_no_change_when_clearing_an_already_empty_slot(tmp_attire_file):
    record, changed = am.update_attire("Aria", top="")
    assert changed == []
    assert record["slots"]["top"] is None


def test_update_strips_whitespace_from_item_text(tmp_attire_file):
    record, _ = am.update_attire("Aria", top="  leather jacket  ")
    assert record["slots"]["top"] == "leather jacket"


def test_update_sets_updated_at_on_change(tmp_attire_file):
    with freeze_time("2026-08-26 10:00:00"):
        record, changed = am.update_attire("Aria", top="jacket")
        assert changed
        assert record["updated_at"].startswith("2026-08-26T10:00:00")


def test_update_does_not_bump_updated_at_when_nothing_actually_changed(tmp_attire_file):
    with freeze_time("2026-08-26 10:00:00"):
        am.update_attire("Aria", top="jacket")

    with freeze_time("2026-08-27 12:00:00"):
        record, changed = am.update_attire("Aria", top="jacket")  # same value again

    assert changed == []
    assert record["updated_at"].startswith("2026-08-26T10:00:00")  # unchanged


def test_update_persists_across_separate_load_calls(tmp_attire_file):
    """Confirms _save() actually wrote to disk, not just mutated an
    in-memory dict that happened to still be around."""
    am.update_attire("Aria", top="jacket")

    reloaded = am._load()
    record = am.resolve_character(reloaded, "Aria", create_if_missing=False)

    assert record["slots"]["top"] == "jacket"


def test_update_is_case_insensitive_for_an_existing_character(tmp_attire_file):
    am.update_attire("Aria", top="jacket")

    record, changed = am.update_attire("ARIA", feet="boots")

    assert changed == ["feet"]
    state = am._load()
    assert len(state["characters"]) == 1  # same character, not a duplicate


# ---------------------------------------------------------------------------
# update_attire - accessories (the one list-valued slot)
# ---------------------------------------------------------------------------

def test_accessories_splits_on_commas(tmp_attire_file):
    record, changed = am.update_attire("Aria", accessories="necklace, gloves")

    assert record["slots"]["accessories"] == ["necklace", "gloves"]
    assert changed == ["accessories"]


def test_accessories_strips_whitespace_around_each_item(tmp_attire_file):
    record, _ = am.update_attire("Aria", accessories="  necklace ,  gloves  ")
    assert record["slots"]["accessories"] == ["necklace", "gloves"]


def test_accessories_drops_empty_items_between_commas(tmp_attire_file):
    record, _ = am.update_attire("Aria", accessories="necklace, , gloves")
    assert record["slots"]["accessories"] == ["necklace", "gloves"]


def test_accessories_empty_string_clears_the_list(tmp_attire_file):
    am.update_attire("Aria", accessories="necklace, gloves")

    record, changed = am.update_attire("Aria", accessories="")

    assert record["slots"]["accessories"] == []
    assert changed == ["accessories"]


def test_accessories_whitespace_only_string_clears_the_list(tmp_attire_file):
    am.update_attire("Aria", accessories="necklace")

    record, changed = am.update_attire("Aria", accessories="   ")

    assert record["slots"]["accessories"] == []
    assert changed == ["accessories"]


def test_accessories_no_change_when_list_is_identical(tmp_attire_file):
    am.update_attire("Aria", accessories="necklace, gloves")

    record, changed = am.update_attire("Aria", accessories="necklace, gloves")

    assert changed == []


def test_accessories_order_change_counts_as_a_change(tmp_attire_file):
    """Lists are compared positionally, not as sets - re-ordering the same
    items is treated as a real change. Documenting current behavior, not
    asserting it's the only reasonable choice."""
    am.update_attire("Aria", accessories="necklace, gloves")

    record, changed = am.update_attire("Aria", accessories="gloves, necklace")

    assert changed == ["accessories"]
    assert record["slots"]["accessories"] == ["gloves", "necklace"]


def test_single_accessory_with_no_comma_is_a_one_item_list(tmp_attire_file):
    record, _ = am.update_attire("Aria", accessories="necklace")
    assert record["slots"]["accessories"] == ["necklace"]


# ---------------------------------------------------------------------------
# get_attire_text
# ---------------------------------------------------------------------------

def test_get_attire_text_for_untracked_character_returns_the_error_message(tmp_attire_file):
    result = am.get_attire_text("Nobody")
    assert "No attire record exists yet" in result


def test_get_attire_text_includes_character_name_and_all_slots(tmp_attire_file):
    am.update_attire("Aria", top="jacket", feet="boots", accessories="necklace")

    result = am.get_attire_text("Aria")

    assert "Aria" in result
    assert "top: jacket" in result
    assert "feet: boots" in result
    assert "head: (none)" in result
    assert "accessories: necklace" in result


def test_get_attire_text_shows_none_for_empty_accessories(tmp_attire_file):
    am.update_attire("Aria", top="jacket")
    result = am.get_attire_text("Aria")
    assert "accessories: (none)" in result


def test_get_attire_text_lists_multiple_accessories_comma_separated(tmp_attire_file):
    am.update_attire("Aria", accessories="necklace, gloves")
    result = am.get_attire_text("Aria")
    assert "accessories: necklace, gloves" in result


# ---------------------------------------------------------------------------
# build_context_text_for_ids
# ---------------------------------------------------------------------------

def test_build_context_returns_empty_string_for_no_ids(tmp_attire_file):
    state = am._load()
    assert am.build_context_text_for_ids(state, []) == ""


def test_build_context_skips_an_id_that_no_longer_exists(tmp_attire_file):
    state = am._load()
    assert am.build_context_text_for_ids(state, ["not-a-real-id"]) == ""


def test_build_context_includes_the_persistent_state_header(tmp_attire_file):
    am.update_attire("Aria", top="jacket")
    state = am._load()
    aria_id = next(iter(state["characters"]))

    result = am.build_context_text_for_ids(state, [aria_id])

    assert "[PERSISTENT ATTIRE STATE]" in result
    assert "Aria" in result
    assert "top: jacket" in result


def test_build_context_includes_seeding_and_subtle_change_guidance(tmp_attire_file):
    """Locks down that the tightened instruction text (added specifically
    so the model doesn't skip subtle changes or wait for a full outfit
    change) is actually present in the injected block, not just in
    main.py's separate GROUP_INSTRUCTIONS."""
    am.update_attire("Aria", top="jacket")
    state = am._load()
    aria_id = next(iter(state["characters"]))

    result = am.build_context_text_for_ids(state, [aria_id])

    assert "attire_manager_update" in result
    assert "cheaper than a stale one" in result


def test_build_context_includes_multiple_characters_when_multiple_ids_given(tmp_attire_file):
    am.update_attire("Aria", top="jacket")
    am.update_attire("Kai", top="hoodie")
    state = am._load()
    ids = list(state["characters"].keys())

    result = am.build_context_text_for_ids(state, ids)

    assert "Aria" in result
    assert "Kai" in result
    assert "jacket" in result
    assert "hoodie" in result


def test_build_context_only_includes_ids_that_still_exist_even_when_mixed_with_missing(tmp_attire_file):
    am.update_attire("Aria", top="jacket")
    state = am._load()
    aria_id = next(iter(state["characters"]))

    result = am.build_context_text_for_ids(state, [aria_id, "not-a-real-id"])

    assert "Aria" in result
    assert result.count("[PERSISTENT ATTIRE STATE]") == 1  # header appears once, not per id
