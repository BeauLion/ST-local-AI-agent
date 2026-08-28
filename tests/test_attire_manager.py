"""
Tests for attire_manager.py: character resolution, the v2 add_item/
remove_item/replace_slot/get_attire_text/build_context_text_for_ids
read/write surface, v1->v2 on-disk migration, and
find_character_names_in_text (the best-effort scan used by main.py for
context injection - see that function's docstring for why it
deliberately does whole-word, case-insensitive matching only, no
fuzzy/nickname matching).

v2 background (see brainstorm-layered-clothing.md): every slot is now a
list, and the old single "update_attire(full value)" call is replaced by
three verbs - add_item (append, cannot touch anything else in the slot),
remove_item (best-effort single-item removal, fail-open on ambiguity),
and replace_slot (the one deliberate full-wipe operation). Several tests
below exist specifically to lock down the failure mode this schema
change was built to close: layering one item onto another (or removing
one of several) must never be able to silently erase a sibling item.

Unlike calendar_manager, there's no staging/confirm safety model here and
no live network seam to fake - every write is immediate and local, so
these tests exercise attire_manager directly against the tmp_attire_file
fixture rather than needing anything like fake_calendars.
"""
import json

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
    assert record["slots"] == {"head": [], "top": [], "bottom": [], "feet": [], "accessories": []}
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
# v1 -> v2 migration (_load / _migrate_record)
# ---------------------------------------------------------------------------

def test_load_migrates_v1_string_slots_into_lists(tmp_attire_file):
    """Simulates a real v1 attire.json on disk (head/top/bottom/feet as a
    string-or-None, only accessories as a list) and confirms _load()
    upgrades every record transparently - no separate migration script."""
    v1_state = {
        "characters": {
            "abc123": {
                "id": "abc123",
                "name": "Aria",
                "slots": {
                    "head": None, "top": "jacket", "bottom": None,
                    "feet": "boots", "accessories": ["necklace"],
                },
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
    }
    tmp_attire_file.write_text(json.dumps(v1_state), encoding="utf-8")

    state = am._load()
    record = state["characters"]["abc123"]

    assert record["slots"] == {
        "head": [], "top": ["jacket"], "bottom": [], "feet": ["boots"], "accessories": ["necklace"],
    }


def test_load_migration_is_idempotent_on_already_list_slots(tmp_attire_file):
    """A record already in v2 format passes through _migrate_record
    untouched - guards against a future _load() call double-wrapping a
    list into a nested one-element list."""
    v2_state = {
        "characters": {
            "abc123": {
                "id": "abc123",
                "name": "Aria",
                "slots": {
                    "head": [], "top": ["jacket"], "bottom": [],
                    "feet": ["boots", "socks"], "accessories": [],
                },
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        }
    }
    tmp_attire_file.write_text(json.dumps(v2_state), encoding="utf-8")

    state = am._load()
    record = state["characters"]["abc123"]

    assert record["slots"]["feet"] == ["boots", "socks"]


def test_migrate_record_handles_missing_slots_key_defensively(tmp_attire_file):
    """Defensive case - a record somehow missing the 'slots' key entirely
    (shouldn't happen via this module's own writes) still gets a full set
    of empty lists rather than crashing _load()."""
    state = {"characters": {"abc": {"id": "abc", "name": "Aria", "updated_at": "x"}}}
    tmp_attire_file.write_text(json.dumps(state), encoding="utf-8")

    loaded = am._load()

    assert loaded["characters"]["abc"]["slots"] == {
        "head": [], "top": [], "bottom": [], "feet": [], "accessories": [],
    }


# ---------------------------------------------------------------------------
# add_item / remove_item / replace_slot - validation
# ---------------------------------------------------------------------------

def test_add_item_requires_character_name(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="character_name is required"):
        am.add_item("", "top", "jacket")


def test_add_item_rejects_unknown_slot(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="Unknown slot"):
        am.add_item("Aria", "hands", "gloves")


def test_add_item_rejects_empty_item(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="item is required"):
        am.add_item("Aria", "top", "")


def test_add_item_rejects_item_text_over_max_length(tmp_attire_file, monkeypatch):
    monkeypatch.setattr(am, "MAX_ATTIRE_ITEM_LENGTH", 5)
    with pytest.raises(AttireManagerError, match="5 characters or fewer"):
        am.add_item("Aria", "top", "way too long")


def test_remove_item_requires_character_name(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="character_name is required"):
        am.remove_item("", "top", "jacket")


def test_remove_item_rejects_unknown_slot(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")
    with pytest.raises(AttireManagerError, match="Unknown slot"):
        am.remove_item("Aria", "hands", "jacket")


def test_remove_item_rejects_empty_item_hint(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")
    with pytest.raises(AttireManagerError, match="item_hint is required"):
        am.remove_item("Aria", "top", "")


def test_remove_item_raises_for_untracked_character(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="No attire record exists yet"):
        am.remove_item("Nobody", "top", "jacket")


def test_replace_slot_requires_character_name(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="character_name is required"):
        am.replace_slot("", "top", "jacket")


def test_replace_slot_rejects_unknown_slot(tmp_attire_file):
    with pytest.raises(AttireManagerError, match="Unknown slot"):
        am.replace_slot("Aria", "hands", "gloves")


def test_replace_slot_rejects_item_text_over_max_length(tmp_attire_file, monkeypatch):
    monkeypatch.setattr(am, "MAX_ATTIRE_ITEM_LENGTH", 5)
    with pytest.raises(AttireManagerError, match="5 characters or fewer"):
        am.replace_slot("Aria", "top", "way too long")


# ---------------------------------------------------------------------------
# add_item - behavior
# ---------------------------------------------------------------------------

def test_add_item_creates_a_new_character_on_first_call(tmp_attire_file):
    record, added = am.add_item("Aria", "top", "leather jacket")

    assert added is True
    assert record["name"] == "Aria"
    assert record["slots"]["top"] == ["leather jacket"]


def test_add_item_appends_without_touching_other_items_in_the_slot(tmp_attire_file):
    """The core fix this whole schema change exists for: layering one
    item onto another must never be able to erase what was already
    there. This is the exact socks/shoes scenario from the live bug."""
    am.add_item("Aria", "feet", "yellow socks")

    record, added = am.add_item("Aria", "feet", "black shoes")

    assert added is True
    assert record["slots"]["feet"] == ["yellow socks", "black shoes"]


def test_add_item_does_not_touch_other_slots(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")

    record, _ = am.add_item("Aria", "feet", "boots")

    assert record["slots"]["top"] == ["jacket"]
    assert record["slots"]["feet"] == ["boots"]


def test_add_item_is_a_no_op_when_item_already_present(tmp_attire_file):
    am.add_item("Aria", "feet", "black shoes")

    record, added = am.add_item("Aria", "feet", "black shoes")

    assert added is False
    assert record["slots"]["feet"] == ["black shoes"]  # not duplicated


def test_add_item_dedup_check_is_case_and_whitespace_insensitive(tmp_attire_file):
    am.add_item("Aria", "feet", "black shoes")

    record, added = am.add_item("Aria", "feet", "  BLACK shoes  ")

    assert added is False
    assert record["slots"]["feet"] == ["black shoes"]


def test_add_item_strips_whitespace_from_item_text(tmp_attire_file):
    record, _ = am.add_item("Aria", "top", "  leather jacket  ")
    assert record["slots"]["top"] == ["leather jacket"]


def test_add_item_is_case_insensitive_for_an_existing_character(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")

    record, added = am.add_item("ARIA", "feet", "boots")

    assert added is True
    state = am._load()
    assert len(state["characters"]) == 1  # same character, not a duplicate


def test_add_item_persists_across_separate_load_calls(tmp_attire_file):
    """Confirms _save() actually wrote to disk, not just mutated an
    in-memory dict that happened to still be around."""
    am.add_item("Aria", "top", "jacket")

    reloaded = am._load()
    record = am.resolve_character(reloaded, "Aria", create_if_missing=False)

    assert record["slots"]["top"] == ["jacket"]


def test_add_item_sets_updated_at_on_change(tmp_attire_file):
    with freeze_time("2026-08-26 10:00:00"):
        record, added = am.add_item("Aria", "top", "jacket")
        assert added
        assert record["updated_at"].startswith("2026-08-26T10:00:00")


def test_add_item_does_not_bump_updated_at_when_already_present(tmp_attire_file):
    with freeze_time("2026-08-26 10:00:00"):
        am.add_item("Aria", "top", "jacket")

    with freeze_time("2026-08-27 12:00:00"):
        record, added = am.add_item("Aria", "top", "jacket")

    assert added is False
    assert record["updated_at"].startswith("2026-08-26T10:00:00")  # unchanged


# ---------------------------------------------------------------------------
# remove_item - behavior
# ---------------------------------------------------------------------------

def test_remove_item_removes_an_exact_match(tmp_attire_file):
    am.add_item("Aria", "feet", "yellow socks")
    am.add_item("Aria", "feet", "black shoes")

    record, removed = am.remove_item("Aria", "feet", "black shoes")

    assert removed == "black shoes"
    assert record["slots"]["feet"] == ["yellow socks"]


def test_remove_item_leaves_other_items_in_the_slot_untouched(tmp_attire_file):
    """Mirrors add_item's core-fix test from the other direction:
    removing one item must never touch a sibling item in the same slot."""
    am.add_item("Aria", "feet", "yellow socks")
    am.add_item("Aria", "feet", "black shoes")

    am.remove_item("Aria", "feet", "black shoes")

    state = am._load()
    record = am.resolve_character(state, "Aria", create_if_missing=False)
    assert record["slots"]["feet"] == ["yellow socks"]


def test_remove_item_matches_case_insensitively(tmp_attire_file):
    am.add_item("Aria", "top", "Leather Jacket")

    record, removed = am.remove_item("Aria", "top", "leather jacket")

    assert removed == "Leather Jacket"
    assert record["slots"]["top"] == []


def test_remove_item_matches_via_substring_when_hint_is_shorter_than_stored(tmp_attire_file):
    """'jacket' should match a stored 'denim jacket' - the item-identity
    fuzziness explicitly accepted in the module docstring."""
    am.add_item("Aria", "top", "denim jacket")

    record, removed = am.remove_item("Aria", "top", "jacket")

    assert removed == "denim jacket"
    assert record["slots"]["top"] == []


def test_remove_item_matches_via_substring_when_hint_is_longer_than_stored(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")

    record, removed = am.remove_item("Aria", "top", "her denim jacket")

    assert removed == "jacket"
    assert record["slots"]["top"] == []


def test_remove_item_is_a_no_op_when_nothing_matches(tmp_attire_file):
    am.add_item("Aria", "feet", "yellow socks")

    record, removed = am.remove_item("Aria", "feet", "sandals")

    assert removed is None
    assert record["slots"]["feet"] == ["yellow socks"]  # unchanged


def test_remove_item_is_a_no_op_when_the_hint_is_ambiguous(tmp_attire_file):
    """Two candidate substring matches - not confident enough to guess
    which one the narrative meant, so nothing changes. Fail-open, per
    the module docstring."""
    am.add_item("Aria", "top", "denim jacket")
    am.add_item("Aria", "top", "leather jacket")

    record, removed = am.remove_item("Aria", "top", "jacket")

    assert removed is None
    assert record["slots"]["top"] == ["denim jacket", "leather jacket"]  # unchanged


def test_remove_item_persists_across_separate_load_calls(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")
    am.remove_item("Aria", "top", "jacket")

    reloaded = am._load()
    record = am.resolve_character(reloaded, "Aria", create_if_missing=False)

    assert record["slots"]["top"] == []


def test_remove_item_sets_updated_at_only_when_something_was_actually_removed(tmp_attire_file):
    with freeze_time("2026-08-26 10:00:00"):
        am.add_item("Aria", "top", "jacket")

    with freeze_time("2026-08-27 12:00:00"):
        record, removed = am.remove_item("Aria", "top", "no such item")

    assert removed is None
    assert record["updated_at"].startswith("2026-08-26T10:00:00")  # unchanged


# ---------------------------------------------------------------------------
# replace_slot - behavior
# ---------------------------------------------------------------------------

def test_replace_slot_creates_a_new_character_on_first_call(tmp_attire_file):
    record, changed = am.replace_slot("Aria", "top", "leather jacket")

    assert changed is True
    assert record["slots"]["top"] == ["leather jacket"]


def test_replace_slot_splits_on_commas(tmp_attire_file):
    record, changed = am.replace_slot("Aria", "accessories", "necklace, gloves")

    assert record["slots"]["accessories"] == ["necklace", "gloves"]
    assert changed is True


def test_replace_slot_strips_whitespace_around_each_item(tmp_attire_file):
    record, _ = am.replace_slot("Aria", "accessories", "  necklace ,  gloves  ")
    assert record["slots"]["accessories"] == ["necklace", "gloves"]


def test_replace_slot_drops_empty_items_between_commas(tmp_attire_file):
    record, _ = am.replace_slot("Aria", "accessories", "necklace, , gloves")
    assert record["slots"]["accessories"] == ["necklace", "gloves"]


def test_replace_slot_empty_string_clears_the_list(tmp_attire_file):
    am.replace_slot("Aria", "accessories", "necklace, gloves")

    record, changed = am.replace_slot("Aria", "accessories", "")

    assert record["slots"]["accessories"] == []
    assert changed is True


def test_replace_slot_whitespace_only_string_clears_the_list(tmp_attire_file):
    am.replace_slot("Aria", "accessories", "necklace")

    record, changed = am.replace_slot("Aria", "accessories", "   ")

    assert record["slots"]["accessories"] == []
    assert changed is True


def test_replace_slot_no_change_when_list_is_identical(tmp_attire_file):
    am.replace_slot("Aria", "accessories", "necklace, gloves")

    record, changed = am.replace_slot("Aria", "accessories", "necklace, gloves")

    assert changed is False


def test_replace_slot_order_change_counts_as_a_change(tmp_attire_file):
    """Lists are compared positionally, not as sets - re-ordering the
    same items is treated as a real change. Documenting current
    behavior, not asserting it's the only reasonable choice."""
    am.replace_slot("Aria", "accessories", "necklace, gloves")

    record, changed = am.replace_slot("Aria", "accessories", "gloves, necklace")

    assert changed is True
    assert record["slots"]["accessories"] == ["gloves", "necklace"]


def test_replace_slot_wipes_items_that_add_item_would_have_preserved(tmp_attire_file):
    """The deliberate distinction from add_item: this is the one call
    allowed to discard something still being worn - e.g. a genuine full
    outfit change scene, not a single item coming on or off."""
    am.add_item("Aria", "feet", "yellow socks")
    am.add_item("Aria", "feet", "black shoes")

    record, changed = am.replace_slot("Aria", "feet", "sandals")

    assert changed is True
    assert record["slots"]["feet"] == ["sandals"]  # socks and shoes both gone


def test_replace_slot_persists_across_separate_load_calls(tmp_attire_file):
    am.replace_slot("Aria", "top", "jacket")

    reloaded = am._load()
    record = am.resolve_character(reloaded, "Aria", create_if_missing=False)

    assert record["slots"]["top"] == ["jacket"]


def test_replace_slot_sets_updated_at_on_change(tmp_attire_file):
    with freeze_time("2026-08-26 10:00:00"):
        record, changed = am.replace_slot("Aria", "top", "jacket")
        assert changed
        assert record["updated_at"].startswith("2026-08-26T10:00:00")


def test_replace_slot_does_not_bump_updated_at_when_nothing_actually_changed(tmp_attire_file):
    with freeze_time("2026-08-26 10:00:00"):
        am.replace_slot("Aria", "top", "jacket")

    with freeze_time("2026-08-27 12:00:00"):
        record, changed = am.replace_slot("Aria", "top", "jacket")  # same value again

    assert changed is False
    assert record["updated_at"].startswith("2026-08-26T10:00:00")  # unchanged


# ---------------------------------------------------------------------------
# get_attire_text
# ---------------------------------------------------------------------------

def test_get_attire_text_for_untracked_character_returns_the_error_message(tmp_attire_file):
    result = am.get_attire_text("Nobody")
    assert "No attire record exists yet" in result


def test_get_attire_text_includes_character_name_and_all_slots(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")
    am.add_item("Aria", "feet", "boots")
    am.add_item("Aria", "accessories", "necklace")

    result = am.get_attire_text("Aria")

    assert "Aria" in result
    assert "top: jacket" in result
    assert "feet: boots" in result
    assert "head: (none)" in result
    assert "accessories: necklace" in result


def test_get_attire_text_shows_none_for_an_empty_slot(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")
    result = am.get_attire_text("Aria")
    assert "accessories: (none)" in result


def test_get_attire_text_lists_multiple_items_in_a_slot_comma_separated(tmp_attire_file):
    am.add_item("Aria", "feet", "yellow socks")
    am.add_item("Aria", "feet", "black shoes")

    result = am.get_attire_text("Aria")

    assert "feet: yellow socks, black shoes" in result


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
    am.add_item("Aria", "top", "jacket")
    state = am._load()
    aria_id = next(iter(state["characters"]))

    result = am.build_context_text_for_ids(state, [aria_id])

    assert "[PERSISTENT ATTIRE STATE]" in result
    assert "Aria" in result
    assert "top: jacket" in result


def test_build_context_is_purely_informational_and_names_no_tool(tmp_attire_file):
    """The main agent has no attire tools of its own - attire tracking is
    handled entirely by the separate post-turn attire_subagent.py pass
    (see main.py's ATTIRE_TOOL_SCHEMAS, which are only ever handed to
    that sub-agent). This block must never instruct the main agent to
    call anything, since it has nothing to call."""
    am.add_item("Aria", "top", "jacket")
    state = am._load()
    aria_id = next(iter(state["characters"]))

    result = am.build_context_text_for_ids(state, [aria_id])

    assert "maintained automatically" in result
    assert "attire_add_item" not in result
    assert "attire_remove_item" not in result
    assert "attire_replace_slot" not in result
    assert "attire_manager_update" not in result


def test_build_context_shows_multiple_items_in_a_layered_slot(tmp_attire_file):
    am.add_item("Aria", "feet", "yellow socks")
    am.add_item("Aria", "feet", "black shoes")
    state = am._load()
    aria_id = next(iter(state["characters"]))

    result = am.build_context_text_for_ids(state, [aria_id])

    assert "feet: yellow socks, black shoes" in result


def test_build_context_includes_multiple_characters_when_multiple_ids_given(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")
    am.add_item("Kai", "top", "hoodie")
    state = am._load()
    ids = list(state["characters"].keys())

    result = am.build_context_text_for_ids(state, ids)

    assert "Aria" in result
    assert "Kai" in result
    assert "jacket" in result
    assert "hoodie" in result


def test_build_context_only_includes_ids_that_still_exist_even_when_mixed_with_missing(tmp_attire_file):
    am.add_item("Aria", "top", "jacket")
    state = am._load()
    aria_id = next(iter(state["characters"]))

    result = am.build_context_text_for_ids(state, [aria_id, "not-a-real-id"])

    assert "Aria" in result
    assert result.count("[PERSISTENT ATTIRE STATE]") == 1  # header appears once, not per id
