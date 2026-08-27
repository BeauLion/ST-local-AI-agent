"""
Tests for memory.py's "personal memory" functions: save_memory (both slot
and freeform modes), update_memory, delete_memory, pin_memory,
unpin_memory, list_memories, list_memories_full, get_pinned_memories,
search_memories.

memory.py itself owns the real embedding model (unlike duration_manager,
which only consumed memory.embed()) - fake_memory_embedder here controls
memory._model directly (see tests/fakes/fake_sentence_transformers.py).
Most tests below don't care about actual embedding VALUES (they're
testing storage/ordering/status logic, not similarity search), so they
leave the fake embedder at its default and only configure specific
vectors where a test is actually about similarity (dedupe, search).

No Pydantic migration for this module: none of these functions validate
their inputs before acting (no "text is required" checks anywhere) -
callers (main.py's tool wrappers) are expected to have already
validated. That's a real, pre-existing asymmetry with calendar_manager/
duration_manager/project_manager, not something these tests should paper
over by inventing validation that doesn't exist.
"""
import json

import memory


def _read_memories(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# save_memory - freeform mode
# ---------------------------------------------------------------------------

def test_save_memory_freeform_creates_a_new_entry(tmp_memory_file):
    result = memory.save_memory("I like coffee")

    assert result["action"] == "created"
    assert result["slot"] is None
    assert result["similar"] is None

    stored = _read_memories(tmp_memory_file)
    assert stored[0]["text"] == "I like coffee"
    assert stored[0]["slot"] is None
    assert stored[0]["pinned"] is False


def test_save_memory_freeform_never_overwrites_existing_entries(tmp_memory_file):
    memory.save_memory("First fact")
    memory.save_memory("Second fact")

    stored = _read_memories(tmp_memory_file)
    assert len(stored) == 2


def test_save_memory_freeform_flags_a_close_dupe_but_still_saves(tmp_memory_file, fake_memory_embedder):
    import config
    fake_memory_embedder.set("I like coffee", [1.0, 0.0, 0.0])
    fake_memory_embedder.set("I love coffee", [1.0, 0.0, 0.0])  # identical vector - score 1.0, well above dedupe threshold

    memory.save_memory("I like coffee")
    result = memory.save_memory("I love coffee")

    assert result["action"] == "created"  # still saved, not blocked
    assert result["similar"] is not None
    assert result["similar"]["text"] == "I like coffee"

    stored = _read_memories(tmp_memory_file)
    assert len(stored) == 2  # both entries exist


def test_save_memory_freeform_below_dedupe_threshold_reports_no_similar(tmp_memory_file, fake_memory_embedder):
    fake_memory_embedder.set("I like coffee", [1.0, 0.0, 0.0])
    fake_memory_embedder.set("Completely unrelated fact", [0.0, 1.0, 0.0])  # orthogonal - score 0.0

    memory.save_memory("I like coffee")
    result = memory.save_memory("Completely unrelated fact")

    assert result["similar"] is None


def test_save_memory_freeform_dedupe_checks_against_slotted_memories_too(tmp_memory_file, fake_memory_embedder):
    """Module docstring: "a freeform fact silently contradicting a slot is
    the case most worth catching" - the dedupe check must scan ALL
    memories, not just other freeform ones."""
    fake_memory_embedder.set("My name is Alex", [1.0, 0.0, 0.0])
    fake_memory_embedder.set("Actually my name is Sam", [1.0, 0.0, 0.0])

    memory.save_memory("My name is Alex", slot="identity")
    result = memory.save_memory("Actually my name is Sam")

    assert result["similar"] is not None
    assert result["similar"]["text"] == "My name is Alex"


# ---------------------------------------------------------------------------
# save_memory - slot mode
# ---------------------------------------------------------------------------

def test_save_memory_slot_creates_a_pinned_entry(tmp_memory_file):
    result = memory.save_memory("Alex", slot="identity")

    assert result["action"] == "created"
    assert result["slot"] == "identity"

    stored = _read_memories(tmp_memory_file)
    assert stored[0]["pinned"] is True
    assert stored[0]["slot"] == "identity"


def test_save_memory_slot_upserts_in_place_on_second_call(tmp_memory_file):
    first = memory.save_memory("Alex", slot="identity")
    second = memory.save_memory("Alexandra", slot="identity")

    assert second["action"] == "updated"
    assert second["id"] == first["id"]

    stored = _read_memories(tmp_memory_file)
    assert len(stored) == 1  # not duplicated
    assert stored[0]["text"] == "Alexandra"


def test_save_memory_slot_never_runs_the_dedupe_check(tmp_memory_file, fake_memory_embedder):
    """Slot mode's docstring: "No dedupe check - upserting by construction
    can't create a duplicate." Even with a fake embedder guaranteed to
    return a perfect-match score, slot mode's result must not report a
    "similar" match - the field doesn't apply to this mode at all."""
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])  # every call gets an identical vector

    memory.save_memory("Alex", slot="identity")
    result = memory.save_memory("Sam", slot="occupation")

    assert result["similar"] is None


def test_save_memory_different_slots_create_separate_entries(tmp_memory_file):
    memory.save_memory("Alex", slot="identity")
    memory.save_memory("Engineer", slot="occupation")

    stored = _read_memories(tmp_memory_file)
    assert len(stored) == 2


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------

def test_update_memory_replaces_text_by_id(tmp_memory_file):
    created = memory.save_memory("Old fact")
    result = memory.update_memory(created["id"], "New fact")

    assert "Updated memory" in result
    stored = _read_memories(tmp_memory_file)
    assert stored[0]["text"] == "New fact"


def test_update_memory_unknown_id_returns_a_message_not_an_exception(tmp_memory_file):
    result = memory.update_memory("nonexistent", "New fact")
    assert "No memory found" in result


def test_update_memory_leaves_slot_and_pinned_untouched(tmp_memory_file):
    created = memory.save_memory("Alex", slot="identity")
    memory.update_memory(created["id"], "Alexandra")

    stored = _read_memories(tmp_memory_file)
    assert stored[0]["slot"] == "identity"
    assert stored[0]["pinned"] is True


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------

def test_delete_memory_removes_the_entry(tmp_memory_file):
    created = memory.save_memory("A fact")
    result = memory.delete_memory(created["id"])

    assert "Deleted memory" in result
    assert _read_memories(tmp_memory_file) == []


def test_delete_memory_unknown_id_returns_a_message_not_an_exception(tmp_memory_file):
    result = memory.delete_memory("nonexistent")
    assert "No memory found" in result


def test_delete_slotted_memory_empties_the_slot_for_reuse(tmp_memory_file):
    created = memory.save_memory("Alex", slot="identity")
    memory.delete_memory(created["id"])

    second = memory.save_memory("Sam", slot="identity")
    assert second["action"] == "created"  # slot is free again, not still "occupied"


# ---------------------------------------------------------------------------
# pin_memory / unpin_memory
# ---------------------------------------------------------------------------

def test_pin_memory_pins_a_freeform_memory(tmp_memory_file):
    created = memory.save_memory("A fact")
    result = memory.pin_memory(created["id"])

    assert "Pinned memory" in result
    stored = _read_memories(tmp_memory_file)
    assert stored[0]["pinned"] is True


def test_pin_memory_unknown_id_returns_a_message(tmp_memory_file):
    assert "No memory found" in memory.pin_memory("nonexistent")


def test_pin_memory_on_a_slotted_memory_is_a_no_op_with_explanation(tmp_memory_file):
    created = memory.save_memory("Alex", slot="identity")
    result = memory.pin_memory(created["id"])
    assert "already always shown" in result


def test_pin_memory_already_pinned_freeform_is_a_no_op_with_explanation(tmp_memory_file):
    created = memory.save_memory("A fact")
    memory.pin_memory(created["id"])
    result = memory.pin_memory(created["id"])
    assert "already pinned" in result


def test_pin_memory_respects_the_freeform_pin_limit(tmp_memory_file):
    import config
    ids = [memory.save_memory(f"Fact {i}")["id"] for i in range(config.MEMORY_MAX_FREEFORM_PINS)]
    for mid in ids:
        memory.pin_memory(mid)

    overflow = memory.save_memory("One too many")
    result = memory.pin_memory(overflow["id"])

    assert "pin limit" in result
    stored = _read_memories(tmp_memory_file)
    overflow_entry = next(m for m in stored if m["id"] == overflow["id"])
    assert overflow_entry["pinned"] is False  # rejected, not silently pinned anyway


def test_unpin_memory_unpins_a_freeform_memory(tmp_memory_file):
    created = memory.save_memory("A fact")
    memory.pin_memory(created["id"])
    result = memory.unpin_memory(created["id"])

    assert "Unpinned memory" in result
    stored = _read_memories(tmp_memory_file)
    assert stored[0]["pinned"] is False


def test_unpin_memory_not_pinned_is_a_no_op_with_explanation(tmp_memory_file):
    created = memory.save_memory("A fact")
    result = memory.unpin_memory(created["id"])
    assert "isn't pinned" in result


def test_unpin_memory_on_a_slotted_memory_is_rejected(tmp_memory_file):
    created = memory.save_memory("Alex", slot="identity")
    result = memory.unpin_memory(created["id"])
    assert "can't be unpinned" in result

    stored = _read_memories(tmp_memory_file)
    assert stored[0]["pinned"] is True  # unchanged


def test_unpin_memory_unknown_id_returns_a_message(tmp_memory_file):
    assert "No memory found" in memory.unpin_memory("nonexistent")


# ---------------------------------------------------------------------------
# list_memories / list_memories_full / get_pinned_memories
# ---------------------------------------------------------------------------

def test_list_memories_puts_slots_first_in_configured_order(tmp_memory_file):
    import config
    # Save in reverse of MEMORY_IDENTITY_SLOTS order, plus one freeform.
    memory.save_memory("Freeform fact")
    memory.save_memory("NYC", slot="location")
    memory.save_memory("Engineer", slot="occupation")
    memory.save_memory("Alex", slot="identity")

    results = memory.list_memories()
    slots_in_order = [m["slot"] for m in results if m["slot"]]
    assert slots_in_order == list(config.MEMORY_IDENTITY_SLOTS)


def test_list_memories_freeform_entries_sorted_most_recent_first(tmp_memory_file):
    """_now_iso() has only second-level precision - three saves back to
    back in a fast test can land in the same second, so freeze_time with
    explicit ticks is needed to guarantee distinct, orderable
    timestamps."""
    from freezegun import freeze_time
    with freeze_time("2026-08-27 10:00:00") as frozen:
        memory.save_memory("Oldest")
        frozen.tick(1)
        memory.save_memory("Middle")
        frozen.tick(1)
        memory.save_memory("Newest")

    results = memory.list_memories()
    assert [m["text"] for m in results] == ["Newest", "Middle", "Oldest"]


def test_list_memories_omits_embedding_and_timestamps(tmp_memory_file):
    memory.save_memory("A fact")
    result = memory.list_memories()[0]
    assert set(result.keys()) == {"id", "text", "slot", "pinned"}


def test_list_memories_full_includes_timestamps(tmp_memory_file):
    memory.save_memory("A fact")
    result = memory.list_memories_full()[0]
    assert "created_at" in result
    assert "updated_at" in result
    assert "embedding" not in result


def test_get_pinned_memories_returns_only_pinned_entries(tmp_memory_file):
    memory.save_memory("Alex", slot="identity")  # always pinned
    pinned_freeform = memory.save_memory("Pinned fact")
    memory.pin_memory(pinned_freeform["id"])
    memory.save_memory("Unpinned fact")  # not pinned

    results = memory.get_pinned_memories()
    texts = {m["text"] for m in results}
    assert texts == {"Alex", "Pinned fact"}


# ---------------------------------------------------------------------------
# search_memories
# ---------------------------------------------------------------------------

def test_search_memories_returns_matches_above_threshold(tmp_memory_file, fake_memory_embedder):
    fake_memory_embedder.set("I like coffee", [1.0, 0.0, 0.0])
    fake_memory_embedder.set("coffee preferences", [1.0, 0.0, 0.0])  # identical - well above threshold
    memory.save_memory("I like coffee")

    results = memory.search_memories("coffee preferences")
    assert len(results) == 1
    assert results[0]["text"] == "I like coffee"


def test_search_memories_excludes_matches_below_threshold(tmp_memory_file, fake_memory_embedder):
    fake_memory_embedder.set("I like coffee", [1.0, 0.0, 0.0])
    fake_memory_embedder.set("completely unrelated query", [0.0, 1.0, 0.0])  # orthogonal
    memory.save_memory("I like coffee")

    results = memory.search_memories("completely unrelated query")
    assert results == []


def test_search_memories_respects_top_k(tmp_memory_file, fake_memory_embedder):
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])  # everything matches equally well
    for i in range(5):
        memory.save_memory(f"Fact {i}")

    results = memory.search_memories("query", top_k=2)
    assert len(results) == 2


def test_search_memories_result_shape_omits_embedding(tmp_memory_file, fake_memory_embedder):
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])
    memory.save_memory("A fact")
    result = memory.search_memories("query")[0]
    assert set(result.keys()) == {"id", "text", "slot", "pinned"}
