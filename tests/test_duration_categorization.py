"""
Tests for duration_manager.py's resolve_category() and
confirm_new_category(). resolve_category has four resolution tiers,
checked in order - these tests are organized around proving each tier
independently, and that an earlier tier always wins over a later one
when both would match:
  1. DURATION_CATEGORY_ALIASES exact hit (config.py's hand-maintained
     synonym table)
  2. Exact/label/substring match against canonical + custom category
     names themselves
  3. Embedding-similarity fallback (fake-controlled here - see
     tests/fakes/fake_memory.py for why real embeddings aren't used)
  4. No match close enough -> None

Tier 3 tests explicitly set BOTH the query vector and any category
vectors that matter, rather than relying on the fake embedder's shared
default for both sides - see test_duration_hooks.py's uncategorized
test for why relying on the shared default for two different pieces of
text is a trap (everything defaults to the same vector unless told
otherwise, which trivially "matches").
"""
import pytest

import config
import duration_manager as dm
from duration_manager import DurationError


# ---------------------------------------------------------------------------
# resolve_category - empty/trivial input
# ---------------------------------------------------------------------------

def test_empty_text_returns_none(tmp_duration_file):
    assert dm.resolve_category("") is None
    assert dm.resolve_category("   ") is None


# ---------------------------------------------------------------------------
# resolve_category - tier 1: alias table
# ---------------------------------------------------------------------------

def test_alias_exact_match_resolves_without_touching_embeddings(tmp_duration_file, fake_embedder):
    """"lit review" is a DURATION_CATEGORY_ALIASES key mapping to
    "reading". Deliberately leaves the fake embedder fully unconfigured -
    if this test passes even though embed() would return nonsense for
    unconfigured text, it proves resolve_category never touched the
    embedding path for an alias hit at all."""
    assert dm.resolve_category("lit review") == "reading"


def test_alias_match_is_case_and_whitespace_insensitive(tmp_duration_file):
    assert dm.resolve_category("  Lit   Review  ") == "reading"


# ---------------------------------------------------------------------------
# resolve_category - tier 2: category name / label / substring match
# ---------------------------------------------------------------------------

def test_exact_canonical_category_name_matches_itself(tmp_duration_file):
    assert dm.resolve_category("household") == "household"


def test_canonical_category_underscore_label_matches_spaced_form(tmp_duration_file):
    """"data_analysis" is stored with an underscore; the label form with a
    space ("data analysis") must resolve to the same canonical name -
    this is also a DURATION_CATEGORY_ALIASES entry, so this doubles as a
    tier-1/tier-2 agreement check, not a conflict."""
    assert dm.resolve_category("data analysis") == "data_analysis"


def test_substring_match_against_a_category_label(tmp_duration_file):
    """"planning" has no alias table entry - "weekly planning session"
    should still resolve via the substring check (label in norm)."""
    assert dm.resolve_category("weekly planning session") == "planning"


def test_custom_category_participates_in_tier_2_matching(tmp_duration_file):
    dm.confirm_new_category("side_projects", reclassify_last=False)
    assert dm.resolve_category("side_projects") == "side_projects"
    # Substring match compares against category.replace("_", " ") - a
    # SPACE-separated label, not the underscore form - so the query text
    # needs the space form too to hit the substring check.
    assert dm.resolve_category("some side projects work") == "side_projects"


# ---------------------------------------------------------------------------
# resolve_category - tier 3: embedding-similarity fallback
# ---------------------------------------------------------------------------

def test_embedding_fallback_resolves_above_threshold(tmp_duration_file, fake_embedder):
    """Query text has no alias/exact/substring match to any canonical
    category, but is pinned to the SAME vector as the "writing" category
    label gets (the shared default) - guaranteeing a perfect 1.0 score,
    comfortably above DURATION_CATEGORY_SIMILARITY_THRESHOLD (0.5)."""
    fake_embedder.set("Ambiguous novel phrase", [1.0, 0.0, 0.0])  # matches the fake embedder's default,
                                                                    # which every category label also gets
    assert dm.resolve_category("Ambiguous novel phrase") == "reading"  # first canonical category, ties broken by iteration order


def test_embedding_fallback_returns_none_below_threshold(tmp_duration_file, fake_embedder):
    fake_embedder.set("Totally unrelated phrase", [0.0, 1.0, 0.0])  # orthogonal to the [1,0,0] default every category gets
    assert dm.resolve_category("Totally unrelated phrase") is None


def test_embedding_fallback_picks_the_best_scoring_category_not_just_the_first_above_threshold(tmp_duration_file, fake_embedder):
    """Two categories score above threshold; resolve_category must return
    the highest-scoring one, not just the first one that clears the bar
    during iteration. Every OTHER category is pushed to the fake
    embedder's default, set here to something orthogonal to the query -
    otherwise they'd silently tie with "writing" at the shared default
    and "reading" (first in DURATION_CANONICAL_CATEGORIES) would win by
    iteration order alone, which is exactly the bug this test exists to
    rule out."""
    fake_embedder.set_default([0.0, 0.0, 1.0])        # every other category: orthogonal, scores ~0
    fake_embedder.set("writing", [1.0, 0.0, 0.0])     # "writing" category label vector
    fake_embedder.set("meetings", [0.9, 0.436, 0.0])  # "meetings" category label vector, high but lower score
    fake_embedder.set("Query text", [1.0, 0.0, 0.0])  # exact match to "writing" -> score 1.0 beats "meetings"'s ~0.9

    assert dm.resolve_category("Query text") == "writing"


def test_embedding_fallback_threshold_boundary_is_inclusive(tmp_duration_file, fake_embedder):
    """Score exactly equal to DURATION_CATEGORY_SIMILARITY_THRESHOLD
    (0.5) must count as a match (`>=`, not `>`, per the source)."""
    import math
    # unit vector at 60 degrees from [1,0,0] has dot product exactly 0.5
    angled = [math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0]
    fake_embedder.set("Boundary phrase", angled)
    assert dm.resolve_category("Boundary phrase") == "reading"


# ---------------------------------------------------------------------------
# confirm_new_category
# ---------------------------------------------------------------------------

def test_confirm_new_category_requires_a_name(tmp_duration_file):
    with pytest.raises(DurationError, match="A category name is required"):
        dm.confirm_new_category("")


def test_confirm_new_category_adds_to_custom_categories(tmp_duration_file):
    dm.confirm_new_category("side projects", reclassify_last=False)
    state = dm._load()
    assert "side_projects" in state["custom_categories"]


def test_confirm_new_category_is_idempotent_for_an_existing_canonical_name(tmp_duration_file):
    dm.confirm_new_category("reading", reclassify_last=False)
    state = dm._load()
    assert state["custom_categories"] == []  # already canonical - not duplicated into custom_categories


def test_confirm_new_category_reclassifies_the_most_recent_uncategorized_entry(tmp_duration_file, fake_embedder):
    fake_embedder.set("Odd task", [0.0, 1.0, 0.0])  # forces uncategorized (orthogonal to default)

    with_active = "task_1"
    dm.on_task_active(with_active, "project_1")
    dm.on_task_done(with_active, "project_1", "Odd task")

    dm.confirm_new_category("side projects", reclassify_last=True)

    state = dm._load()
    assert state["entries"][0]["category"] == "side_projects"


def test_confirm_new_category_reclassify_only_touches_the_most_recent_uncategorized_entry(tmp_duration_file, fake_embedder):
    fake_embedder.set("Odd task one", [0.0, 1.0, 0.0])
    fake_embedder.set("Odd task two", [0.0, 1.0, 0.0])

    dm.on_task_active("task_1", "project_1")
    dm.on_task_done("task_1", "project_1", "Odd task one")
    dm.on_task_active("task_2", "project_1")
    dm.on_task_done("task_2", "project_1", "Odd task two")

    dm.confirm_new_category("side projects", reclassify_last=True)

    state = dm._load()
    categories = [e["category"] for e in state["entries"]]
    assert categories == ["uncategorized", "side_projects"]  # only the most recent (last) one changed


def test_confirm_new_category_without_reclassify_leaves_entries_untouched(tmp_duration_file, fake_embedder):
    fake_embedder.set("Odd task", [0.0, 1.0, 0.0])
    dm.on_task_active("task_1", "project_1")
    dm.on_task_done("task_1", "project_1", "Odd task")

    dm.confirm_new_category("side projects", reclassify_last=False)

    state = dm._load()
    assert state["entries"][0]["category"] == "uncategorized"
