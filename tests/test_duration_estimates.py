"""
Tests for duration_manager.py's get_estimate() - resolves a query to a
category (via resolve_category, already covered separately), then
reports a median/MAD estimate with a confidence tier based on how many
logged entries exist for that category. Per config.py:
  MIN_ENTRIES_FOR_ESTIMATE = 5, MIN_ENTRIES_FOR_CONFIDENT_ESTIMATE = 10
  n == 0            -> confidence "insufficient", median/mad None
  0 < n < 5         -> confidence "insufficient"
  5 <= n < 10       -> confidence "rough"
  n >= 10           -> confidence "confident"
"""
import duration_manager as dm
import config


def _seed_entries(tmp_duration_file, category: str, values: list):
    """Writes entries directly to the state file - bypasses on_task_done
    entirely so these tests don't depend on the active/inactive lifecycle
    at all, only on get_estimate's own math over whatever's logged."""
    state = dm._load()
    for i, v in enumerate(values):
        state["entries"].append({
            "id": f"dur_test_{category}_{i}", "category": category,
            "elapsed_anchor_minutes": v, "logged_value_minutes": v,
            "confirmation_state": "accepted", "task_id": f"task_{i}",
            "project_id": "project_1", "title": f"Task {i}",
            "timestamp": dm._now_iso(),
        })
    dm._save(state)


# ---------------------------------------------------------------------------
# Category doesn't resolve at all
# ---------------------------------------------------------------------------

def test_unresolvable_query_returns_not_resolved(tmp_duration_file, fake_embedder):
    fake_embedder.set("Totally unrelated phrase", [0.0, 1.0, 0.0])
    result = dm.get_estimate("Totally unrelated phrase")
    assert result == {"resolved": False}


# ---------------------------------------------------------------------------
# Category resolves but has zero logged entries
# ---------------------------------------------------------------------------

def test_resolved_category_with_no_entries(tmp_duration_file):
    result = dm.get_estimate("reading")
    assert result["resolved"] is True
    assert result["category"] == "reading"
    assert result["n"] == 0
    assert result["confidence"] == "insufficient"
    assert result["median_minutes"] is None
    assert result["mad_minutes"] is None


# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------

def test_below_minimum_entries_is_insufficient(tmp_duration_file):
    _seed_entries(tmp_duration_file, "reading", [30, 40, 45, 50])  # 4 entries, < MIN_ENTRIES_FOR_ESTIMATE (5)
    result = dm.get_estimate("reading")
    assert result["n"] == 4
    assert result["confidence"] == "insufficient"


def test_at_minimum_entries_is_rough(tmp_duration_file):
    _seed_entries(tmp_duration_file, "reading", [30, 40, 45, 50, 35])  # exactly 5
    result = dm.get_estimate("reading")
    assert result["n"] == 5
    assert result["confidence"] == "rough"


def test_just_below_confident_threshold_is_still_rough(tmp_duration_file):
    _seed_entries(tmp_duration_file, "reading", [30] * 9)  # 9, < MIN_ENTRIES_FOR_CONFIDENT_ESTIMATE (10)
    result = dm.get_estimate("reading")
    assert result["n"] == 9
    assert result["confidence"] == "rough"


def test_at_confident_threshold(tmp_duration_file):
    _seed_entries(tmp_duration_file, "reading", [30] * 10)  # exactly 10
    result = dm.get_estimate("reading")
    assert result["n"] == 10
    assert result["confidence"] == "confident"


# ---------------------------------------------------------------------------
# Median / MAD calculation
# ---------------------------------------------------------------------------

def test_median_and_mad_are_calculated_correctly(tmp_duration_file):
    # values: 10, 20, 30, 40, 50 -> median 30; abs deviations: 20,10,0,10,20 -> MAD median = 10
    _seed_entries(tmp_duration_file, "reading", [10, 20, 30, 40, 50])
    result = dm.get_estimate("reading")
    assert result["median_minutes"] == 30.0
    assert result["mad_minutes"] == 10.0


def test_only_entries_for_the_resolved_category_are_counted(tmp_duration_file):
    _seed_entries(tmp_duration_file, "reading", [30, 30, 30, 30, 30])
    _seed_entries(tmp_duration_file, "writing", [999, 999])  # different category - must not leak in

    result = dm.get_estimate("reading")
    assert result["n"] == 5
    assert result["median_minutes"] == 30.0


def test_query_resolves_via_alias_before_counting_entries(tmp_duration_file):
    """get_estimate("lit review") should resolve to "reading" via the
    alias table, then count entries logged under "reading" - not require
    entries to literally be logged as "lit review"."""
    _seed_entries(tmp_duration_file, "reading", [20, 20, 20, 20, 20])
    result = dm.get_estimate("lit review")
    assert result["category"] == "reading"
    assert result["n"] == 5
