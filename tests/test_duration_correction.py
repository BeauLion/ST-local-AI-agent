"""
Tests for duration_manager.py's correct_entry() - lets the user manually
override a logged duration after the fact. Match priority, per the
source, is:
  1. Title substring match (most recent by timestamp among matches)
  2. Category match via resolve_category (most recent among matches)
  3. No query at all -> the single most recent entry overall
  4. Nothing matches -> DurationError
"""
import pytest

import duration_manager as dm
from duration_manager import DurationError


def _seed_entry(tmp_duration_file, *, title, category, minutes, timestamp):
    state = dm._load()
    state["entries"].append({
        "id": f"dur_{title.replace(' ', '_')}", "category": category,
        "elapsed_anchor_minutes": minutes, "logged_value_minutes": minutes,
        "confirmation_state": "accepted", "task_id": f"task_{title}",
        "project_id": "project_1", "title": title, "timestamp": timestamp,
    })
    dm._save(state)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_unparseable_value_raises(tmp_duration_file):
    _seed_entry(tmp_duration_file, title="Standup", category="meetings", minutes=15, timestamp="2026-08-26T09:00:00+00:00")
    with pytest.raises(DurationError, match="Couldn't parse a duration"):
        dm.correct_entry("Standup", "not a duration")


def test_no_entries_at_all_raises(tmp_duration_file):
    with pytest.raises(DurationError, match="No logged durations exist yet"):
        dm.correct_entry("anything", "20min")


# ---------------------------------------------------------------------------
# Title match (tier 1)
# ---------------------------------------------------------------------------

def test_title_substring_match_updates_the_entry(tmp_duration_file):
    _seed_entry(tmp_duration_file, title="Weekly Standup", category="meetings", minutes=15, timestamp="2026-08-26T09:00:00+00:00")

    result = dm.correct_entry("standup", "10min")

    assert "Weekly Standup" in result
    assert "~10 min" in result
    state = dm._load()
    assert state["entries"][0]["logged_value_minutes"] == 10.0
    assert state["entries"][0]["confirmation_state"] == "corrected"


def test_title_match_picks_the_most_recent_when_multiple_match(tmp_duration_file):
    _seed_entry(tmp_duration_file, title="Standup Monday", category="meetings", minutes=15, timestamp="2026-08-24T09:00:00+00:00")
    _seed_entry(tmp_duration_file, title="Standup Tuesday", category="meetings", minutes=15, timestamp="2026-08-25T09:00:00+00:00")

    dm.correct_entry("standup", "20min")

    state = dm._load()
    updated = [e for e in state["entries"] if e["confirmation_state"] == "corrected"]
    assert len(updated) == 1
    assert updated[0]["title"] == "Standup Tuesday"


# ---------------------------------------------------------------------------
# Category match (tier 2) - only reached when title match fails entirely
# ---------------------------------------------------------------------------

def test_category_match_used_when_title_does_not_match_anything(tmp_duration_file):
    _seed_entry(tmp_duration_file, title="Grant application", category="research_admin", minutes=45, timestamp="2026-08-26T09:00:00+00:00")

    result = dm.correct_entry("research_admin", "60min")

    assert "Grant application" in result
    state = dm._load()
    assert state["entries"][0]["logged_value_minutes"] == 60.0


def test_category_match_via_alias(tmp_duration_file):
    """"lit review" resolves to "reading" via the alias table - the same
    resolution get_estimate relies on."""
    _seed_entry(tmp_duration_file, title="Paper on transformers", category="reading", minutes=45, timestamp="2026-08-26T09:00:00+00:00")

    result = dm.correct_entry("lit review", "50min")

    assert "Paper on transformers" in result
    state = dm._load()
    assert state["entries"][0]["logged_value_minutes"] == 50.0


def test_category_match_picks_the_most_recent_among_category_matches(tmp_duration_file):
    _seed_entry(tmp_duration_file, title="First reading", category="reading", minutes=30, timestamp="2026-08-24T09:00:00+00:00")
    _seed_entry(tmp_duration_file, title="Second reading", category="reading", minutes=30, timestamp="2026-08-25T09:00:00+00:00")

    dm.correct_entry("reading", "40min")

    state = dm._load()
    updated = [e for e in state["entries"] if e["confirmation_state"] == "corrected"]
    assert len(updated) == 1
    assert updated[0]["title"] == "Second reading"


# ---------------------------------------------------------------------------
# No query - most recent entry overall (tier 3)
# ---------------------------------------------------------------------------

def test_empty_query_corrects_the_single_most_recent_entry_overall(tmp_duration_file):
    _seed_entry(tmp_duration_file, title="Old task", category="reading", minutes=30, timestamp="2026-08-24T09:00:00+00:00")
    _seed_entry(tmp_duration_file, title="Newest task", category="email", minutes=10, timestamp="2026-08-26T09:00:00+00:00")

    result = dm.correct_entry("", "15min")

    assert "Newest task" in result
    state = dm._load()
    updated = [e for e in state["entries"] if e["confirmation_state"] == "corrected"]
    assert len(updated) == 1
    assert updated[0]["title"] == "Newest task"


# ---------------------------------------------------------------------------
# Nothing matches at all
# ---------------------------------------------------------------------------

def test_query_matching_nothing_raises(tmp_duration_file, fake_embedder):
    fake_embedder.set("completely unrelated", [0.0, 1.0, 0.0])  # forces resolve_category -> None too
    _seed_entry(tmp_duration_file, title="Standup", category="meetings", minutes=15, timestamp="2026-08-26T09:00:00+00:00")

    with pytest.raises(DurationError, match="Couldn't find a logged duration matching"):
        dm.correct_entry("completely unrelated", "20min")
