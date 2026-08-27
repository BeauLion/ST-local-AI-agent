"""
Tests for duration_manager.py's on_task_active/on_task_inactive/
on_task_done - the hooks project_manager.py calls on task status
transitions to track how long a task was actually "active" for.

Covers: window open/close lifecycle, the overlap-detection logic that
skips logging when two tasks were active at once (both the
still-active-elsewhere case and the recently-closed-elsewhere case), and
the categorized-vs-uncategorized flag message shapes. Category
RESOLUTION correctness (aliases, embedding fallback, thresholds) is
covered separately in test_duration_categorization.py - these tests
pin exact category names via aliases already in config.py so the
category-related assertions here don't depend on embedding behavior at
all.
"""
import json

from freezegun import freeze_time

import duration_manager as dm


FROZEN_NOW = "2026-08-26 10:00:00+00:00"


def _read_state(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# on_task_active / on_task_inactive
# ---------------------------------------------------------------------------

def test_on_task_active_opens_a_window(tmp_duration_file):
    with freeze_time(FROZEN_NOW):
        dm.on_task_active("task_1", "project_1")

    state = _read_state(tmp_duration_file)
    assert state["active_windows"]["task_1"]["project_id"] == "project_1"
    assert state["active_windows"]["task_1"]["started_at"] == "2026-08-26T10:00:00+00:00"


def test_on_task_inactive_closes_the_window_without_logging(tmp_duration_file):
    dm.on_task_active("task_1", "project_1")
    dm.on_task_inactive("task_1")

    state = _read_state(tmp_duration_file)
    assert state["active_windows"] == {}
    assert state["entries"] == []  # no duration logged - inactive isn't done


def test_on_task_inactive_with_no_open_window_is_a_silent_no_op(tmp_duration_file):
    """No window to close means duration_manager never even calls _save()
    - the file may not exist at all yet. The only real assertion here is
    that this doesn't raise."""
    dm.on_task_inactive("nonexistent_task")  # should not raise


# ---------------------------------------------------------------------------
# on_task_done - the no-window case
# ---------------------------------------------------------------------------

def test_on_task_done_with_no_active_window_logs_nothing_and_returns_none(tmp_duration_file):
    result = dm.on_task_done("task_1", "project_1", "Some Task")
    assert result is None

    state = _read_state(tmp_duration_file)
    assert state["entries"] == []


# ---------------------------------------------------------------------------
# on_task_done - the normal (no overlap) path
# ---------------------------------------------------------------------------

def test_on_task_done_logs_elapsed_time_and_closes_the_window(tmp_duration_file):
    with freeze_time(FROZEN_NOW) as frozen:
        dm.on_task_active("task_1", "project_1")
        frozen.tick(1800)  # 30 minutes

        result = dm.on_task_done("task_1", "project_1", "Reply to emails")

    assert "~30 min logged" in result
    assert "Reply to emails" in result

    state = _read_state(tmp_duration_file)
    assert "task_1" not in state["active_windows"]  # window closed
    assert len(state["entries"]) == 1
    entry = state["entries"][0]
    assert entry["task_id"] == "task_1"
    assert entry["title"] == "Reply to emails"
    assert round(entry["elapsed_anchor_minutes"]) == 30
    assert entry["logged_value_minutes"] == entry["elapsed_anchor_minutes"]
    assert entry["confirmation_state"] == "accepted"


def test_on_task_done_records_a_recently_closed_window(tmp_duration_file):
    with freeze_time(FROZEN_NOW) as frozen:
        dm.on_task_active("task_1", "project_1")
        frozen.tick(600)
        dm.on_task_done("task_1", "project_1", "Quick task")

    state = _read_state(tmp_duration_file)
    assert len(state["recently_closed_windows"]) == 1
    assert state["recently_closed_windows"][0]["task_id"] == "task_1"


def test_on_task_done_categorizes_via_alias_and_uses_the_confident_flag_wording(tmp_duration_file):
    """"email" is a direct DURATION_CATEGORY_ALIASES hit (config.py) - no
    embedding fallback involved, so this is safe to assert without
    touching the fake embedder at all."""
    with freeze_time(FROZEN_NOW) as frozen:
        dm.on_task_active("task_1", "project_1")
        frozen.tick(300)
        result = dm.on_task_done("task_1", "project_1", "email")

    assert "category: email" in result
    assert "Reply if that's off" in result

    state = _read_state(tmp_duration_file)
    assert state["entries"][0]["category"] == "email"


def test_on_task_done_falls_back_to_uncategorized_and_offers_to_track_it(tmp_duration_file, fake_embedder):
    """A title with no alias/exact/substring match, and a fake embedding
    score below DURATION_CATEGORY_SIMILARITY_THRESHOLD against every
    canonical category, must resolve to None -> logged as
    "uncategorized" with the category-suggestion flag wording, not the
    "category: X" wording. Only the QUERY text's vector needs to be
    orthogonal here - category vectors are left at the fake embedder's
    default ([1, 0, 0]), which is the same vector every category label
    gets since none of them are individually configured."""
    fake_embedder.set("Completely novel activity", [0.0, 1.0, 0.0])

    with freeze_time(FROZEN_NOW) as frozen:
        dm.on_task_active("task_1", "project_1")
        frozen.tick(300)
        result = dm.on_task_done("task_1", "project_1", "Completely novel activity")

    assert "(uncategorized)" in result
    assert "tracked category" in result

    state = _read_state(tmp_duration_file)
    assert state["entries"][0]["category"] == "uncategorized"


# ---------------------------------------------------------------------------
# on_task_done - overlap detection
# ---------------------------------------------------------------------------

def test_on_task_done_skips_logging_when_another_task_is_still_active_concurrently(tmp_duration_file):
    with freeze_time(FROZEN_NOW) as frozen:
        dm.on_task_active("task_1", "project_1")
        frozen.tick(60)
        dm.on_task_active("task_2", "project_1")  # started 1 min after task_1, both now active
        frozen.tick(600)

        result = dm.on_task_done("task_1", "project_1", "Overlapped task")

    assert "not logged" in result
    assert "overlapped with another active task" in result

    state = _read_state(tmp_duration_file)
    assert state["entries"] == []  # nothing logged for the overlapping window
    assert "task_1" not in state["active_windows"]  # window still closes even though nothing was logged
    assert "task_2" in state["active_windows"]  # the other task is untouched


def test_on_task_done_skips_logging_when_window_overlaps_a_recently_closed_one(tmp_duration_file):
    """task_2's window fully contains task_1's already-closed window in
    wall-clock time - even though task_1 wasn't literally open anymore
    when task_2 finished, the two overlapped in real time, so task_2's
    duration also shouldn't be trusted as a clean anchor."""
    with freeze_time(FROZEN_NOW) as frozen:
        dm.on_task_active("task_1", "project_1")
        frozen.tick(300)
        dm.on_task_done("task_1", "project_1", "First task")  # closes at +5min, no overlap yet

        frozen.tick(-240)  # rewind: task_2 "started" 1 min after task_1 originally started
        dm.on_task_active("task_2", "project_1")
        frozen.tick(600)  # task_2 finishes well after task_1's window ended

        result = dm.on_task_done("task_2", "project_1", "Second task")

    assert "not logged" in result
    state = _read_state(tmp_duration_file)
    assert len(state["entries"]) == 1  # only the first, non-overlapping task got logged
    assert state["entries"][0]["title"] == "First task"


def test_on_task_done_does_not_flag_non_overlapping_sequential_tasks(tmp_duration_file):
    with freeze_time(FROZEN_NOW) as frozen:
        dm.on_task_active("task_1", "project_1")
        frozen.tick(300)
        dm.on_task_done("task_1", "project_1", "First")

        dm.on_task_active("task_2", "project_1")
        frozen.tick(300)
        result = dm.on_task_done("task_2", "project_1", "Second")

    assert "not logged" not in result
    state = _read_state(tmp_duration_file)
    assert len(state["entries"]) == 2
