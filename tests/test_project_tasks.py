"""
Tests for project_manager.py's task-level mutation functions: create_task,
set_task_status (including the single-active-task enforcement and its
duration_manager hook calls), update_task_details, update_task_notes,
delete_task, reorder_tasks, and _recalculate_next_action.

duration_manager is faked throughout (fake_duration fixture) - these
tests check that project_manager calls on_task_active/inactive/done at
the right times with the right arguments, not that duration_manager's
own hooks behave correctly (covered in test_duration_hooks.py).
"""
import pytest

import project_manager as pm
from project_manager import ProjectManagerError


@pytest.fixture
def project(tmp_project_file, fake_duration):
    return pm.create_project("Thesis")


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------

def test_create_task_requires_a_title(project):
    with pytest.raises(ProjectManagerError, match="Task title is required"):
        pm.create_task(project["id"], "")


def test_create_task_returns_a_fully_shaped_task(project):
    task = pm.create_task(project["id"], "Write intro")

    assert task["title"] == "Write intro"
    assert task["status"] == "pending"
    assert task["priority"] == "normal"
    assert task["short_id"] == "THE-001"
    assert task["archived"] is False


def test_create_task_increments_short_id_number(project):
    t1 = pm.create_task(project["id"], "First task")
    t2 = pm.create_task(project["id"], "Second task")
    assert t1["short_id"] == "THE-001"
    assert t2["short_id"] == "THE-002"


def test_create_task_raises_for_unknown_project(tmp_project_file, fake_duration):
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.create_task("nonexistent", "A task")


def test_create_task_rejects_duplicate_open_title(project):
    pm.create_task(project["id"], "Write intro")
    with pytest.raises(ProjectManagerError, match="already exists"):
        pm.create_task(project["id"], "write intro")


def test_create_task_allows_reusing_title_of_a_done_task(project):
    """A duplicate title is only blocked among OPEN tasks (not done/
    cancelled) - re-adding a task that was already completed before
    shouldn't be permanently blocked."""
    t1 = pm.create_task(project["id"], "Write intro")
    pm.set_task_status(project["id"], t1["id"], "done")

    t2 = pm.create_task(project["id"], "Write intro")  # should succeed
    assert t2["id"] != t1["id"]


def test_create_task_invalid_priority_falls_back_to_normal(project):
    task = pm.create_task(project["id"], "Task", priority="not_a_real_priority")
    assert task["priority"] == "normal"


def test_create_task_updates_next_action(project):
    pm.create_task(project["id"], "Write intro")
    state = pm._load()
    assert state["projects"][project["id"]]["next_action"] == "Write intro"


def test_create_task_does_not_call_duration_manager(project, fake_duration):
    pm.create_task(project["id"], "Write intro")
    assert fake_duration.calls == []


# ---------------------------------------------------------------------------
# set_task_status - basic transitions
# ---------------------------------------------------------------------------

def test_set_task_status_updates_status(project):
    task = pm.create_task(project["id"], "Write intro")
    pm.set_task_status(project["id"], task["id"], "blocked")

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["status"] == "blocked"


def test_set_task_status_rejects_invalid_status(project):
    task = pm.create_task(project["id"], "Write intro")
    with pytest.raises(ProjectManagerError, match="Invalid task status"):
        pm.set_task_status(project["id"], task["id"], "not_a_real_status")


def test_set_task_status_raises_for_unknown_task(project):
    with pytest.raises(ProjectManagerError, match="Task not found"):
        pm.set_task_status(project["id"], "nonexistent", "done")


def test_set_task_status_to_same_status_is_a_no_op_returning_no_flag(project, fake_duration):
    task = pm.create_task(project["id"], "Write intro")
    result_task, flag = pm.set_task_status(project["id"], task["id"], "pending")

    assert flag is None
    assert fake_duration.calls == []


def test_set_task_status_to_done_sets_completed_and_archived_fields(project):
    task = pm.create_task(project["id"], "Write intro")
    pm.set_task_status(project["id"], task["id"], "done")

    state = pm._load()
    stored = state["projects"][project["id"]]["tasks"][task["id"]]
    assert stored["completed_at"] is not None
    assert stored["archived"] is True
    assert stored["archived_at"] is not None


def test_set_task_status_away_from_done_clears_completed_and_archived_fields(project):
    task = pm.create_task(project["id"], "Write intro")
    pm.set_task_status(project["id"], task["id"], "done")
    pm.set_task_status(project["id"], task["id"], "pending")

    state = pm._load()
    stored = state["projects"][project["id"]]["tasks"][task["id"]]
    assert stored["completed_at"] is None
    assert stored["archived"] is False
    assert stored["archived_at"] is None


# ---------------------------------------------------------------------------
# set_task_status - duration_manager hook interaction
# ---------------------------------------------------------------------------

def test_moving_to_active_calls_on_task_active(project, fake_duration):
    task = pm.create_task(project["id"], "Write intro")
    pm.set_task_status(project["id"], task["id"], "active")

    assert ("active", task["id"], project["id"]) in fake_duration.calls


def test_moving_away_from_active_calls_on_task_inactive(project, fake_duration):
    task = pm.create_task(project["id"], "Write intro")
    pm.set_task_status(project["id"], task["id"], "active")
    fake_duration.calls.clear()

    pm.set_task_status(project["id"], task["id"], "blocked")

    assert ("inactive", task["id"]) in fake_duration.calls


def test_moving_to_done_from_active_calls_inactive_then_done(project, fake_duration):
    """_apply_task_status calls on_task_inactive BEFORE on_task_done when
    the task was active - the "done" transition needs the active window
    closed for accounting first."""
    task = pm.create_task(project["id"], "Write intro")
    pm.set_task_status(project["id"], task["id"], "active")
    fake_duration.calls.clear()

    pm.set_task_status(project["id"], task["id"], "done")

    kinds = [c[0] for c in fake_duration.calls]
    assert kinds.index("inactive") < kinds.index("done")


def test_moving_to_done_from_pending_does_not_call_inactive_first(project, fake_duration):
    """A task that was never active shouldn't trigger an on_task_inactive
    call on its way to done - there's no active window to close."""
    task = pm.create_task(project["id"], "Write intro")
    pm.set_task_status(project["id"], task["id"], "done")

    kinds = [c[0] for c in fake_duration.calls]
    assert "inactive" not in kinds
    assert "done" in kinds


def test_set_task_status_returns_the_flag_from_on_task_done(project, fake_duration):
    fake_duration.set_done_return("~30 min logged for \u201cWrite intro\u201d (category: writing).")
    task = pm.create_task(project["id"], "Write intro")

    _, flag = pm.set_task_status(project["id"], task["id"], "done")

    assert flag == "~30 min logged for \u201cWrite intro\u201d (category: writing)."


def test_setting_a_second_task_active_deactivates_the_first(project, fake_duration):
    """Module enforces at most one active task per project - activating a
    second task must flip the first back to pending AND call
    on_task_inactive for it."""
    t1 = pm.create_task(project["id"], "First task")
    t2 = pm.create_task(project["id"], "Second task")
    pm.set_task_status(project["id"], t1["id"], "active")
    fake_duration.calls.clear()

    pm.set_task_status(project["id"], t2["id"], "active")

    state = pm._load()
    tasks = state["projects"][project["id"]]["tasks"]
    assert tasks[t1["id"]]["status"] == "pending"
    assert tasks[t2["id"]]["status"] == "active"
    assert ("inactive", t1["id"]) in fake_duration.calls
    assert ("active", t2["id"], project["id"]) in fake_duration.calls


# ---------------------------------------------------------------------------
# update_task_details
# ---------------------------------------------------------------------------

def test_update_task_details_updates_only_given_fields(project):
    task = pm.create_task(project["id"], "Write intro", priority="low", notes="Old notes")

    pm.update_task_details(project["id"], task["id"], title="Write introduction")

    state = pm._load()
    stored = state["projects"][project["id"]]["tasks"][task["id"]]
    assert stored["title"] == "Write introduction"
    assert stored["priority"] == "low"  # untouched
    assert stored["notes"] == "Old notes"  # untouched


def test_update_task_details_raises_for_unknown_task(project):
    with pytest.raises(ProjectManagerError, match="Task not found"):
        pm.update_task_details(project["id"], "nonexistent", title="New title")


def test_update_task_details_invalid_priority_falls_back_to_normal(project):
    task = pm.create_task(project["id"], "Write intro")
    pm.update_task_details(project["id"], task["id"], priority="not_a_real_priority")

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["priority"] == "normal"


# ---------------------------------------------------------------------------
# update_task_notes
# ---------------------------------------------------------------------------

def test_update_task_notes_replace_mode(project):
    task = pm.create_task(project["id"], "Write intro", notes="Old")
    pm.update_task_notes(project["id"], task["id"], "replace", "New notes")

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["notes"] == "New notes"


def test_update_task_notes_clear_mode(project):
    task = pm.create_task(project["id"], "Write intro", notes="Old")
    pm.update_task_notes(project["id"], task["id"], "clear")

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["notes"] == ""


def test_update_task_notes_rejects_invalid_mode(project):
    task = pm.create_task(project["id"], "Write intro")
    with pytest.raises(ProjectManagerError, match="Invalid note mode"):
        pm.update_task_notes(project["id"], task["id"], "not_a_real_mode")


def test_update_task_notes_requires_text_for_replace(project):
    task = pm.create_task(project["id"], "Write intro")
    with pytest.raises(ProjectManagerError, match="Note text is required"):
        pm.update_task_notes(project["id"], task["id"], "replace", "")


def test_update_task_notes_does_not_require_text_for_clear(project):
    task = pm.create_task(project["id"], "Write intro", notes="Old")
    pm.update_task_notes(project["id"], task["id"], "clear")  # no text arg - should not raise


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------

def test_delete_task_removes_it(project):
    task = pm.create_task(project["id"], "Write intro")
    pm.delete_task(project["id"], task["id"])

    state = pm._load()
    assert task["id"] not in state["projects"][project["id"]]["tasks"]


def test_delete_task_raises_for_unknown_task(project):
    with pytest.raises(ProjectManagerError, match="Task not found"):
        pm.delete_task(project["id"], "nonexistent")


# ---------------------------------------------------------------------------
# reorder_tasks
# ---------------------------------------------------------------------------

def test_reorder_tasks_updates_sort_order(project):
    t1 = pm.create_task(project["id"], "First")
    t2 = pm.create_task(project["id"], "Second")

    pm.reorder_tasks(project["id"], [t2["id"], t1["id"]])

    state = pm._load()
    tasks = state["projects"][project["id"]]["tasks"]
    assert tasks[t2["id"]]["sort_order"] == 0
    assert tasks[t1["id"]]["sort_order"] == 1


def test_reorder_tasks_ignores_unknown_ids_in_the_list(project):
    t1 = pm.create_task(project["id"], "First")
    pm.reorder_tasks(project["id"], ["nonexistent", t1["id"]])  # should not raise

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][t1["id"]]["sort_order"] == 1


# ---------------------------------------------------------------------------
# _recalculate_next_action
# ---------------------------------------------------------------------------

def test_next_action_prefers_the_active_task_over_pending(project):
    pm.create_task(project["id"], "Pending task")
    active_task = pm.create_task(project["id"], "Active task")
    pm.set_task_status(project["id"], active_task["id"], "active")

    state = pm._load()
    assert state["projects"][project["id"]]["next_action"] == "Active task"


def test_next_action_is_empty_when_all_tasks_are_done_or_cancelled(project):
    task = pm.create_task(project["id"], "Only task")
    pm.set_task_status(project["id"], task["id"], "done")

    state = pm._load()
    assert state["projects"][project["id"]]["next_action"] == ""


def test_next_action_falls_back_to_earliest_sort_order_pending_task(project):
    t1 = pm.create_task(project["id"], "First created")
    t2 = pm.create_task(project["id"], "Second created")
    pm.reorder_tasks(project["id"], [t2["id"], t1["id"]])  # t2 now sort_order 0

    state = pm._load()
    assert state["projects"][project["id"]]["next_action"] == "Second created"
