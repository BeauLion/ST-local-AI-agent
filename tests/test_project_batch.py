"""
Tests for project_manager.py's batch operation functions: batch_update
(normalize -> de-duplicate against current state -> apply atomically)
and set_all_tasks_status.

Key behaviors under test:
  - Operations are validated/normalized BEFORE anything is applied
    (a bad operation anywhere in the batch means nothing in the batch
    is applied - true atomicity, not best-effort).
  - _validate_batch_operations silently DROPS no-op operations (e.g.
    setting a task to the status it's already at) rather than raising -
    only genuinely actionable operations reach _apply_operation_to_project.
  - Duplicate-title detection for create_task operations happens both
    against EXISTING open tasks and against other create_task operations
    in the SAME batch.
"""
import pytest

import project_manager as pm
from project_manager import ProjectManagerError


@pytest.fixture
def project(tmp_project_file, fake_duration):
    return pm.create_project("Thesis")


# ---------------------------------------------------------------------------
# batch_update - validation happens before anything is applied
# ---------------------------------------------------------------------------

def test_batch_requires_a_non_empty_list(project):
    with pytest.raises(ProjectManagerError, match="Batch operations are required"):
        pm.batch_update(project["id"], [])


def test_batch_rejects_non_list(project):
    with pytest.raises(ProjectManagerError, match="Batch operations are required"):
        pm.batch_update(project["id"], "not a list")


def test_batch_enforces_max_size(project):
    import config
    too_many = [{"type": "create_task", "title": f"Task {i}"} for i in range(config.MAX_BATCH_OPERATIONS + 1)]
    with pytest.raises(ProjectManagerError, match="at most .* operations"):
        pm.batch_update(project["id"], too_many)


def test_batch_raises_for_unknown_project():
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.batch_update("nonexistent", [{"type": "create_task", "title": "A task"}])


def test_batch_rejects_non_dict_operation(project):
    with pytest.raises(ProjectManagerError, match="must be an object"):
        pm.batch_update(project["id"], ["not a dict"])


def test_batch_rejects_unsupported_operation_type(project):
    with pytest.raises(ProjectManagerError, match="Unsupported batch operation"):
        pm.batch_update(project["id"], [{"type": "delete_everything"}])


def test_one_bad_operation_blocks_the_entire_batch(project):
    """A batch containing one valid create_task and one invalid operation
    (missing title) must apply NEITHER - true atomicity, not partial
    application."""
    ops = [
        {"type": "create_task", "title": "Good task"},
        {"type": "create_task", "title": ""},  # invalid - empty title
    ]
    with pytest.raises(ProjectManagerError):
        pm.batch_update(project["id"], ops)

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"] == {}  # nothing was created


# ---------------------------------------------------------------------------
# batch_update - create_task operations
# ---------------------------------------------------------------------------

def test_batch_creates_multiple_tasks(project):
    ops = [{"type": "create_task", "title": "Task A"}, {"type": "create_task", "title": "Task B"}]
    descriptions, flags = pm.batch_update(project["id"], ops)

    state = pm._load()
    titles = {t["title"] for t in state["projects"][project["id"]]["tasks"].values()}
    assert titles == {"Task A", "Task B"}
    assert len(descriptions) == 2


def test_batch_create_task_rejects_duplicate_against_existing_open_task(project):
    pm.create_task(project["id"], "Existing task")
    ops = [{"type": "create_task", "title": "Existing task"}]

    with pytest.raises(ProjectManagerError, match="already exists"):
        pm.batch_update(project["id"], ops)


def test_batch_create_task_rejects_duplicate_within_the_same_batch(project):
    ops = [
        {"type": "create_task", "title": "Same title"},
        {"type": "create_task", "title": "same title"},
    ]
    with pytest.raises(ProjectManagerError, match="Duplicate task in batch"):
        pm.batch_update(project["id"], ops)


# ---------------------------------------------------------------------------
# batch_update - update_task_status operations
# ---------------------------------------------------------------------------

def test_batch_updates_task_status(project, fake_duration):
    task = pm.create_task(project["id"], "A task")
    ops = [{"type": "update_task_status", "task": task["id"], "status": "active"}]

    pm.batch_update(project["id"], ops)

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["status"] == "active"
    assert ("active", task["id"], project["id"]) in fake_duration.calls


def test_batch_update_task_status_no_op_is_silently_dropped_not_raised(project):
    """Setting a task to the status it's already at is a no-op that gets
    filtered out during validation, not an error - and if it's the ONLY
    operation in the batch, the whole batch becomes a no-op returning
    empty results rather than raising."""
    task = pm.create_task(project["id"], "A task")  # starts "pending"
    ops = [{"type": "update_task_status", "task": task["id"], "status": "pending"}]

    descriptions, flags = pm.batch_update(project["id"], ops)

    assert descriptions == []
    assert flags == []


def test_batch_update_task_status_resolves_task_by_title(project):
    task = pm.create_task(project["id"], "A task")
    ops = [{"type": "update_task_status", "task": "A task", "status": "active"}]

    pm.batch_update(project["id"], ops)

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["status"] == "active"


def test_batch_update_task_status_unknown_task_raises(project):
    ops = [{"type": "update_task_status", "task": "nonexistent task", "status": "active"}]
    with pytest.raises(ProjectManagerError, match="Task not found or ambiguous"):
        pm.batch_update(project["id"], ops)


def test_batch_update_task_status_invalid_status_raises(project):
    task = pm.create_task(project["id"], "A task")
    ops = [{"type": "update_task_status", "task": task["id"], "status": "not_a_real_status"}]
    with pytest.raises(ProjectManagerError, match="Invalid task status"):
        pm.batch_update(project["id"], ops)


def test_batch_activating_a_task_deactivates_another_active_task(project, fake_duration):
    t1 = pm.create_task(project["id"], "First")
    t2 = pm.create_task(project["id"], "Second")
    pm.set_task_status(project["id"], t1["id"], "active")
    fake_duration.calls.clear()

    ops = [{"type": "update_task_status", "task": t2["id"], "status": "active"}]
    pm.batch_update(project["id"], ops)

    state = pm._load()
    tasks = state["projects"][project["id"]]["tasks"]
    assert tasks[t1["id"]]["status"] == "pending"
    assert tasks[t2["id"]]["status"] == "active"


# ---------------------------------------------------------------------------
# batch_update - update_task_notes operations
# ---------------------------------------------------------------------------

def test_batch_updates_task_notes(project):
    task = pm.create_task(project["id"], "A task", notes="Old notes")
    ops = [{"type": "update_task_notes", "task": task["id"], "mode": "replace", "text": "New notes"}]

    pm.batch_update(project["id"], ops)

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["notes"] == "New notes"


def test_batch_update_task_notes_no_op_is_silently_dropped(project):
    task = pm.create_task(project["id"], "A task", notes="Same notes")
    ops = [{"type": "update_task_notes", "task": task["id"], "mode": "replace", "text": "Same notes"}]

    descriptions, flags = pm.batch_update(project["id"], ops)
    assert descriptions == []


def test_batch_update_task_notes_invalid_mode_raises(project):
    task = pm.create_task(project["id"], "A task")
    ops = [{"type": "update_task_notes", "task": task["id"], "mode": "not_a_real_mode", "text": "x"}]
    with pytest.raises(ProjectManagerError, match="Invalid note mode"):
        pm.batch_update(project["id"], ops)


def test_batch_update_task_notes_requires_text_for_non_clear_modes(project):
    task = pm.create_task(project["id"], "A task")
    ops = [{"type": "update_task_notes", "task": task["id"], "mode": "append", "text": ""}]
    with pytest.raises(ProjectManagerError, match="Note text is required"):
        pm.batch_update(project["id"], ops)


def test_batch_update_task_notes_clear_mode_needs_no_text(project):
    task = pm.create_task(project["id"], "A task", notes="Old")
    ops = [{"type": "update_task_notes", "task": task["id"], "mode": "clear"}]

    pm.batch_update(project["id"], ops)

    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["notes"] == ""


# ---------------------------------------------------------------------------
# batch_update - project-level side effects
# ---------------------------------------------------------------------------

def test_batch_recalculates_next_action(project):
    ops = [{"type": "create_task", "title": "New task"}]
    pm.batch_update(project["id"], ops)

    state = pm._load()
    assert state["projects"][project["id"]]["next_action"] == "New task"


def test_batch_all_operations_being_no_ops_returns_empty_without_touching_project_updated_at(project):
    task = pm.create_task(project["id"], "A task")
    state_before = pm._load()
    updated_at_before = state_before["projects"][project["id"]]["updated_at"]

    ops = [{"type": "update_task_status", "task": task["id"], "status": "pending"}]  # no-op
    pm.batch_update(project["id"], ops)

    state_after = pm._load()
    assert state_after["projects"][project["id"]]["updated_at"] == updated_at_before


# ---------------------------------------------------------------------------
# set_all_tasks_status
# ---------------------------------------------------------------------------

def test_set_all_tasks_status_updates_every_non_archived_task(project):
    t1 = pm.create_task(project["id"], "First")
    t2 = pm.create_task(project["id"], "Second")

    pm.set_all_tasks_status(project["id"], "cancelled")

    state = pm._load()
    tasks = state["projects"][project["id"]]["tasks"]
    assert tasks[t1["id"]]["status"] == "cancelled"
    assert tasks[t2["id"]]["status"] == "cancelled"


def test_set_all_tasks_status_does_not_touch_project_status(project):
    pm.create_task(project["id"], "A task")
    pm.set_all_tasks_status(project["id"], "done")

    state = pm._load()
    assert state["projects"][project["id"]]["status"] == "active"  # project itself untouched


def test_set_all_tasks_status_skips_already_archived_tasks(project):
    task = pm.create_task(project["id"], "A task")
    pm.set_task_status(project["id"], task["id"], "done")  # archived=True now

    descriptions, flags = pm.set_all_tasks_status(project["id"], "cancelled")

    assert descriptions == []  # nothing to do - the only task is archived
    state = pm._load()
    assert state["projects"][project["id"]]["tasks"][task["id"]]["status"] == "done"  # untouched


def test_set_all_tasks_status_skips_tasks_already_at_that_status(project):
    t1 = pm.create_task(project["id"], "First")
    t2 = pm.create_task(project["id"], "Second")
    pm.set_task_status(project["id"], t2["id"], "blocked")

    descriptions, flags = pm.set_all_tasks_status(project["id"], "blocked")

    assert len(descriptions) == 1  # only t1 needed changing


def test_set_all_tasks_status_rejects_invalid_status(project):
    with pytest.raises(ProjectManagerError, match="Invalid task status"):
        pm.set_all_tasks_status(project["id"], "not_a_real_status")


def test_set_all_tasks_status_raises_for_unknown_project():
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.set_all_tasks_status("nonexistent", "done")


def test_set_all_tasks_status_returns_empty_when_there_are_no_tasks(project):
    descriptions, flags = pm.set_all_tasks_status(project["id"], "done")
    assert descriptions == []
    assert flags == []
