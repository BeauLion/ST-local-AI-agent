"""
Tests for project_manager.py's read-only view builders: project_overview
(the project_manager_get_overview tool's payload) and build_context_text
(the system-prompt injection block for the focused project).
"""
import pytest

import project_manager as pm


@pytest.fixture
def project(tmp_project_file, fake_duration):
    return pm.create_project("Thesis")


# ---------------------------------------------------------------------------
# project_overview
# ---------------------------------------------------------------------------

def test_overview_includes_focused_project_id(project):
    state = pm._load()
    overview = pm.project_overview(state)
    assert overview["focused_project_id"] == project["id"]


def test_overview_lists_projects_with_their_tasks(project):
    pm.create_task(project["id"], "Write intro")
    state = pm._load()

    overview = pm.project_overview(state)

    assert len(overview["projects"]) == 1
    p = overview["projects"][0]
    assert p["short_code"] == "THE"
    assert p["name"] == "Thesis"
    assert len(p["tasks"]) == 1
    assert p["tasks"][0]["title"] == "Write intro"


def test_overview_with_no_projects_is_empty(tmp_project_file, fake_duration):
    state = pm._load()
    overview = pm.project_overview(state)
    assert overview["projects"] == []
    assert overview["focused_project_id"] is None


def test_overview_next_action_is_none_when_empty_string(project):
    state = pm._load()
    overview = pm.project_overview(state)
    assert overview["projects"][0]["next_action"] is None  # "" coerced to None, not shown as empty string


# ---------------------------------------------------------------------------
# build_context_text - no focused project
# ---------------------------------------------------------------------------

def test_context_text_is_empty_when_nothing_focused(tmp_project_file, fake_duration):
    state = pm._load()
    assert pm.build_context_text(state) == ""


# ---------------------------------------------------------------------------
# build_context_text - basic shape
# ---------------------------------------------------------------------------

def test_context_text_includes_project_header_fields(project):
    state = pm._load()
    text = pm.build_context_text(state)

    assert "[PERSISTENT PROJECT STATE]" in text
    assert "THE" in text and "Thesis" in text
    assert "Project status: active" in text
    assert "Next action: None set" in text  # empty next_action


def test_context_text_shows_next_action_when_set(project):
    pm.create_task(project["id"], "Write intro")
    state = pm._load()
    text = pm.build_context_text(state)
    assert "Next action: Write intro" in text


def test_context_text_lists_none_when_there_are_no_tasks(project):
    state = pm._load()
    text = pm.build_context_text(state)
    assert "Tasks: none" in text


# ---------------------------------------------------------------------------
# build_context_text - task ordering and status symbols
# ---------------------------------------------------------------------------

def test_context_text_orders_tasks_active_then_blocked_then_pending_then_cancelled(project):
    pending = pm.create_task(project["id"], "Pending task")
    cancelled = pm.create_task(project["id"], "Cancelled task")
    pm.set_task_status(project["id"], cancelled["id"], "cancelled")
    blocked = pm.create_task(project["id"], "Blocked task")
    pm.set_task_status(project["id"], blocked["id"], "blocked")
    active = pm.create_task(project["id"], "Active task")
    pm.set_task_status(project["id"], active["id"], "active")

    state = pm._load()
    text = pm.build_context_text(state)

    order = [text.index(t) for t in ["Active task", "Blocked task", "Pending task", "Cancelled task"]]
    assert order == sorted(order)


def test_context_text_uses_correct_status_symbols(project):
    task = pm.create_task(project["id"], "A task")
    pm.set_task_status(project["id"], task["id"], "active")
    state = pm._load()
    text = pm.build_context_text(state)
    assert "[ACTIVE]" in text


def test_context_text_excludes_archived_done_tasks(project):
    task = pm.create_task(project["id"], "A task")
    pm.set_task_status(project["id"], task["id"], "done")  # archived=True
    state = pm._load()
    text = pm.build_context_text(state)
    assert "A task" not in text
    assert "Tasks: none" in text


def test_context_text_truncates_to_max_tasks_in_context(project):
    import config
    for i in range(config.MAX_TASKS_IN_CONTEXT + 3):
        pm.create_task(project["id"], f"Task {i}")

    state = pm._load()
    text = pm.build_context_text(state)

    # Count task BULLET lines only ("- [ ] ...") - "Next action: Task 0"
    # also contains a task title and would otherwise inflate the count.
    bullet_lines = [l for l in text.split("\n") if l.startswith("- ")]
    assert len(bullet_lines) == config.MAX_TASKS_IN_CONTEXT


# ---------------------------------------------------------------------------
# build_context_text - note tags shown inline
# ---------------------------------------------------------------------------

def test_context_text_shows_inline_tags_for_a_tagged_task(project, fake_duration):
    pm.create_task(project["id"], "Tagged task", notes="dur: 45m\neffort: medium\nSome prose")
    state = pm._load()
    text = pm.build_context_text(state)
    assert "~45m" in text
    assert "medium effort" in text


def test_context_text_shows_no_bracket_suffix_for_untagged_task(project):
    pm.create_task(project["id"], "Plain task", notes="Just prose, no tags")
    state = pm._load()
    text = pm.build_context_text(state)
    # Only the task's own BULLET line matters here - "Next action: Plain
    # task" also legitimately contains the title and isn't what this
    # test is checking.
    bullet_lines = [l for l in text.split("\n") if l.startswith("- ") and "Plain task" in l]
    assert len(bullet_lines) == 1
    assert bullet_lines[0].rstrip().endswith("Plain task")  # no trailing "[...]" tag suffix
