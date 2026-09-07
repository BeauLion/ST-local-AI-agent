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


def test_overview_task_includes_deadline_duration_effort_when(project):
    pm.create_task(
        project["id"], "Loaded task", deadline="2026-09-10",
        duration="45m", effort="medium", when="afternoon weekend",
    )
    state = pm._load()
    overview = pm.project_overview(state)
    task = overview["projects"][0]["tasks"][0]
    assert task["deadline"] == "2026-09-10"
    assert task["duration_minutes"] == 45.0
    assert task["effort"] == "medium"
    assert task["when"] == "afternoon weekend"


def test_overview_task_unset_fields_are_none(project):
    pm.create_task(project["id"], "Bare task")
    state = pm._load()
    overview = pm.project_overview(state)
    task = overview["projects"][0]["tasks"][0]
    assert task["deadline"] is None
    assert task["duration_minutes"] is None
    assert task["effort"] is None
    assert task["when"] is None


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
# build_context_text - deadline/duration/effort/when shown inline
# ---------------------------------------------------------------------------
# Scoped to the task's own bullet line, not "in text" anywhere - the
# trailing instructions paragraph below also contains an example string
# like "~45m · medium effort", so a bare substring check would pass
# even if the per-task rendering were broken.

def _bullet_line_for(text: str, title: str) -> str:
    lines = [l for l in text.split("\n") if l.startswith("- ") and title in l]
    assert len(lines) == 1
    return lines[0]


def test_context_text_shows_inline_properties_for_a_task_with_all_fields_set(project, fake_duration):
    pm.create_task(
        project["id"], "Loaded task", deadline="2026-09-10",
        duration="45m", effort="medium", when="afternoon weekend",
    )
    state = pm._load()
    text = pm.build_context_text(state)
    line = _bullet_line_for(text, "Loaded task")
    assert "due 2026-09-10" in line
    assert "~45m" in line
    assert "medium effort" in line
    assert "afternoon weekend" in line


def test_context_text_shows_only_the_fields_that_are_set(project):
    pm.create_task(project["id"], "Partial task", effort="high")
    state = pm._load()
    text = pm.build_context_text(state)
    line = _bullet_line_for(text, "Partial task")
    assert "high effort" in line
    assert "due" not in line
    assert "~" not in line


def test_context_text_shows_no_bracket_suffix_for_a_task_with_nothing_set(project):
    pm.create_task(project["id"], "Plain task", notes="Just prose, no properties")
    state = pm._load()
    text = pm.build_context_text(state)
    line = _bullet_line_for(text, "Plain task")
    assert line.rstrip().endswith("Plain task")  # no trailing "[...]" suffix
