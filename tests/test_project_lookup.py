"""
Tests for project_manager.py's lookup functions: find_project (by ID,
short code, exact/partial name, or the composite "CODE — Name" label),
get_task (same resolution style, plus the "no query -> the single
active task" shortcut), and resolve_project (find_project but raises
instead of returning None).

_resolve_unique_match's tiered fallback (exact id -> short id -> exact
name -> partial name, each tier only consulted if the previous tier
found nothing) is exercised indirectly through find_project/get_task
here, since that's the only way either function is actually used.
"""
import pytest

import project_manager as pm
from project_manager import ProjectManagerError


def _make_state(*projects):
    """projects: list of (id, short_code, name) tuples. status defaults
    to "active" - find_project routes through get_projects(), which
    sorts on "status", so every project dict needs that field even
    though these tests aren't about status filtering."""
    return {
        "projects": {
            pid: {"id": pid, "short_code": code, "name": name, "status": "active", "tasks": {}}
            for pid, code, name in projects
        },
        "focused_project_id": None,
    }


# ---------------------------------------------------------------------------
# find_project
# ---------------------------------------------------------------------------

def test_find_project_empty_query_returns_focused_project():
    state = _make_state(("p1", "THE", "Thesis"))
    state["focused_project_id"] = "p1"
    assert pm.find_project(state, "") is state["projects"]["p1"]


def test_find_project_empty_query_with_no_focus_returns_none():
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.find_project(state, "") is None


def test_find_project_by_exact_id():
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.find_project(state, "p1")["id"] == "p1"


def test_find_project_by_short_code_case_insensitive():
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.find_project(state, "the")["id"] == "p1"


def test_find_project_by_exact_name_case_insensitive():
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.find_project(state, "thesis")["id"] == "p1"


def test_find_project_by_partial_name_when_unambiguous():
    state = _make_state(("p1", "THE", "Thesis Chapter Three"))
    assert pm.find_project(state, "chapter three")["id"] == "p1"


def test_find_project_partial_name_ambiguous_raises():
    state = _make_state(("p1", "TH1", "Thesis Draft"), ("p2", "TH2", "Thesis Notes"))
    with pytest.raises(ProjectManagerError, match="Ambiguous project"):
        pm.find_project(state, "thesis")


def test_find_project_multiple_exact_name_matches_raises():
    """Two projects can't normally share an exact name (create_project
    blocks it), but find_project's own matching logic should still be
    safe if state ever contains this - raises rather than picking one
    arbitrarily."""
    state = _make_state(("p1", "TH1", "Thesis"), ("p2", "TH2", "thesis"))
    with pytest.raises(ProjectManagerError, match="Multiple projects have the exact same name"):
        pm.find_project(state, "thesis")


def test_find_project_no_match_returns_none():
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.find_project(state, "Nonexistent Project") is None


def test_find_project_by_composite_code_and_name_label():
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.find_project(state, "THE \u2014 Thesis")["id"] == "p1"


def test_find_project_composite_label_falls_back_to_code_only_match():
    """A composite-looking query where the name half doesn't match should
    still resolve via the code half alone, if unambiguous."""
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.find_project(state, "THE \u2014 Wrong Name")["id"] == "p1"


def test_find_project_by_short_code_alone_with_hyphen_separator_style():
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.find_project(state, "THE - Thesis")["id"] == "p1"


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------

def _project_with_tasks(*tasks):
    """tasks: list of (id, short_id, title, status) tuples."""
    return {
        "id": "p1", "tasks": {
            tid: {"id": tid, "short_id": sid, "title": title, "status": status}
            for tid, sid, title, status in tasks
        },
    }


def test_get_task_with_none_project_returns_none():
    assert pm.get_task(None, "anything") is None


def test_get_task_empty_query_with_no_active_task_returns_none():
    project = _project_with_tasks(("t1", "THE-001", "Write intro", "pending"))
    assert pm.get_task(project, "") is None


def test_get_task_empty_query_returns_the_single_active_task():
    project = _project_with_tasks(
        ("t1", "THE-001", "Write intro", "active"),
        ("t2", "THE-002", "Write conclusion", "pending"),
    )
    assert pm.get_task(project, "")["id"] == "t1"


def test_get_task_empty_query_with_multiple_active_tasks_raises():
    project = _project_with_tasks(
        ("t1", "THE-001", "Write intro", "active"),
        ("t2", "THE-002", "Write conclusion", "active"),
    )
    with pytest.raises(ProjectManagerError, match="Multiple tasks are active"):
        pm.get_task(project, "")


def test_get_task_by_short_id_case_insensitive():
    project = _project_with_tasks(("t1", "THE-001", "Write intro", "pending"))
    assert pm.get_task(project, "the-001")["id"] == "t1"


def test_get_task_by_exact_title():
    project = _project_with_tasks(("t1", "THE-001", "Write intro", "pending"))
    assert pm.get_task(project, "Write intro")["id"] == "t1"


def test_get_task_by_partial_title_when_unambiguous():
    project = _project_with_tasks(("t1", "THE-001", "Write the introduction chapter", "pending"))
    assert pm.get_task(project, "introduction")["id"] == "t1"


def test_get_task_ambiguous_partial_title_raises():
    project = _project_with_tasks(
        ("t1", "THE-001", "Write introduction", "pending"),
        ("t2", "THE-002", "Revise introduction", "pending"),
    )
    with pytest.raises(ProjectManagerError, match="Ambiguous task"):
        pm.get_task(project, "introduction")


# ---------------------------------------------------------------------------
# resolve_project
# ---------------------------------------------------------------------------

def test_resolve_project_returns_found_project():
    state = _make_state(("p1", "THE", "Thesis"))
    assert pm.resolve_project(state, "Thesis")["id"] == "p1"


def test_resolve_project_with_no_ref_falls_back_to_focused():
    state = _make_state(("p1", "THE", "Thesis"))
    state["focused_project_id"] = "p1"
    assert pm.resolve_project(state, None)["id"] == "p1"


def test_resolve_project_raises_when_nothing_found():
    state = _make_state(("p1", "THE", "Thesis"))
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.resolve_project(state, "Nonexistent")


def test_resolve_project_raises_when_no_ref_and_nothing_focused():
    state = _make_state(("p1", "THE", "Thesis"))
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.resolve_project(state, None)
