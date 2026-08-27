"""
Tests for project_manager.py's project-level mutation functions:
create_project, set_focused_project, set_project_status, rename_project,
delete_project.
"""
import pytest

import project_manager as pm
from project_manager import ProjectManagerError


# ---------------------------------------------------------------------------
# create_project
# ---------------------------------------------------------------------------

def test_create_project_requires_a_name(tmp_project_file, fake_duration):
    with pytest.raises(ProjectManagerError, match="Project name is required"):
        pm.create_project("")


def test_create_project_returns_a_fully_shaped_project(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")

    assert project["name"] == "Thesis"
    assert project["short_code"] == "THE"
    assert project["status"] == "active"
    assert project["next_action"] == ""
    assert project["tasks"] == {}
    assert project["next_task_number"] == 1
    assert project["id"].startswith("project_")


def test_create_project_persists_to_disk(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    state = pm._load()
    assert project["id"] in state["projects"]


def test_create_project_sets_it_as_focused(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    state = pm._load()
    assert state["focused_project_id"] == project["id"]


def test_create_project_rejects_duplicate_name_case_insensitively(tmp_project_file, fake_duration):
    pm.create_project("Thesis")
    with pytest.raises(ProjectManagerError, match="already exists"):
        pm.create_project("thesis")


def test_create_second_project_becomes_the_new_focus(tmp_project_file, fake_duration):
    pm.create_project("Thesis")
    second = pm.create_project("Side Project")
    state = pm._load()
    assert state["focused_project_id"] == second["id"]


# ---------------------------------------------------------------------------
# set_focused_project
# ---------------------------------------------------------------------------

def test_set_focused_project_updates_state(tmp_project_file, fake_duration):
    p1 = pm.create_project("Thesis")
    p2 = pm.create_project("Side Project")

    pm.set_focused_project(p1["id"])

    state = pm._load()
    assert state["focused_project_id"] == p1["id"]


def test_set_focused_project_raises_for_unknown_id(tmp_project_file, fake_duration):
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.set_focused_project("nonexistent")


# ---------------------------------------------------------------------------
# set_project_status
# ---------------------------------------------------------------------------

def test_set_project_status_updates_status(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    pm.set_project_status(project["id"], "paused")

    state = pm._load()
    assert state["projects"][project["id"]]["status"] == "paused"


def test_set_project_status_rejects_invalid_status(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    with pytest.raises(ProjectManagerError, match="Invalid project status"):
        pm.set_project_status(project["id"], "not_a_real_status")


def test_set_project_status_raises_for_unknown_project(tmp_project_file, fake_duration):
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.set_project_status("nonexistent", "paused")


# ---------------------------------------------------------------------------
# rename_project
# ---------------------------------------------------------------------------

def test_rename_project_updates_name(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    pm.rename_project(project["id"], "Thesis Final")

    state = pm._load()
    assert state["projects"][project["id"]]["name"] == "Thesis Final"


def test_rename_project_requires_a_name(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    with pytest.raises(ProjectManagerError, match="Project name is required"):
        pm.rename_project(project["id"], "")


def test_rename_project_raises_for_unknown_project(tmp_project_file, fake_duration):
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.rename_project("nonexistent", "New Name")


def test_rename_project_does_not_change_its_short_code(tmp_project_file, fake_duration):
    """Renaming shouldn't retroactively re-derive/reassign the short
    code - short IDs already issued to tasks (e.g. THE-001) would become
    misleading if the code moved out from under them."""
    project = pm.create_project("Thesis")
    original_code = project["short_code"]
    pm.rename_project(project["id"], "Completely Different Name")

    state = pm._load()
    assert state["projects"][project["id"]]["short_code"] == original_code


# ---------------------------------------------------------------------------
# delete_project
# ---------------------------------------------------------------------------

def test_delete_project_removes_it_from_state(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    pm.delete_project(project["id"])

    state = pm._load()
    assert project["id"] not in state["projects"]


def test_delete_project_raises_for_unknown_project(tmp_project_file, fake_duration):
    with pytest.raises(ProjectManagerError, match="Project not found"):
        pm.delete_project("nonexistent")


def test_deleting_the_focused_project_reassigns_focus_to_a_remaining_project(tmp_project_file, fake_duration):
    p1 = pm.create_project("Thesis")
    p2 = pm.create_project("Side Project")
    pm.set_focused_project(p1["id"])

    pm.delete_project(p1["id"])

    state = pm._load()
    assert state["focused_project_id"] == p2["id"]


def test_deleting_the_last_remaining_project_clears_focus(tmp_project_file, fake_duration):
    project = pm.create_project("Thesis")
    pm.delete_project(project["id"])

    state = pm._load()
    assert state["focused_project_id"] is None


def test_deleting_a_non_focused_project_leaves_focus_unchanged(tmp_project_file, fake_duration):
    p1 = pm.create_project("Thesis")
    p2 = pm.create_project("Side Project")
    pm.set_focused_project(p1["id"])

    pm.delete_project(p2["id"])

    state = pm._load()
    assert state["focused_project_id"] == p1["id"]
