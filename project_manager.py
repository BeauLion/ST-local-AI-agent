"""
Project manager: projects + tasks, with short IDs (e.g. "TD-003"), notes,
priorities, and batch operations.

This used to live entirely inside the SillyTavern "Lightweight Project
Manager" extension (index.js), with state stored in the browser via
extensionSettings and tools registered client-side with SillyTavern's
registerFunctionTool API. Neither of those ever actually reached this
project's local backend: the agent server (main.py) always overwrites the
outgoing `tools` list with its own, so client-registered tools were
silently dropped.

This module ports that data model and logic to Python so it's driven the
same way every other tool in this project is: main.py exposes it to the
model as tools, and the SillyTavern extension becomes a thin UI that reads
and writes this server over HTTP instead of owning any state itself.

Storage is a single JSON file (projects.json) - consistent with memory.py's
approach, and plenty for personal-scale use (a handful of projects, at most
a few hundred tasks each).

Note: project archiving, which the JS version briefly shipped and then
disabled (v0.3.7.1), was left out entirely here rather than ported disabled.
Projects only ever have status active / paused / completed.
"""

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import (
    MAX_BATCH_OPERATIONS,
    MAX_PROJECT_CODE_LENGTH,
    MAX_PROJECT_NAME_LENGTH,
    MAX_TASK_NOTE_LENGTH,
    MAX_TASK_TITLE_LENGTH,
    MAX_TASKS_IN_CONTEXT,
    PROJECT_DATA_DIR,
    PROJECT_STATUSES,
    TASK_PRIORITIES,
    TASK_STATUSES,
)

DATA_DIR = Path(PROJECT_DATA_DIR)
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "projects.json"

# All reads/writes go through _load/_save under this lock. The agent server
# is single-process, so this is enough to stop two concurrent requests from
# corrupting the JSON file; it's not meant to scale beyond that.
_lock = threading.Lock()

DEFAULT_STATE = {"projects": {}, "focused_project_id": None}


class ProjectManagerError(Exception):
    """Raised for any invalid operation - caught by main.py's tool wrappers
    and turned into a plain error string the model/UI can see."""


# --------------------------- storage ---------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = json.loads(json.dumps(DEFAULT_STATE))
    state.setdefault("projects", {})
    state.setdefault("focused_project_id", None)
    if state["focused_project_id"] and state["focused_project_id"] not in state["projects"]:
        state["focused_project_id"] = None
    return state


def _save(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------- small helpers ---------------------------

def _normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_key(value) -> str:
    return _normalize_text(value).lower()


def _require_length(value, label, maximum) -> str:
    clean = _normalize_text(value)
    if not clean:
        raise ProjectManagerError(f"{label} is required.")
    if len(clean) > maximum:
        raise ProjectManagerError(f"{label} must be {maximum} characters or fewer.")
    return clean


def _derive_project_code(name: str) -> str:
    words = re.findall(r"[^\W\d_]+|\d+", name.upper(), flags=re.UNICODE)
    if not words:
        return "PRJ"
    if len(words) == 1:
        return words[0][:3]
    return "".join(w[0] for w in words[:4])[:MAX_PROJECT_CODE_LENGTH]


def _allocate_project_code(state: dict, name: str) -> str:
    used = {p["short_code"] for p in state["projects"].values()}
    base = _derive_project_code(name) or "PRJ"
    if base not in used:
        return base
    for suffix in range(2, 10000):
        suffix_str = str(suffix)
        candidate = f"{base[:max(1, MAX_PROJECT_CODE_LENGTH - len(suffix_str))]}{suffix_str}"
        if candidate not in used:
            return candidate
    raise ProjectManagerError("Could not allocate a unique project code.")


def _format_task_short_id(project: dict, number: int) -> str:
    return f"{project['short_code']}-{number:03d}"


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _recalculate_next_action(project: dict):
    open_tasks = [
        t for t in project["tasks"].values()
        if not t["archived"] and t["status"] not in ("done", "cancelled")
    ]
    open_tasks.sort(key=lambda t: (0 if t["status"] == "active" else 1, t.get("sort_order", 0)))
    project["next_action"] = open_tasks[0]["title"] if open_tasks else ""


# --------------------------- lookups ---------------------------

def get_projects(state: dict) -> list:
    projects = list(state["projects"].values())
    projects.sort(key=lambda p: (p["status"] != "active", p["name"].lower()))
    return projects


def get_focused_project(state: dict):
    fid = state.get("focused_project_id")
    return state["projects"].get(fid) if fid else None


def _resolve_unique_match(items, query, *, id_field="id", short_id_field=None, text_field=None, label="item"):
    raw = _normalize_text(query)
    if not raw:
        return None

    for item in items:
        if item[id_field] == raw:
            return item

    if short_id_field:
        short_key = raw.upper()
        matches = [i for i in items if i.get(short_id_field, "").upper() == short_key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ProjectManagerError(f"Multiple {label}s match short ID {short_key}.")

    key = _normalize_key(raw)
    exact = [i for i in items if _normalize_key(i.get(text_field, "")) == key]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ProjectManagerError(f"Multiple {label}s have the exact same name. Use the ID.")

    partial = [i for i in items if key in _normalize_key(i.get(text_field, ""))]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        options = ", ".join(f"\u201c{i[text_field]}\u201d" for i in partial[:5])
        raise ProjectManagerError(f"Ambiguous {label}: {options}. Use a more specific name or ID.")
    return None


def find_project(state: dict, query: str):
    """Resolve a project by ID, short code, exact/partial name, or the
    composite 'CODE — Name' label the UI/context text displays."""
    if not _normalize_text(query):
        return get_focused_project(state)

    raw = _normalize_text(query)
    projects = get_projects(state)

    parts = [p for p in re.split(r"\s+(?:\u2014|\u2013|-)\s+", raw) if _normalize_text(p)]
    if len(parts) >= 2:
        code = parts[0].upper()
        name = _normalize_key(" ".join(parts[1:]))
        for project in projects:
            if project["short_code"].upper() == code and _normalize_key(project["name"]) == name:
                return project
        code_only = [p for p in projects if p["short_code"].upper() == code]
        if len(code_only) == 1:
            return code_only[0]

    return _resolve_unique_match(
        projects, raw, short_id_field="short_code", text_field="name", label="project"
    )


def get_task(project, query: str):
    if not project:
        return None
    tasks = list(project["tasks"].values())
    if not _normalize_text(query):
        active = [t for t in tasks if t["status"] == "active"]
        if len(active) == 1:
            return active[0]
        if len(active) > 1:
            raise ProjectManagerError("Multiple tasks are active. Specify a task name or ID.")
        return None
    return _resolve_unique_match(tasks, query, short_id_field="short_id", text_field="title", label="task")


def resolve_project(state: dict, project_ref):
    """Like find_project, but raises if nothing usable is found - the
    shape every tool/endpoint needs before it can act."""
    project = find_project(state, project_ref) if project_ref else get_focused_project(state)
    if not project:
        raise ProjectManagerError("Project not found. Create or focus a project first.")
    return project


# --------------------------- mutations (projects) ---------------------------

def create_project(name: str) -> dict:
    clean_name = _require_length(name, "Project name", MAX_PROJECT_NAME_LENGTH)
    with _lock:
        state = _load()
        if any(p["name"].lower() == clean_name.lower() for p in state["projects"].values()):
            raise ProjectManagerError("A project with that name already exists.")

        project_id = _make_id("project")
        project = {
            "id": project_id,
            "short_code": _allocate_project_code(state, clean_name),
            "next_task_number": 1,
            "name": clean_name,
            "status": "active",
            "next_action": "",
            "created_at": _now(),
            "updated_at": _now(),
            "tasks": {},
        }
        state["projects"][project_id] = project
        state["focused_project_id"] = project_id
        _save(state)
        return project


def set_focused_project(project_id: str) -> dict:
    with _lock:
        state = _load()
        if project_id not in state["projects"]:
            raise ProjectManagerError("Project not found.")
        state["focused_project_id"] = project_id
        state["projects"][project_id]["updated_at"] = _now()
        _save(state)
        return state["projects"][project_id]


def set_project_status(project_id: str, status: str) -> dict:
    if status not in PROJECT_STATUSES:
        raise ProjectManagerError(f"Invalid project status: {status}")
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project:
            raise ProjectManagerError("Project not found.")
        project["status"] = status
        project["updated_at"] = _now()
        _save(state)
        return project


def rename_project(project_id: str, name: str) -> dict:
    clean_name = _require_length(name, "Project name", MAX_PROJECT_NAME_LENGTH)
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project:
            raise ProjectManagerError("Project not found.")
        project["name"] = clean_name
        project["updated_at"] = _now()
        _save(state)
        return project


def delete_project(project_id: str):
    with _lock:
        state = _load()
        if project_id not in state["projects"]:
            raise ProjectManagerError("Project not found.")
        del state["projects"][project_id]
        if state["focused_project_id"] == project_id:
            remaining = list(state["projects"].keys())
            state["focused_project_id"] = remaining[0] if remaining else None
        _save(state)


# --------------------------- mutations (tasks) ---------------------------

def create_task(project_id: str, title: str, *, priority="normal", notes="") -> dict:
    clean_title = _require_length(title, "Task title", MAX_TASK_TITLE_LENGTH)
    if priority not in TASK_PRIORITIES:
        priority = "normal"
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project:
            raise ProjectManagerError("Project not found.")

        duplicate = any(
            t["status"] not in ("done", "cancelled") and t["title"].lower() == clean_title.lower()
            for t in project["tasks"].values()
        )
        if duplicate:
            raise ProjectManagerError(f"An open task with that title already exists: {clean_title}")

        task_id = _make_id("task")
        number = project.get("next_task_number", 1)
        project["next_task_number"] = number + 1
        task = {
            "id": task_id,
            "short_id": _format_task_short_id(project, number),
            "title": clean_title,
            "status": "pending",
            "created_at": _now(),
            "updated_at": _now(),
            "completed_at": None,
            "archived": False,
            "archived_at": None,
            "priority": priority,
            "notes": str(notes or "")[:MAX_TASK_NOTE_LENGTH],
            "sort_order": len(project["tasks"]),
        }
        project["tasks"][task_id] = task
        project["updated_at"] = _now()
        _recalculate_next_action(project)
        _save(state)
        return task


def _apply_task_status(task: dict, status: str):
    task["status"] = status
    task["updated_at"] = _now()
    task["completed_at"] = _now() if status == "done" else None
    task["archived"] = status == "done"
    task["archived_at"] = _now() if status == "done" else None


def set_task_status(project_id: str, task_id: str, status: str) -> dict:
    if status not in TASK_STATUSES:
        raise ProjectManagerError(f"Invalid task status: {status}")
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project or task_id not in project["tasks"]:
            raise ProjectManagerError("Task not found in the selected project.")
        task = project["tasks"][task_id]
        if task["status"] == status:
            return task

        _apply_task_status(task, status)
        if status == "active":
            for other in project["tasks"].values():
                if other["id"] != task_id and other["status"] == "active":
                    other["status"] = "pending"
                    other["updated_at"] = _now()

        project["updated_at"] = _now()
        _recalculate_next_action(project)
        _save(state)
        return task


def update_task_details(project_id: str, task_id: str, *, title=None, priority=None, notes=None) -> dict:
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project or task_id not in project["tasks"]:
            raise ProjectManagerError("Task not found.")
        task = project["tasks"][task_id]
        if title is not None:
            task["title"] = _require_length(title, "Task title", MAX_TASK_TITLE_LENGTH)
        if priority is not None:
            task["priority"] = priority if priority in TASK_PRIORITIES else "normal"
        if notes is not None:
            task["notes"] = str(notes or "")[:MAX_TASK_NOTE_LENGTH]
        task["updated_at"] = _now()
        project["updated_at"] = _now()
        _recalculate_next_action(project)
        _save(state)
        return task


def update_task_notes(project_id: str, task_id: str, mode: str, text: str = "") -> dict:
    if mode not in ("replace", "append", "clear"):
        raise ProjectManagerError(f"Invalid note mode: {mode}")
    if mode != "clear" and not _normalize_text(text):
        raise ProjectManagerError("Note text is required for replace or append.")
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project or task_id not in project["tasks"]:
            raise ProjectManagerError("Task not found.")
        task = project["tasks"][task_id]
        current = task.get("notes", "")
        if mode == "clear":
            next_notes = ""
        elif mode == "append":
            next_notes = "\n".join(filter(None, [current, _normalize_text(text)]))
        else:
            next_notes = str(text or "")
        task["notes"] = next_notes[:MAX_TASK_NOTE_LENGTH]
        task["updated_at"] = _now()
        project["updated_at"] = _now()
        _save(state)
        return task


def delete_task(project_id: str, task_id: str):
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project or task_id not in project["tasks"]:
            raise ProjectManagerError("Task not found.")
        del project["tasks"][task_id]
        project["updated_at"] = _now()
        _recalculate_next_action(project)
        _save(state)


def reorder_tasks(project_id: str, ordered_task_ids: list):
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project:
            raise ProjectManagerError("Project not found.")
        for index, task_id in enumerate(ordered_task_ids):
            if task_id in project["tasks"]:
                project["tasks"][task_id]["sort_order"] = index
        project["updated_at"] = _now()
        _recalculate_next_action(project)
        _save(state)


# --------------------------- batch operations ---------------------------

def _normalize_batch_operation(operation: dict, project: dict) -> dict:
    if not isinstance(operation, dict):
        raise ProjectManagerError("Every batch operation must be an object.")
    op_type = _normalize_text(operation.get("type")).lower()

    if op_type == "create_task":
        return {
            "type": op_type,
            "title": _require_length(operation.get("title"), "Task title", MAX_TASK_TITLE_LENGTH),
            "priority": operation.get("priority") if operation.get("priority") in TASK_PRIORITIES else "normal",
            "notes": str(operation.get("notes") or "")[:MAX_TASK_NOTE_LENGTH],
        }

    if op_type == "update_task_status":
        status = operation.get("status")
        if status not in TASK_STATUSES:
            raise ProjectManagerError(f"Invalid task status: {status}")
        task_ref = operation.get("task") or operation.get("taskId") or operation.get("task_id")
        task = get_task(project, task_ref)
        if not task:
            raise ProjectManagerError(f"Task not found or ambiguous: {task_ref}")
        return {"type": op_type, "task_id": task["id"], "status": status}

    if op_type == "update_task_notes":
        mode = operation.get("mode")
        if mode not in ("replace", "append", "clear"):
            raise ProjectManagerError(f"Invalid note mode: {mode}")
        task_ref = operation.get("task") or operation.get("taskId") or operation.get("task_id")
        task = get_task(project, task_ref)
        if not task:
            raise ProjectManagerError(f"Task not found or ambiguous: {task_ref}")
        text = operation.get("text")
        if mode != "clear" and not _normalize_text(text):
            raise ProjectManagerError("Note text is required for replace or append.")
        return {"type": op_type, "task_id": task["id"], "mode": mode, "text": str(text or "")[:MAX_TASK_NOTE_LENGTH]}

    raise ProjectManagerError(f"Unsupported batch operation: {op_type or 'missing type'}")


def _validate_batch_operations(project: dict, operations: list) -> list:
    if not isinstance(operations, list) or len(operations) < 1:
        raise ProjectManagerError("Batch operations are required.")
    if len(operations) > MAX_BATCH_OPERATIONS:
        raise ProjectManagerError(f"A batch may contain at most {MAX_BATCH_OPERATIONS} operations.")

    normalized = [_normalize_batch_operation(op, project) for op in operations]

    actionable = []
    for op in normalized:
        if op["type"] == "update_task_status":
            task = project["tasks"].get(op["task_id"])
            if task and task["status"] != op["status"]:
                actionable.append(op)
        elif op["type"] == "update_task_notes":
            task = project["tasks"].get(op["task_id"])
            if not task:
                actionable.append(op)
                continue
            current = task.get("notes", "")
            if op["mode"] == "clear":
                next_notes = ""
            elif op["mode"] == "append":
                next_notes = "\n".join(filter(None, [current, _normalize_text(op["text"])]))
            else:
                next_notes = op["text"]
            if next_notes != current:
                actionable.append(op)
        else:
            actionable.append(op)

    existing_titles = {
        _normalize_key(t["title"]) for t in project["tasks"].values()
        if t["status"] not in ("done", "cancelled")
    }
    batch_titles = set()
    for op in actionable:
        if op["type"] == "create_task":
            key = _normalize_key(op["title"])
            if key in existing_titles:
                raise ProjectManagerError(f"An open task already exists: {op['title']}")
            if key in batch_titles:
                raise ProjectManagerError(f"Duplicate task in batch: {op['title']}")
            batch_titles.add(key)

    return actionable


def _describe_batch_operation(operation: dict, project: dict) -> str:
    if operation["type"] == "create_task":
        return f"Add \u201c{operation['title']}\u201d"
    task = project["tasks"].get(operation["task_id"])
    title = task["title"] if task else operation["task_id"]
    if operation["type"] == "update_task_status":
        return f"Set \u201c{title}\u201d to {operation['status']}"
    if operation["type"] == "update_task_notes":
        action = {"append": "Append notes to", "clear": "Clear notes for"}.get(operation["mode"], "Replace notes for")
        return f"{action} \u201c{title}\u201d"
    return "Unknown operation"


def _apply_operation_to_project(project: dict, operation: dict):
    if operation["type"] == "create_task":
        task_id = _make_id("task")
        number = project.get("next_task_number", 1)
        project["next_task_number"] = number + 1
        project["tasks"][task_id] = {
            "id": task_id,
            "short_id": _format_task_short_id(project, number),
            "title": operation["title"],
            "status": "pending",
            "created_at": _now(), "updated_at": _now(), "completed_at": None,
            "archived": False, "archived_at": None,
            "priority": operation.get("priority", "normal"),
            "notes": operation.get("notes", ""),
            "sort_order": len(project["tasks"]),
        }
        return

    task = project["tasks"].get(operation["task_id"])
    if not task:
        raise ProjectManagerError("A batch target task no longer exists.")

    if operation["type"] == "update_task_status":
        if task["status"] == operation["status"]:
            raise ProjectManagerError(f"\u201c{task['title']}\u201d is already {operation['status']}.")
        _apply_task_status(task, operation["status"])
        if operation["status"] == "active":
            for other in project["tasks"].values():
                if other["id"] != task["id"] and other["status"] == "active":
                    other["status"] = "pending"
                    other["updated_at"] = _now()
        return

    if operation["type"] == "update_task_notes":
        current = task.get("notes", "")
        if operation["mode"] == "clear":
            next_notes = ""
        elif operation["mode"] == "append":
            next_notes = "\n".join(filter(None, [current, _normalize_text(operation["text"])]))
        else:
            next_notes = operation["text"]
        if next_notes == current:
            raise ProjectManagerError(f"Notes for \u201c{task['title']}\u201d already match the requested result.")
        task["notes"] = next_notes[:MAX_TASK_NOTE_LENGTH]
        task["updated_at"] = _now()
        return

    raise ProjectManagerError("Unsupported batch operation.")


def batch_update(project_id: str, operations: list) -> list:
    """Validates and applies a batch atomically: nothing is written to disk
    unless every operation in the (already de-duplicated) batch succeeds."""
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project:
            raise ProjectManagerError("Project not found.")
        actionable = _validate_batch_operations(project, operations)
        if not actionable:
            return []

        descriptions = [_describe_batch_operation(op, project) for op in actionable]
        for op in actionable:
            _apply_operation_to_project(project, op)
        project["updated_at"] = _now()
        _recalculate_next_action(project)
        _save(state)
        return descriptions


def set_all_tasks_status(project_id: str, status: str) -> list:
    """Sets every non-archived task to `status` in one batch, without ever
    touching the project's own status - even if that empties the open list."""
    if status not in TASK_STATUSES:
        raise ProjectManagerError(f"Invalid task status: {status}")
    with _lock:
        state = _load()
        project = state["projects"].get(project_id)
        if not project:
            raise ProjectManagerError("Project not found.")
        operations = [
            {"type": "update_task_status", "task_id": t["id"], "status": status}
            for t in project["tasks"].values()
            if not t["archived"] and t["status"] != status
        ]
        if not operations:
            return []
        descriptions = [_describe_batch_operation(op, project) for op in operations]
        for op in operations:
            _apply_operation_to_project(project, op)
        project["updated_at"] = _now()
        _recalculate_next_action(project)
        _save(state)
        return descriptions


# --------------------------- read-only views ---------------------------

def project_overview(state: dict) -> dict:
    """What the project_manager_get_overview tool hands back to the model."""
    return {
        "focused_project_id": state.get("focused_project_id"),
        "projects": [
            {
                "id": p["id"],
                "short_code": p["short_code"],
                "name": p["name"],
                "status": p["status"],
                "next_action": p["next_action"] or None,
                "tasks": [
                    {"id": t["id"], "short_id": t["short_id"], "title": t["title"], "status": t["status"]}
                    for t in p["tasks"].values()
                ],
            }
            for p in get_projects(state)
        ],
    }


def build_context_text(state: dict) -> str:
    """Compact, authoritative summary of the focused project for injection
    into the model's system prompt - the server-side equivalent of the old
    extension's buildContextText()/setExtensionPrompt()."""
    project = get_focused_project(state)
    if not project:
        return ""

    tasks = [t for t in project["tasks"].values() if not t["archived"]]
    order = {"active": 0, "blocked": 1, "pending": 2, "cancelled": 3}
    tasks.sort(key=lambda t: (order.get(t["status"], 9), t.get("sort_order", 0)))
    selected = tasks[:MAX_TASKS_IN_CONTEXT]

    symbols = {"pending": "[ ]", "active": "[ACTIVE]", "blocked": "[BLOCKED]", "cancelled": "[CANCELLED]"}
    lines = [
        "[PERSISTENT PROJECT STATE]",
        f"Focused project: {project['short_code']} \u2014 {project['name']}",
        f"Project status: {project['status']}",
        f"Next action: {project['next_action'] or 'None set'}",
    ]
    if selected:
        lines.append("Tasks:")
        lines.extend(f"- {symbols[t['status']]} {t['short_id']} {t['title']}" for t in selected)
    else:
        lines.append("Tasks: none")

    lines.append(
        "Treat this state as authoritative. Use project-manager tools only for concrete state "
        "changes. Avoid duplicates. When one user message contains multiple concrete changes for "
        "the same project, use project_manager_batch_update exactly once instead of separate tool "
        "calls. For requests to set every task in one project to the same status, use "
        "project_manager_set_all_tasks_status exactly once; do not enumerate tasks first. "
        "Completing all tasks does not complete the project - only change project status when the "
        "user explicitly requests that project-level change. Keep planning responses concise and "
        "prioritize one next action."
    )
    return "\n".join(lines)
