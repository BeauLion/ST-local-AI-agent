"""
Phase 2: Tool-calling loop.

The agent server now:
  1. Injects a list of available tools into every request sent to llama-server
  2. If the model asks to call a tool, executes it in Python and feeds the
     result back to the model
  3. Loops until the model gives a final answer (or we hit a safety limit)
  4. Returns that final answer to SillyTavern, in whatever format
     (streaming or not) SillyTavern originally asked for

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8100 --reload
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

import docker

import httpx
from ddgs import DDGS
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

import memory
import project_manager
from project_manager import ProjectManagerError
from config import (
    CORS_ALLOWED_ORIGINS,
    DOCKER_CPU_COUNT,
    DOCKER_IMAGE,
    DOCKER_MEM_LIMIT,
    DOCKER_NETWORK_DISABLED,
    DOCKER_TIMEOUT_SECONDS,
    LLAMA_SERVER_URL,
    MAX_TOOL_ITERATIONS,
    SAFE_FILES_DIR,
    WRITE_FILE_ALLOWED_EXTENSIONS,
    WRITE_FILE_MAX_CHARS,
)

app = FastAPI()

# Lets the SillyTavern extension's browser-side fetch() calls (a different
# origin/port than this server) reach the /projects endpoints below.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The ONLY folder the agent is allowed to read files from.
SAFE_FILES_DIR = Path(SAFE_FILES_DIR).resolve()
SAFE_FILES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema) + the Python functions
# that actually execute them. Add new tools here in later phases.
# ---------------------------------------------------------------------------

def get_current_time(args: dict) -> str:
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


def calculate(args: dict) -> str:
    expression = args.get("expression", "")
    if not expression:
        return "Error: no expression provided."

    # Only allow digits, arithmetic operators, parentheses, decimals, and
    # whitespace - blocks any attempt to run arbitrary Python via eval.
    allowed = set("0123456789.+-*/() \t")
    if not set(expression) <= allowed:
        return "Error: expression contains characters that aren't allowed (only numbers and + - * / ( ) are permitted)."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def web_search(args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "Error: no query provided."
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"- {r['title']}\n  {r['href']}\n  {r['body']}")
    return "\n".join(lines)


def list_files(args: dict) -> str:
    files = [f.name for f in SAFE_FILES_DIR.iterdir() if f.is_file()]
    if not files:
        return f"No files in {SAFE_FILES_DIR}."
    return "\n".join(files)


def get_weather(args: dict) -> str:
    location = args.get("location", "")
    if not location:
        return "Error: no location provided."

    try:
        geo_resp = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=10,
        )
        geo_results = geo_resp.json().get("results")
        if not geo_results:
            return f"Could not find a location matching '{location}'."

        place = geo_results[0]
        lat, lon = place["latitude"], place["longitude"]

        weather_resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m",
            },
            timeout=10,
        )
        current = weather_resp.json().get("current", {})
        if "temperature_2m" not in current:
            return "Weather service did not return current data."

        name = place.get("name", location)
        country = place.get("country", "")
        return (
            f"Current weather in {name}, {country}: "
            f"{current['temperature_2m']}°C, "
            f"wind {current.get('wind_speed_10m')} km/h "
            f"(observed at {current.get('time')})"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


def save_memory_tool(args: dict) -> str:
    fact = args.get("fact", "")
    if not fact:
        return "Error: no fact provided."
    memory.save_memory(fact)
    return f"Saved to long-term memory: {fact}"


def search_documents_tool(args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "Error: no query provided."
    results = memory.search_documents(query, SAFE_FILES_DIR)
    if not results:
        return "No relevant content found in the indexed documents."
    lines = []
    for source, text in results:
        snippet = text.strip().replace("\n", " ")[:400]
        lines.append(f"[{source}]: {snippet}")
    return "\n".join(lines)


_docker_client = None


def _get_docker_client():
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def run_python(args: dict) -> str:
    code = args.get("code", "")
    if not code:
        return "Error: no code provided."

    try:
        client = _get_docker_client()
    except Exception as e:
        return f"Error: Docker isn't available ({e}). Is Docker Desktop running?"

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        container = None
        try:
            container = client.containers.run(
                DOCKER_IMAGE,
                command=["python", "/sandbox/snippet.py"],
                volumes={tmp_dir: {"bind": "/sandbox", "mode": "ro"}},
                working_dir="/sandbox",
                network_disabled=DOCKER_NETWORK_DISABLED,   # no internet access from inside
                mem_limit=DOCKER_MEM_LIMIT,
                nano_cpus=DOCKER_CPU_COUNT * 1_000_000_000,  # capped at DOCKER_CPU_COUNT CPU core(s)
                detach=True,
            )
            result = container.wait(timeout=DOCKER_TIMEOUT_SECONDS)
            exit_code = result.get("StatusCode", 1)
            logs = container.logs().decode("utf-8", errors="replace")[-3000:]
        except docker.errors.ImageNotFound:
            return f"Error: {DOCKER_IMAGE} image not found. Run 'docker pull {DOCKER_IMAGE}' once."
        except Exception as e:
            return f"Error running sandboxed code: {e}"
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        if exit_code != 0:
            return f"Exit code {exit_code}. Output:\n{logs}"
        return logs or "(no output, nothing was printed)"


def read_file(args: dict) -> str:
    filename = args.get("filename", "")
    if not filename:
        return "Error: no filename provided."

    # Resolve the path and make sure it's still inside SAFE_FILES_DIR.
    # This blocks tricks like "../../secrets.txt".
    target = (SAFE_FILES_DIR / filename).resolve()
    if SAFE_FILES_DIR not in target.parents and target != SAFE_FILES_DIR:
        return "Error: access denied outside the allowed folder."
    if not target.is_file():
        return f"Error: '{filename}' not found."

    try:
        return memory.extract_text(target, max_chars=5000)[:5000]
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(args: dict) -> str:
    filename = args.get("filename", "")
    content = args.get("content", "")
    mode = args.get("mode", "overwrite")

    if not filename:
        return "Error: no filename provided."
    if mode not in ("overwrite", "append"):
        return f"Error: mode must be 'overwrite' or 'append', got '{mode}'."

    if len(content) > WRITE_FILE_MAX_CHARS:
        return f"Error: content too long ({len(content)} chars, max {WRITE_FILE_MAX_CHARS})."

    target = (SAFE_FILES_DIR / filename).resolve()
    if SAFE_FILES_DIR not in target.parents and target != SAFE_FILES_DIR:
        return "Error: access denied outside the allowed folder."

    if target.suffix.lower() not in WRITE_FILE_ALLOWED_EXTENSIONS:
        return "Error: write_file only supports .txt or .md files."

    try:
        file_mode = "a" if mode == "append" else "w"
        with open(target, file_mode, encoding="utf-8") as f:
            f.write(content)
        action = "Appended to" if mode == "append" else "Wrote"
        return f"{action} '{filename}' ({len(content)} chars)."
    except Exception as e:
        return f"Error writing file: {e}"

# ---------------------------------------------------------------------------
# Project manager tools (formerly the SillyTavern extension's client-side
# tools - see project_manager.py for the actual data model/logic). Each
# wrapper here just translates tool args <-> project_manager calls and turns
# ProjectManagerError into a plain string the model can read and react to,
# the same pattern the file tools above use.
# ---------------------------------------------------------------------------

def project_manager_get_overview(args: dict) -> str:
    state = project_manager._load()
    overview = project_manager.project_overview(state)
    if not overview["projects"]:
        return "No projects exist."
    return json.dumps(overview, ensure_ascii=False)


def project_manager_create_task(args: dict) -> str:
    try:
        state = project_manager._load()
        project = project_manager.resolve_project(state, args.get("project"))
        task = project_manager.create_task(project["id"], args.get("title", ""))
        return f"Created task {task['short_id']}: {task['title']}"
    except ProjectManagerError as e:
        return f"Error: {e}"


def project_manager_update_task_status(args: dict) -> str:
    try:
        state = project_manager._load()
        project = project_manager.resolve_project(state, args.get("project"))
        task = project_manager.get_task(project, args.get("task"))
        if not task:
            return "Error: Task not found or ambiguous. Call project_manager_get_overview and use an exact task ID."
        status = args.get("status")
        if task["status"] == status:
            return f"No change needed; \u201c{task['title']}\u201d is already {status}."
        updated = project_manager.set_task_status(project["id"], task["id"], status)
        return f"Set {updated['short_id']} (\u201c{updated['title']}\u201d) to {status}."
    except ProjectManagerError as e:
        return f"Error: {e}"


def project_manager_update_task_notes(args: dict) -> str:
    try:
        state = project_manager._load()
        project = project_manager.resolve_project(state, args.get("project"))
        task = project_manager.get_task(project, args.get("task"))
        if not task:
            return "Error: Task not found or ambiguous. Call project_manager_get_overview and use an exact task ID."
        updated = project_manager.update_task_notes(
            project["id"], task["id"], args.get("mode", "replace"), args.get("text", "")
        )
        return f"Updated notes for {updated['short_id']} (\u201c{updated['title']}\u201d)."
    except ProjectManagerError as e:
        return f"Error: {e}"


def project_manager_set_all_tasks_status(args: dict) -> str:
    try:
        state = project_manager._load()
        project = project_manager.resolve_project(state, args.get("project"))
        status = args.get("status")
        applied = project_manager.set_all_tasks_status(project["id"], status)
        if not applied:
            return f"No change needed; every non-archived task in {project['short_code']} \u2014 {project['name']} is already {status}."
        return f"Set {len(applied)} task(s) in {project['short_code']} to {status}:\n" + "\n".join(applied)
    except ProjectManagerError as e:
        return f"Error: {e}"


def project_manager_batch_update(args: dict) -> str:
    try:
        state = project_manager._load()
        project = project_manager.resolve_project(state, args.get("project"))
        applied = project_manager.batch_update(project["id"], args.get("operations", []))
        if not applied:
            return "No change needed; every requested batch operation is already satisfied."
        return f"Applied {len(applied)} change(s) to {project['short_code']}:\n" + "\n".join(applied)
    except ProjectManagerError as e:
        return f"Error: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current real-world date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an exact arithmetic expression (numbers and + - * / ( ) only). Always use this for any math instead of computing it yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(2026 - 1889)'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information not in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current real-world weather and temperature for a specific place. Always use this instead of web_search for weather/temperature questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country, e.g. 'Amsterdam, Netherlands'."}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files available for reading.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save an important fact about the user for recall in future conversations (e.g. their name, preferences, ongoing projects). Do not save trivial small talk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember, written as a standalone sentence."}
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Semantically search the user's uploaded .txt and .pdf documents for relevant passages. Use this to find specific information within long or multiple documents, instead of read_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": f"Run a short Python snippet for calculations, data processing, or logic too complex for the calculate tool. Executes in an isolated Docker container with no network access and a {DOCKER_TIMEOUT_SECONDS}-second timeout. Use print() for output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code. Use print() to produce output."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the allowed files folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file to read."}
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new .txt or .md file, or overwrite/append to an existing one, in the allowed files folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file, e.g. 'notes.txt'."},
                    "content": {"type": "string", "description": "The text to write."},
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "'overwrite' replaces the whole file (or creates it if new). 'append' adds to the end of an existing file. Defaults to 'overwrite'.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_manager_get_overview",
            "description": "Read the authoritative persistent project/task overview. Call this before proposing project or task changes whenever the relevant project or task is uncertain.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_manager_create_task",
            "description": "Create a concrete actionable task only when the user clearly states an intention, obligation, or requested action. Do not create tasks from hypotheticals, examples, general discussion, or vague wishes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Existing project ID, short code, or exact project name. Omit only when the focused project is clearly intended."},
                    "title": {"type": "string", "description": "Short, concrete, verb-led task title."},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_manager_update_task_status",
            "description": "Update an existing task's status only when the user clearly says it was started, blocked, completed, cancelled, or returned to pending. Never infer completion from phrases such as 'almost finished'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Existing project ID, short code, or exact project name. Omit only when the focused project is clearly intended."},
                    "task": {"type": "string", "description": "Existing task ID, short ID, or an unambiguous task title."},
                    "status": {"type": "string", "enum": ["pending", "active", "blocked", "done", "cancelled"]},
                },
                "required": ["task", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_manager_update_task_notes",
            "description": "Replace, append to, or clear the notes of an existing task when the user explicitly asks. Use append for additional context and replace only when the user wants existing notes overwritten.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Existing project ID, short code, or exact project name. Omit only when the focused project is clearly intended."},
                    "task": {"type": "string", "description": "Existing task ID, short ID, or an unambiguous task title."},
                    "mode": {"type": "string", "enum": ["replace", "append", "clear"]},
                    "text": {"type": "string", "description": "Note text. Required for replace and append; omit for clear."},
                },
                "required": ["task", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_manager_set_all_tasks_status",
            "description": "Set every non-archived task in one project to the same status in a single operation. Use this for requests such as 'mark every task in the current project as done'. Do not enumerate tasks or use the batch tool for this. Tasks already in the requested status are skipped. This never changes the project's own status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Existing project ID, short code, or exact project name. Omit when the focused project is intended."},
                    "status": {"type": "string", "enum": ["pending", "active", "blocked", "done", "cancelled"]},
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_manager_batch_update",
            "description": "Combine multiple concrete changes from one user message into one atomic project update. Prefer this over separate tool calls when the changes concern the same project. Already-satisfied operations are safely skipped.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Existing project ID, short code, or exact project name. Omit only when the focused project is clearly intended."},
                    "operations": {
                        "type": "array", "minItems": 1, "maxItems": 25,
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["create_task", "update_task_status", "update_task_notes"]},
                                "title": {"type": "string", "description": "For create_task."},
                                "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                                "notes": {"type": "string"},
                                "task": {"type": "string", "description": "For status or note updates: task ID, short ID, or unambiguous title."},
                                "status": {"type": "string", "enum": ["pending", "active", "blocked", "done", "cancelled"]},
                                "mode": {"type": "string", "enum": ["replace", "append", "clear"]},
                                "text": {"type": "string"},
                            },
                            "required": ["type"],
                        },
                    },
                },
                "required": ["operations"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "run_python": run_python,
    "web_search": web_search,
    "get_weather": get_weather,
    "list_files": list_files,
    "read_file": read_file,
    "save_memory": save_memory_tool,
    "search_documents": search_documents_tool,
    "write_file": write_file,
    "project_manager_get_overview": project_manager_get_overview,
    "project_manager_create_task": project_manager_create_task,
    "project_manager_update_task_status": project_manager_update_task_status,
    "project_manager_update_task_notes": project_manager_update_task_notes,
    "project_manager_set_all_tasks_status": project_manager_set_all_tasks_status,
    "project_manager_batch_update": project_manager_batch_update,
}


# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{LLAMA_SERVER_URL}/v1/models")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Project manager HTTP API - what the SillyTavern extension's UI talks to.
# Plain REST, no tool-schema wrapping: these are user clicks, not model
# calls. Every write here uses the exact same project_manager functions the
# model's tools use above, so the UI and the model can never disagree about
# what's a valid change.
# ---------------------------------------------------------------------------

def _pm_error(e: ProjectManagerError):
    raise HTTPException(status_code=400, detail=str(e))


def _serialize_task(task: dict) -> dict:
    return task


def _serialize_project(project: dict, *, include_tasks: bool = True) -> dict:
    data = {k: v for k, v in project.items() if k != "tasks"}
    if include_tasks:
        data["tasks"] = [_serialize_task(t) for t in project["tasks"].values()]
    return data


@app.get("/projects")
async def api_list_projects():
    state = await asyncio.to_thread(project_manager._load)
    return {
        "focused_project_id": state.get("focused_project_id"),
        "projects": [_serialize_project(p) for p in project_manager.get_projects(state)],
    }


@app.post("/projects")
async def api_create_project(request: Request):
    body = await request.json()
    try:
        project = await asyncio.to_thread(project_manager.create_project, body.get("name", ""))
    except ProjectManagerError as e:
        _pm_error(e)
    return _serialize_project(project)


@app.get("/projects/{project_id}")
async def api_get_project(project_id: str):
    state = await asyncio.to_thread(project_manager._load)
    project = state["projects"].get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return _serialize_project(project)


@app.patch("/projects/{project_id}")
async def api_update_project(project_id: str, request: Request):
    body = await request.json()
    try:
        project = None
        if "name" in body:
            project = await asyncio.to_thread(project_manager.rename_project, project_id, body["name"])
        if "status" in body:
            project = await asyncio.to_thread(project_manager.set_project_status, project_id, body["status"])
        if project is None:
            state = await asyncio.to_thread(project_manager._load)
            project = state["projects"].get(project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found.")
    except ProjectManagerError as e:
        _pm_error(e)
    return _serialize_project(project)


@app.post("/projects/{project_id}/focus")
async def api_focus_project(project_id: str):
    try:
        project = await asyncio.to_thread(project_manager.set_focused_project, project_id)
    except ProjectManagerError as e:
        _pm_error(e)
    return _serialize_project(project)


@app.delete("/projects/{project_id}")
async def api_delete_project(project_id: str):
    try:
        await asyncio.to_thread(project_manager.delete_project, project_id)
    except ProjectManagerError as e:
        _pm_error(e)
    return {"deleted": project_id}


@app.post("/projects/{project_id}/tasks")
async def api_create_task(project_id: str, request: Request):
    body = await request.json()
    try:
        task = await asyncio.to_thread(
            project_manager.create_task,
            project_id, body.get("title", ""),
            priority=body.get("priority", "normal"), notes=body.get("notes", ""),
        )
    except ProjectManagerError as e:
        _pm_error(e)
    return _serialize_task(task)


@app.patch("/projects/{project_id}/tasks/{task_id}")
async def api_update_task(project_id: str, task_id: str, request: Request):
    body = await request.json()
    try:
        task = None
        if "status" in body:
            task = await asyncio.to_thread(project_manager.set_task_status, project_id, task_id, body["status"])
        if any(k in body for k in ("title", "priority", "notes")):
            task = await asyncio.to_thread(
                project_manager.update_task_details, project_id, task_id,
                title=body.get("title"), priority=body.get("priority"), notes=body.get("notes"),
            )
        if "notes_mode" in body:
            task = await asyncio.to_thread(
                project_manager.update_task_notes, project_id, task_id,
                body["notes_mode"], body.get("text", ""),
            )
        if task is None:
            raise HTTPException(status_code=400, detail="No recognized fields to update.")
    except ProjectManagerError as e:
        _pm_error(e)
    return _serialize_task(task)


@app.delete("/projects/{project_id}/tasks/{task_id}")
async def api_delete_task(project_id: str, task_id: str):
    try:
        await asyncio.to_thread(project_manager.delete_task, project_id, task_id)
    except ProjectManagerError as e:
        _pm_error(e)
    return {"deleted": task_id}


@app.post("/projects/{project_id}/tasks/reorder")
async def api_reorder_tasks(project_id: str, request: Request):
    body = await request.json()
    try:
        await asyncio.to_thread(project_manager.reorder_tasks, project_id, body.get("ordered_task_ids", []))
        state = await asyncio.to_thread(project_manager._load)
        project = state["projects"].get(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")
    except ProjectManagerError as e:
        _pm_error(e)
    return _serialize_project(project)


async def _stream_chat(client: httpx.AsyncClient, body: dict):
    """POST one chat-completion request with stream=True and yield the
    decoded JSON of each SSE chunk from llama-server."""
    async with client.stream(
        "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                return
            yield json.loads(payload)


async def agent_loop(upstream_body: dict):
    """
    Drives the tool-calling loop against llama-server, streaming the whole
    way. Yields:
      ("delta", text)   - a piece of the FINAL answer, forwarded the moment
                           it's clear this iteration isn't a tool call.
      ("done", message) - the complete final assistant message (role +
                           content), once the loop is finished.
    Tool-call iterations are executed internally and never reach the caller
    as deltas - only the model's eventual direct answer streams through.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        for _ in range(MAX_TOOL_ITERATIONS):
            content = ""
            tool_calls = {}
            mode = None  # becomes "content" or "tool_calls" once known

            async for chunk in _stream_chat(client, upstream_body):
                delta = chunk["choices"][0].get("delta", {})

                delta_tool_calls = delta.get("tool_calls")
                if delta_tool_calls:
                    mode = "tool_calls"
                    for tc in delta_tool_calls:
                        idx = tc.get("index", 0)
                        entry = tool_calls.setdefault(
                            idx,
                            {"id": None, "type": "function",
                             "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            entry["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            entry["function"]["arguments"] += fn["arguments"]
                    continue

                delta_content = delta.get("content")
                if delta_content:
                    if mode is None:
                        mode = "content"
                    content += delta_content
                    # buffered, not yielded here — see below

            if mode == "tool_calls" and tool_calls:
                message = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
                }
                print(f"[AGENT] Model requested {len(message['tool_calls'])} tool call(s)")
                upstream_body["messages"].append(message)

                for call in message["tool_calls"]:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    if name in TOOL_FUNCTIONS:
                        # Run in a thread so a slow web search doesn't freeze the server.
                        result = await asyncio.to_thread(TOOL_FUNCTIONS[name], args)
                    else:
                        result = f"Error: unknown tool '{name}'"

                    print(f"[AGENT] {name}({args}) ->")
                    print(f"[AGENT]   {str(result)[:400]}")

                    upstream_body["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": str(result),
                        }
                    )
                continue  # loop again so the model can use the tool result

            # No tool call -> this is the final answer (already streamed above).
            print("[AGENT] Model answered directly, without calling any tool.")
            if content:
                yield ("delta", content)
            yield ("done", {"role": "assistant", "content": content})
            return

        # Hit MAX_TOOL_ITERATIONS without a final answer - bail out safely.
        bail_message = "(Agent stopped: too many tool calls in a row.)"
        yield ("delta", bail_message)
        yield ("done", {"role": "assistant", "content": bail_message})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    client_wants_stream = body.get("stream", False)

    # We always stream from llama-server internally (agent_loop needs deltas
    # to detect tool calls early), regardless of what the client asked for.
    upstream_body = dict(body)
    upstream_body["stream"] = True
    upstream_body["tools"] = TOOLS
    upstream_body["tool_choice"] = "auto"

    tool_instruction = {
        "role": "system",
        "content": (
            "You have tools available: get_current_time, calculate, run_python, "
            "web_search, get_weather, list_files, read_file, save_memory, write_file, "
            "search_documents, project_manager_get_overview, project_manager_create_task, "
            "project_manager_update_task_status, project_manager_update_task_notes, "
            "project_manager_set_all_tasks_status, project_manager_batch_update. "
            "You MUST call the relevant tool whenever the user "
            "asks about current events, real-time facts, dates/times, weather, "
            "exact arithmetic, or anything you are not fully certain of from "
            "memory. For weather/temperature questions, always use get_weather, "
            "never web_search. Use search_documents (not read_file) when looking "
            "for specific information inside long or multiple documents. Use "
            "calculate for simple arithmetic, or run_python for anything needing "
            "actual code logic. Call save_memory when the user shares a durable "
            "fact about themselves worth remembering - not for small talk. "
            "Use write_file when the user asks you to create, save, write out, or "
            "update a .txt or .md file - use mode 'overwrite' to replace a file's "
            "contents (or create a new one) and mode 'append' to add to the end of "
            "an existing file without erasing it. "
            "Use the project_manager_* tools only when the user clearly states a "
            "concrete project/task action (create, start, block, complete, cancel, "
            "or annotate a task) - never from hypotheticals or vague wishes. Call "
            "project_manager_get_overview first if which project or task is meant "
            "isn't already clear from the persistent project state below. When one "
            "message contains several concrete changes for the same project, use "
            "project_manager_batch_update once instead of separate calls; for "
            "'mark everything as done'-style requests, use "
            "project_manager_set_all_tasks_status once instead of enumerating tasks. "
            "Project-manager changes apply immediately - there is no separate "
            "confirmation step, so only call these tools when the user's intent is "
            "unambiguous. "
            "IMPORTANT for multi-step questions: if answering fully requires "
            "several pieces of information, call tools one at a time in sequence, "
            "using each result to decide your next step, before giving your final "
            "answer. Do not stop after one tool call if the question isn't fully "
            "answered yet. Never guess or invent facts, dates, statistics, or "
            "search results that a tool could actually check for you. If a tool "
            "returns no useful result, say so honestly instead of making "
            "something up."
            "When you decide to call a tool, call it directly - "
            "do not write any explanation, plan, or commentary before or "
            "alongside the tool call. Save your explanation, if any, for your "
            "final answer after the tool result comes back."
        ),
    }
    messages_to_prepend = [tool_instruction]

    # Server-side equivalent of the old extension's setExtensionPrompt():
    # inject the focused project's state directly, no tool call needed.
    project_state_text = await asyncio.to_thread(
        lambda: project_manager.build_context_text(project_manager._load())
    )
    if project_state_text:
        messages_to_prepend.append({"role": "system", "content": project_state_text})

    # Auto-recall: silently check if any saved memories are relevant to what
    # the user just said, and inject them - no tool call needed for this part.
    last_user_msg = next(
        (m["content"] for m in reversed(body["messages"]) if m.get("role") == "user"),
        None,
    )
    if last_user_msg:
        relevant = await asyncio.to_thread(memory.search_memories, last_user_msg)
        if relevant:
            print(f"[AGENT] Recalled {len(relevant)} relevant memory item(s)")
            messages_to_prepend.append({
                "role": "system",
                "content": "Relevant things you remember about this user from past "
                            "conversations:\n" + "\n".join(f"- {m}" for m in relevant),
            })

    upstream_body["messages"] = messages_to_prepend + upstream_body["messages"]

    if not client_wants_stream:
        final_message = {"role": "assistant", "content": ""}
        async for kind, payload in agent_loop(upstream_body):
            if kind == "done":
                final_message = payload
        final_data = {
            "choices": [{"message": final_message, "finish_reason": "stop"}]
        }
        return JSONResponse(content=final_data)

    # Client wants real streaming: forward each content delta to SillyTavern
    # the moment it arrives from llama-server.
    async def event_stream():
        async for kind, payload in agent_loop(upstream_body):
            if kind == "delta":
                chunk = {
                    "choices": [{"delta": {"content": payload}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            # "done" carries the full message for the non-streaming path only;
            # its content has already been sent as deltas above.

        done_chunk = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
