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
import time
import uuid
from datetime import datetime
from pathlib import Path

import docker

import httpx
from ddgs import DDGS
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

import calendar_manager
import memory
import project_manager
from calendar_manager import CalendarError
from project_manager import ProjectManagerError
from config import (
    CORS_ALLOWED_ORIGINS,
    DELETE_FILE_ALLOWED_EXTENSIONS,
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


class LlamaServerError(Exception):
    """Raised when llama-server can't be reached, or drops the connection
    mid-response. Caught in agent_loop and turned into a friendly message
    instead of an unhandled 500."""


# Prefixes used across the tool functions above to signal failure. Not
# fully consistent historically (most use "Error:", but web_search uses
# "Search failed:" and get_weather uses "Weather lookup failed:") - this
# widens detection to catch all of them rather than fixing every tool
# function's wording, which would be a larger, riskier change.
_FAILURE_PREFIXES = ("Error:", "Error running", "Error evaluating", "Error reading", "Error writing",
                     "Exit code", "Search failed:", "Weather lookup failed:")


def _tool_call_failed(result) -> bool:
    text = str(result)
    return any(text.startswith(p) for p in _FAILURE_PREFIXES)


# Calendar UID-dependent write tools (edit/delete) require a UID that can
# only be known once a search/list call's result has actually been seen.
# If the model requests one of these in the SAME response as a read call
# (parallel tool calls - this model does this sometimes), it has no real
# UID yet and will fabricate a plausible-looking one; iCloud rejects it
# with an opaque 412 and no clear explanation why. Caught here in code
# rather than relying on the "never guess a UID" prompt instruction alone -
# verified live this session that instruction does not reliably prevent it,
# because the model isn't disobeying, it's generating both calls before
# either has executed.
_CALENDAR_READ_TOOLS = {"calendar_list_events", "calendar_search_events"}
_CALENDAR_UID_WRITE_TOOLS = {"calendar_edit_event", "calendar_delete_event"}


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
    global _docker_client
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
            # Covers the daemon being stopped/restarted mid-session, a crashed
            # container, etc. Drop the cached client so the *next* call
            # reconnects fresh instead of reusing one pointed at a dead daemon.
            _docker_client = None
            return f"Error running sandboxed code: {e}. If Docker Desktop was closed or restarted, try again."
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


def edit_file(args: dict) -> str:
    filename = args.get("filename", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")

    if not filename:
        return "Error: no filename provided."
    if not old_text:
        return "Error: no old_text provided to find."

    target = (SAFE_FILES_DIR / filename).resolve()
    if SAFE_FILES_DIR not in target.parents and target != SAFE_FILES_DIR:
        return "Error: access denied outside the allowed folder."
    if not target.is_file():
        return f"Error: '{filename}' not found."
    if target.suffix.lower() not in WRITE_FILE_ALLOWED_EXTENSIONS:
        return "Error: edit_file only supports .txt or .md files."

    try:
        current = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    count = current.count(old_text)
    if count == 0:
        return f"Error: old_text not found in '{filename}'. Nothing was changed."
    if count > 1:
        return (f"Error: old_text appears {count} times in '{filename}' - it must be "
                f"unique. Include more surrounding context and try again.")

    updated = current.replace(old_text, new_text, 1)
    if len(updated) > WRITE_FILE_MAX_CHARS:
        return f"Error: resulting file would be too long ({len(updated)} chars, max {WRITE_FILE_MAX_CHARS})."

    try:
        target.write_text(updated, encoding="utf-8")
        return f"Edited '{filename}': replaced 1 occurrence."
    except Exception as e:
        return f"Error writing file: {e}"


def delete_file(args: dict) -> str:
    filename = args.get("filename", "")
    if not filename:
        return "Error: no filename provided."

    target = (SAFE_FILES_DIR / filename).resolve()
    if SAFE_FILES_DIR not in target.parents and target != SAFE_FILES_DIR:
        return "Error: access denied outside the allowed folder."
    if not target.is_file():
        return f"Error: '{filename}' not found."
    if target.suffix.lower() not in DELETE_FILE_ALLOWED_EXTENSIONS:
        return f"Error: delete_file only supports {', '.join(DELETE_FILE_ALLOWED_EXTENSIONS)} files."

    try:
        target.unlink()
        return f"Deleted '{filename}'."
    except Exception as e:
        return f"Error deleting file: {e}"

# ---------------------------------------------------------------------------
# Calendar tools (calendar_manager.py) - iCloud via CalDAV. Read tools
# execute immediately; write tools only STAGE a change and return a
# description - only calendar_confirm_pending actually touches iCloud. See
# calendar_manager.py's module docstring for the full safety rationale.
# ---------------------------------------------------------------------------

def _format_events(events: list) -> str:
    lines = []
    for e in events:
        line = f"[{e['uid']}] {e['title']}: {e['start']}"
        if e["end"]:
            line += f" - {e['end']}"
        if e["location"]:
            line += f" @ {e['location']}"
        if e.get("calendar"):
            line += f" ({e['calendar']})"
        lines.append(line)
    return "\n".join(lines)


def calendar_list_calendars(args: dict) -> str:
    try:
        names = calendar_manager.list_calendar_names()
    except CalendarError as e:
        return f"Error: {e}"
    return "Available iCloud calendars: " + ", ".join(names)


def calendar_list_events(args: dict) -> str:
    try:
        events = calendar_manager.list_events(
            args.get("start"), args.get("end"), args.get("calendar_name")
        )
    except CalendarError as e:
        return f"Error: {e}"
    return _format_events(events) if events else "No events found in that range."


def calendar_search_events(args: dict) -> str:
    try:
        events = calendar_manager.search_events(
            args.get("query", ""), args.get("start"), args.get("end"), args.get("calendar_name")
        )
    except CalendarError as e:
        return f"Error: {e}"
    return _format_events(events) if events else "No matching events found."


def calendar_create_event(args: dict) -> str:
    try:
        return calendar_manager.stage_create_event(
            args.get("title", ""), args.get("start", ""), args.get("end"),
            args.get("location", ""), args.get("description", ""), args.get("calendar_name"),
        )
    except CalendarError as e:
        return f"Error: {e}"


def calendar_edit_event(args: dict) -> str:
    try:
        return calendar_manager.stage_edit_event(
            args.get("event_uid", ""), args.get("title"), args.get("start"), args.get("end"),
            args.get("location"), args.get("description"), args.get("calendar_name"),
        )
    except CalendarError as e:
        return f"Error: {e}"


def calendar_delete_event(args: dict) -> str:
    try:
        return calendar_manager.stage_delete_event(args.get("event_uid", ""), args.get("calendar_name"))
    except CalendarError as e:
        return f"Error: {e}"


def calendar_confirm_pending(args: dict) -> str:
    try:
        return calendar_manager.confirm_pending()
    except CalendarError as e:
        return f"Error: {e}"


def calendar_cancel_pending(args: dict) -> str:
    return calendar_manager.cancel_pending()


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
            "name": "edit_file",
            "description": "Make a targeted edit to an existing .txt or .md file by replacing one exact, unique piece of text with new text. Use this instead of write_file when only part of a file needs to change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the existing file to edit."},
                    "old_text": {"type": "string", "description": "The exact existing text to find and replace. Must be unique within the file - include surrounding context if needed."},
                    "new_text": {"type": "string", "description": "The text to replace it with."},
                },
                "required": ["filename", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Permanently delete an existing file from the allowed files folder. This cannot be undone - only call this when the user clearly and explicitly asks to delete a specific named file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file to delete."}
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_list_calendars",
            "description": "List the names of all iCloud calendars on the user's account. Use this if the user asks what calendars they have, or if events seem to be missing from calendar_list_events/calendar_search_events results and you want to check what calendar_name values are valid. Read-only, executes immediately.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_list_events",
            "description": "List events across ALL of the user's iCloud calendars in a date range (or just one calendar if calendar_name is given). Use for questions like 'what's on my calendar' or 'what do I have this week'. Defaults to the next 14 days if no range is given. Read-only, executes immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start of range, 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'. Defaults to now. For a single specific day, pass the same date for both start and end - that returns the whole day."},
                    "end": {"type": "string", "description": "End of range, same format. Defaults to 14 days after start. Same value as start is valid and means 'just that one day'."},
                    "calendar_name": {"type": "string", "description": "Only needed if the user has multiple iCloud calendars and named one. Omit otherwise."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_search_events",
            "description": "Search across ALL of the user's iCloud calendars by keyword in event titles, locations, and descriptions (or just one calendar if calendar_name is given). Use this to find a specific event (e.g. before editing or deleting it) rather than guessing its UID. Read-only, executes immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword to search for."},
                    "start": {"type": "string", "description": "Optional start of search range, 'YYYY-MM-DD'. Defaults to 7 days ago."},
                    "end": {"type": "string", "description": "Optional end of search range, 'YYYY-MM-DD'. Defaults to 90 days ahead."},
                    "calendar_name": {"type": "string", "description": "Only needed for a specific named calendar. Omit otherwise."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Propose creating a new REAL event on the user's iCloud calendar (an actual appointment, meeting, or plan - not an in-character/roleplay scheduled action). This does NOT create it yet - it only stages the change and returns a description of exactly what would be created. You must relay that description to the user and get an explicit confirmation before calling calendar_confirm_pending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title."},
                    "start": {"type": "string", "description": "Start time, 'YYYY-MM-DD HH:MM'."},
                    "end": {"type": "string", "description": "End time, same format. Defaults to 1 hour after start."},
                    "location": {"type": "string", "description": "Optional location."},
                    "description": {"type": "string", "description": "Optional notes/description."},
                    "calendar_name": {"type": "string", "description": "Only needed for a specific named calendar. Omit otherwise."},
                },
                "required": ["title", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_edit_event",
            "description": "Propose editing an existing iCloud calendar event by its UID (get this from calendar_list_events or calendar_search_events first - never guess a UID). This does NOT apply the edit yet - it only stages the change and returns a description of exactly what would change. Relay that to the user and get explicit confirmation before calling calendar_confirm_pending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_uid": {"type": "string", "description": "Exact UID of the event to edit, from a prior list/search result."},
                    "title": {"type": "string", "description": "New title, if changing it."},
                    "start": {"type": "string", "description": "New start time 'YYYY-MM-DD HH:MM', if changing it."},
                    "end": {"type": "string", "description": "New end time, if changing it."},
                    "location": {"type": "string", "description": "New location, if changing it."},
                    "description": {"type": "string", "description": "New description/notes, if changing it."},
                    "calendar_name": {"type": "string", "description": "Only needed for a specific named calendar. Omit otherwise."},
                },
                "required": ["event_uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_delete_event",
            "description": "Propose permanently deleting an existing iCloud calendar event by its UID (get this from calendar_list_events or calendar_search_events first - never guess a UID). This does NOT delete it yet - it only stages the change. Relay the description to the user and get explicit confirmation before calling calendar_confirm_pending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_uid": {"type": "string", "description": "Exact UID of the event to delete, from a prior list/search result."},
                    "calendar_name": {"type": "string", "description": "Only needed for a specific named calendar. Omit otherwise."},
                },
                "required": ["event_uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_confirm_pending",
            "description": "Apply the currently staged calendar change (from calendar_create_event, calendar_edit_event, or calendar_delete_event) to the real iCloud calendar. Only call this as your very next tool call after the user's following message clearly and explicitly confirms (e.g. 'yes', 'confirm', 'go ahead', 'do it'). Never call this speculatively, preemptively, or without an explicit confirmation message from the user in between.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_cancel_pending",
            "description": "Discard the currently staged calendar change without applying it. Call this if the user declines, says no, or asks for something different instead of confirming.",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
    "edit_file": edit_file,
    "delete_file": delete_file,
    "calendar_list_calendars": calendar_list_calendars,
    "calendar_list_events": calendar_list_events,
    "calendar_search_events": calendar_search_events,
    "calendar_create_event": calendar_create_event,
    "calendar_edit_event": calendar_edit_event,
    "calendar_delete_event": calendar_delete_event,
    "calendar_confirm_pending": calendar_confirm_pending,
    "calendar_cancel_pending": calendar_cancel_pending,
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
    try:
        async with client.stream(
            "POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=body
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    return
                yield json.loads(payload)
    except httpx.ConnectError as e:
        raise LlamaServerError(
            "Could not connect to llama-server. Is it still running?"
        ) from e
    except (httpx.ReadError, httpx.RemoteProtocolError) as e:
        raise LlamaServerError(
            "Lost connection to llama-server mid-response (it may have crashed)."
        ) from e
    except httpx.TimeoutException as e:
        raise LlamaServerError("llama-server timed out responding.") from e
    except httpx.HTTPStatusError as e:
        raise LlamaServerError(
            f"llama-server returned an error (HTTP {e.response.status_code})."
        ) from e
    except json.JSONDecodeError as e:
        raise LlamaServerError(
            "llama-server sent a malformed response (it may have crashed mid-reply)."
        ) from e


async def agent_loop(upstream_body: dict):
    """
    Drives the tool-calling loop against llama-server, streaming the whole
    way. Yields:
      ("delta", text)    - a piece of the FINAL answer, forwarded the
                            moment it's clear this iteration isn't a tool
                            call.
      ("done", message)  - the complete final assistant message (role +
                            content), once the loop is finished.
      ("handoff", message) - the model wants to call one or more tools this
                            server doesn't own (registered client-side by a
                            SillyTavern extension, e.g. the check-ins
                            extension's schedule_character_action). Passed
                            back to the caller UNRESOLVED, exactly as
                            llama-server produced it, so SillyTavern can run
                            its own registered handler and continue the
                            conversation itself. The loop stops here - this
                            server never guesses at a client tool's result.

    Tool-call iterations for tools this server DOES own are executed
    internally and never reach the caller as deltas - only the model's
    eventual direct answer, or a handoff, does.

    Simplification: if a single model turn requests a MIX of server-owned
    and client-owned tools together, the whole turn is handed off unresolved
    rather than partially executed - partially resolving would leave some
    tool_call_ids answered and others not, which breaks the next request.
    This is rare in practice (mixed-origin tool calls in one turn); if it
    happens, the server-owned tools simply get called again next turn once
    the client resolves its half and sends the follow-up request.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        for _ in range(MAX_TOOL_ITERATIONS):
            content = ""
            tool_calls = {}
            mode = None  # becomes "content" or "tool_calls" once known

            try:
                async for chunk in _stream_chat(client, upstream_body):
                    delta = chunk["choices"][0].get("delta", {})

                    delta_tool_calls = delta.get("tool_calls")
                    if delta_tool_calls:
                        mode = "tool_calls"
                        for tc in delta_tool_calls:
                            idx = tc.get("index", 0)
                            entry = tool_calls.setdefault(
                                idx,
                                {"index": idx, "id": None, "type": "function",
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
            except LlamaServerError as e:
                print(f"[AGENT] {e}")
                yield ("delta", f"⚠️ {e}")
                yield ("done", {"role": "assistant", "content": str(e)})
                return

            if mode == "tool_calls" and tool_calls:
                calls = [tool_calls[i] for i in sorted(tool_calls)]
                message = {"role": "assistant", "content": content or None, "tool_calls": calls}

                unknown = [c for c in calls if c["function"]["name"] not in TOOL_FUNCTIONS]
                if unknown:
                    names = [c["function"]["name"] for c in unknown]
                    print(f"[AGENT] Handing off {len(unknown)} client-owned tool call(s) to SillyTavern: {names}")
                    yield ("handoff", message)
                    return

                print(f"[AGENT] Model requested {len(calls)} tool call(s)")
                upstream_body["messages"].append(message)

                names_in_batch = {c["function"]["name"] for c in calls}
                blocked_uid_writes = (
                    names_in_batch & _CALENDAR_UID_WRITE_TOOLS
                    if names_in_batch & _CALENDAR_READ_TOOLS
                    else set()
                )

                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    if name in blocked_uid_writes:
                        # Blocked before ever reaching calendar_manager - the
                        # UID in `args` was necessarily fabricated, since the
                        # search/list call in this same batch hadn't run yet
                        # when this call was generated.
                        result = (
                            f"Error: {name} was called in the same response as a calendar "
                            f"search/list call, before that result could be seen - so the "
                            f"event_uid you used cannot be correct, it must have been "
                            f"guessed. Wait for the search/list result, THEN call {name} "
                            f"again as a separate follow-up using the exact uid field from "
                            f"that result. Do not guess or invent a UID."
                        )
                    else:
                        # Run in a thread so a slow web search doesn't freeze the server.
                        result = await asyncio.to_thread(TOOL_FUNCTIONS[name], args)

                    print(f"[AGENT] {name}({args}) ->")
                    print(f"[AGENT]   {str(result)[:400]}")

                    upstream_body["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": str(result),
                        }
                    )

                    # Placement fix (handover-9): a failure-honesty clause
                    # sitting once in the system prompt wasn't reliably
                    # followed (handover-7). Injecting a short reminder
                    # right next to the failed result, every time, is more
                    # effective for a 14B model than one buried instruction.
                    if _tool_call_failed(result):
                        upstream_body["messages"].append(
                            {
                                "role": "system",
                                "content": (
                                    "The tool call above FAILED. Tell the user honestly that "
                                    "it failed and why. Do not invent, guess, or substitute "
                                    "any numbers, facts, or output in its place - not even as "
                                    "an example or hypothetical."
                                ),
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

    # SillyTavern extensions can register their own tools (e.g. the
    # check-ins extension's schedule_character_action) - these arrive as
    # `tools` on the incoming request. Merge them in alongside this
    # server's own list instead of discarding them: any client tool whose
    # name doesn't collide with one of ours gets added, so the model can
    # call it. Calls to these are handed back to SillyTavern unresolved by
    # agent_loop (see "handoff" above) instead of this server guessing at
    # an answer.
    client_tools = [
        t for t in (body.get("tools") or [])
        if isinstance(t, dict) and t.get("function", {}).get("name") not in TOOL_FUNCTIONS
    ]
    client_tool_names = {t["function"]["name"] for t in client_tools if t.get("function", {}).get("name")}

    # We always stream from llama-server internally (agent_loop needs deltas
    # to detect tool calls early), regardless of what the client asked for.
    upstream_body = dict(body)
    upstream_body["stream"] = True
    upstream_body["tools"] = TOOLS + client_tools
    upstream_body["tool_choice"] = "auto"

    tool_instruction = {
        "role": "system",
        "content": (
            "You have tools available: get_current_time, calculate, run_python, "
            "web_search, get_weather, list_files, read_file, save_memory, write_file, "
            "edit_file, delete_file, search_documents, calendar_list_calendars, "
            "calendar_list_events, "
            "calendar_search_events, calendar_create_event, calendar_edit_event, "
            "calendar_delete_event, calendar_confirm_pending, calendar_cancel_pending, "
            "project_manager_get_overview, project_manager_create_task, "
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
            "Use edit_file for a small targeted change inside an existing file instead "
            "of rewriting the whole thing with write_file. Only call delete_file when "
            "the user clearly and explicitly asks to delete a specific named file - "
            "never as a side effect of another request, and never guess the filename "
            "if it's ambiguous; ask the user to confirm instead. "
            "Use calendar_list_events or calendar_search_events freely to read the "
            "user's iCloud calendar - these are read-only, need no confirmation, and "
            "search across all of the user's calendars by default. If the user asks "
            "what calendars they have, or expected events aren't showing up, call "
            "calendar_list_calendars to see the actual calendar names. "
            "CRITICAL: a system message near the top of this conversation, labeled "
            "[CURRENT DATE/TIME], gives you today's real date and time on every "
            "single request - use it to compute any relative date phrase ('this "
            "week', 'today', 'tomorrow', 'next month', etc.) for calendar_list_events, "
            "calendar_search_events, calendar_create_event, and calendar_edit_event. "
            "You do NOT need to call get_current_time for this - the date is already "
            "provided fresh every turn. NEVER guess or assume a date/year from memory "
            "or training data for a calendar call - a wrong year will silently return "
            "the wrong (usually empty) results instead of erroring, so this mistake is "
            "easy to make and easy to miss. If you don't need a specific range, you "
            "may also omit start/end entirely and let the tool default to today "
            "onward. "
            "calendar_create_event, calendar_edit_event, and calendar_delete_event "
            "NEVER change the real calendar by themselves - they only stage a "
            "proposed change and return a description of it. After calling one, tell "
            "the user exactly what will happen and wait for their reply. Only call "
            "calendar_confirm_pending as your very next tool call if the user's "
            "following message clearly and explicitly confirms (e.g. 'yes', "
            "'confirm', 'go ahead') - never call it speculatively, preemptively, or "
            "on the same turn as staging the change. If the user declines or wants "
            "something different, call calendar_cancel_pending instead. Always look "
            "up an event's UID with calendar_list_events or calendar_search_events "
            "before editing or deleting it - never guess a UID. "
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
            + (
                f"Additional tools provided by the connected frontend are also "
                f"available this turn: {', '.join(sorted(client_tool_names))}. Call "
                f"them normally, following their own descriptions, and only when they "
                f"clearly fit the user's request. "
                if client_tool_names else ""
            ) +
            "IMPORTANT for multi-step questions: if answering fully requires "
            "several pieces of information, call tools one at a time in sequence, "
            "using each result to decide your next step, before giving your final "
            "answer. Do not stop after one tool call if the question isn't fully "
            "answered yet. Never guess or invent facts, dates, statistics, or "
            "search results that a tool could actually check for you. If a tool "
            "returns no useful result, say so honestly instead of making "
            "something up. If a tool result begins with 'Error:', the tool call "
            "FAILED - you must tell the user it failed and relay the reason "
            "(e.g. 'Docker isn't running'). Never substitute your own guessed "
            "numbers, facts, or output in place of a failed tool's result, even "
            "if you present it as an example or hypothetical."
            "When you decide to call a tool, call it directly - "
            "do not write any explanation, plan, or commentary before or "
            "alongside the tool call. Save your explanation, if any, for your "
            "final answer after the tool result comes back."
        ),
    }
    messages_to_prepend = [tool_instruction]

    # Inject the real current date/time on EVERY request, the same pattern
    # used for project state and memory recall below. Previously the model
    # was only told to call get_current_time when reasoning about relative
    # dates ("tomorrow", "this week") - that worked right after the call,
    # but in a longer conversation that tool result can scroll out of the
    # 8192-token context window, and the model reverts to guessing (usually
    # a training-data-plausible year like 2023). Injecting it fresh every
    # turn removes the dependency on the model remembering to check, or on
    # that earlier result still being in context - this matters most for
    # the calendar tools, where a wrong year silently returns the wrong
    # (often empty) results instead of erroring.
    current_datetime_text = (
        f"[CURRENT DATE/TIME] {datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S')}. "
        f"This is the real, authoritative current date and time. Use it for any "
        f"relative date/time calculation (today, tomorrow, this week, next Friday, "
        f"etc.) - especially for the calendar tools. Never guess or assume a "
        f"different date or year from memory or training data."
    )
    messages_to_prepend.append({"role": "system", "content": current_datetime_text})

    # Server-side equivalent of the old extension's setExtensionPrompt():
    # inject the focused project's state directly, no tool call needed.
    project_state_text = await asyncio.to_thread(
        lambda: project_manager.build_context_text(project_manager._load())
    )
    if project_state_text:
        messages_to_prepend.append({"role": "system", "content": project_state_text})
    print(f"[AGENT] Project state text sent to model:\n{project_state_text or '(none)'}")

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
    # Full OpenAI chat-completion responses carry an id/object/created/model
    # envelope alongside "choices", and each choice normally has its own
    # "index". Plain text replies apparently don't need this for SillyTavern
    # to just pull out choices[0].message.content, but its tool-calling
    # logic may specifically check for these fields before trusting/
    # executing a tool call - and until this session, no response with
    # tool_calls had ever actually reached a real client, so this gap was
    # never exercised. Adding the full envelope is cheap and brings us to
    # spec regardless of whether this turns out to be the actual cause.
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())
    model_name = body.get("model") or "agent"

    if not client_wants_stream:
        final_message = {"role": "assistant", "content": ""}
        finish_reason = "stop"
        async for kind, payload in agent_loop(upstream_body):
            if kind in ("done", "handoff"):
                final_message = payload
                finish_reason = "tool_calls" if kind == "handoff" else "stop"
        final_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model_name,
            "choices": [{"index": 0, "message": final_message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JSONResponse(content=final_data)

    # Client wants real streaming: forward each content delta to SillyTavern
    # the moment it arrives from llama-server.
    def sse(choice: dict) -> bytes:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_name,
            "choices": [choice],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    async def event_stream():
        async for kind, payload in agent_loop(upstream_body):
            if kind == "delta":
                yield sse({"index": 0, "delta": {"content": payload}, "finish_reason": None})
                continue

            if kind == "handoff":
                # Emit the whole unresolved tool_calls list in one delta -
                # we already have it fully assembled server-side, so there's
                # no need to fake incremental streaming for it. SillyTavern
                # accumulates tool_calls by index either way.
                delta = {"tool_calls": payload["tool_calls"]}
                if payload.get("content"):
                    delta["content"] = payload["content"]
                yield sse({"index": 0, "delta": delta, "finish_reason": None})
                yield sse({"index": 0, "delta": {}, "finish_reason": "tool_calls"})
                yield b"data: [DONE]\n\n"
                return
            # "done" carries the full message for the non-streaming path only;
            # its content has already been sent as deltas above.

        yield sse({"index": 0, "delta": {}, "finish_reason": "stop"})
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
