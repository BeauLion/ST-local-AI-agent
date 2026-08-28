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
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import docker

import httpx
import numpy as np
from ddgs import DDGS
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

import attire_manager
import attire_subagent
import calendar_manager
import duration_manager
import memory
import project_manager
from console_log import alog, flush as flush_console
from attire_manager import AttireManagerError
from calendar_manager import CalendarError
from duration_manager import DurationError
from project_manager import ProjectManagerError
from prompt_log_engine import log_prompt, log_console, router as prompt_log_router
from config import (
    AGENT_API_KEY,
    ATTIRE_SUBAGENT_TIMEOUT_SECONDS,
    CORS_ALLOWED_ORIGINS,
    DELETE_FILE_ALLOWED_EXTENSIONS,
    DOCKER_CPU_COUNT,
    DOCKER_IMAGE,
    DOCKER_MEM_LIMIT,
    DOCKER_NETWORK_DISABLED,
    DOCKER_TIMEOUT_SECONDS,
    LLAMA_CONTEXT,
    LLAMA_SERVER_URL,
    MAX_TOOL_ITERATIONS,
    MEMORY_IDENTITY_SLOTS,
    SAFE_FILES_DIR,
    TOOL_SELECTION_ALWAYS_INCLUDE,
    TOOL_SELECTION_CONTEXT_CHAR_LIMIT,
    TOOL_SELECTION_ENABLED,
    TOOL_SELECTION_MIN_SCORE,
    TOOL_SELECTION_RESCUE_SCORE,
    TOOL_SELECTION_RESCUE_TOP_K,
    WRITE_FILE_ALLOWED_EXTENSIONS,
    WRITE_FILE_MAX_CHARS,
)

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Starts the calendar cache's background refresh thread once, when the
    # server actually comes up (not on every reload-triggered reimport -
    # see calendar_manager.start_background_refresh()'s own guard too).
    calendar_manager.start_background_refresh()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(prompt_log_router)


# ---------------------------------------------------------------------------
# Prompt-inspection logging, annotations, and the /prompt-log-viewer page
# now all live in prompt_log_engine.py - see the log_prompt/log_console
# imports above and app.include_router(prompt_log_router) below.
# ---------------------------------------------------------------------------


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

# Live testing also showed a second failure mode: after staging a change,
# the model sometimes responds to the user's plain "confirm" by calling the
# STAGE tool again (e.g. calendar_edit_event a second time) instead of
# calendar_confirm_pending - silently re-staging the identical change
# forever instead of ever actually applying it. Detected here in code
# rather than relying on the prompt instruction alone, which live testing
# showed isn't reliably followed. Only matches short, bare confirmations
# (e.g. "yes", "confirm") - a longer message is treated as a new request,
# not blocked, in case the user is actually asking to change something.
_CALENDAR_STAGE_TOOLS = {"calendar_create_event", "calendar_create_events_batch", "calendar_edit_event", "calendar_delete_event"}

# A stage tool and calendar_confirm_pending in the SAME response batch is
# unsafe even though this server executes calls sequentially within a
# batch (so confirm_pending would see the fresh stage): the user never
# actually saw the staged description before it got applied, which
# defeats the whole point of the two-step confirm safety model. Live
# testing showed this happen for real - only harmless that time because
# the paired stage call happened to fail (network timeout) first.
_CALENDAR_APPLY_TOOLS = {"calendar_confirm_pending"}
_BARE_CONFIRMATION_RE = re.compile(
    r"""^["'\s]*(yes|yeah|yep|sure|ok|okay|confirm|confirmed|go ahead|do it|correct|proceed)[.!]?["'\s]*$""",
    re.IGNORECASE,
)

# Short closing remarks never trigger select_tools()'s context-widening
# tier, even though they (correctly) score below TOOL_SELECTION_MIN_SCORE
# alone same as any other ambiguous message. Without this carve-out, "thanks!"
# right after a tool-heavy reply would fold that reply's tool-flavored
# language into the widened query and needlessly resurrect the group for a
# message that isn't actually asking for anything. Deliberately separate
# from _BARE_CONFIRMATION_RE above - overlapping word list, different job
# (that one gates calendar apply-on-confirm, not tool selection).
_CLOSING_REMARK_RE = re.compile(
    r"""^["'\s]*(thanks|thank you|thx|ty|cool|nice|great|perfect|sounds good|"""
    r"""got it|no worries|nvm|never ?mind|that'?s all|all good)[.,!]?["'\s]*$""",
    re.IGNORECASE,
)


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
    slot = args.get("slot") or None
    if not fact:
        return "Error: no fact provided."
    if slot and slot not in MEMORY_IDENTITY_SLOTS:
        return f"Error: '{slot}' is not a valid slot. Valid slots: {', '.join(MEMORY_IDENTITY_SLOTS)}."
    result = memory.save_memory(fact, slot=slot)
    if slot:
        verb = "Updated" if result["action"] == "updated" else "Saved"
        return f"{verb} slot '{slot}' [id: {result['id']}]: {fact}"
    msg = f"Saved to long-term memory [id: {result['id']}]: {fact}"
    similar = result.get("similar")
    if similar:
        msg += (
            f"\nNote: this looks similar to an existing memory - "
            f"[id: {similar['id']}] {similar['text']} (similarity {similar['score']:.2f}). "
            f"If it's the same fact restated, use update_memory or delete_memory instead of "
            f"leaving both stored."
        )
    return msg


def update_memory_tool(args: dict) -> str:
    memory_id = args.get("id", "")
    new_text = args.get("new_text", "")
    if not memory_id or not new_text:
        return "Error: both id and new_text are required."
    return memory.update_memory(memory_id, new_text)


def delete_memory_tool(args: dict) -> str:
    memory_id = args.get("id", "")
    if not memory_id:
        return "Error: id is required."
    return memory.delete_memory(memory_id)


def pin_memory_tool(args: dict) -> str:
    memory_id = args.get("id", "")
    if not memory_id:
        return "Error: id is required."
    return memory.pin_memory(memory_id)


def unpin_memory_tool(args: dict) -> str:
    memory_id = args.get("id", "")
    if not memory_id:
        return "Error: id is required."
    return memory.unpin_memory(memory_id)


def list_memories_tool(args: dict) -> str:
    memories = memory.list_memories()
    if not memories:
        return "No memories saved yet."
    lines = []
    for m in memories:
        tag = f" (slot: {m['slot']})" if m.get("slot") else (" (pinned)" if m.get("pinned") else "")
        lines.append(f"[id: {m['id']}]{tag} {m['text']}")
    return "\n".join(lines)


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


def calendar_create_events_batch(args: dict) -> str:
    try:
        return calendar_manager.stage_create_events_batch(
            args.get("events", []), args.get("calendar_name")
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


def calendar_check_availability(args: dict) -> str:
    try:
        result = calendar_manager.check_availability(
            args.get("start"), args.get("end"), args.get("when")
        )
    except CalendarError as e:
        return f"Error: {e}"

    cache_note = (
        " (answered from the cached calendar - may be a few minutes out "
        "of date for very recent changes)"
        if result["source"] == "cache" else ""
    )
    if result["free"]:
        return f"Free - no conflicts found.{cache_note}"

    lines = [f"NOT free - conflicts found{cache_note}:"]
    for c in result["conflicts"]:
        lines.append(f"- {c['title']}: {c['start']} - {c['end']}")
    return "\n".join(lines)


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
        task = project_manager.create_task(
            project["id"], args.get("title", ""), notes=args.get("notes", "")
        )
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
        updated, flag = project_manager.set_task_status(project["id"], task["id"], status)
        message = f"Set {updated['short_id']} (\u201c{updated['title']}\u201d) to {status}."
        if flag:
            message += "\n\n" + flag
        return message
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
        applied, flags = project_manager.set_all_tasks_status(project["id"], status)
        if not applied:
            return f"No change needed; every non-archived task in {project['short_code']} \u2014 {project['name']} is already {status}."
        message = f"Set {len(applied)} task(s) in {project['short_code']} to {status}:\n" + "\n".join(applied)
        if flags:
            message += "\n\n" + "\n".join(flags)
        return message
    except ProjectManagerError as e:
        return f"Error: {e}"


def project_manager_batch_update(args: dict) -> str:
    try:
        state = project_manager._load()
        project = project_manager.resolve_project(state, args.get("project"))
        applied, flags = project_manager.batch_update(project["id"], args.get("operations", []))
        if not applied:
            return "No change needed; every requested batch operation is already satisfied."
        message = f"Applied {len(applied)} change(s) to {project['short_code']}:\n" + "\n".join(applied)
        if flags:
            message += "\n\n" + "\n".join(flags)
        return message
    except ProjectManagerError as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Duration tracking tools (duration_manager.py). Task completions log a
# duration anchor automatically (see project_manager._apply_task_status) -
# these three tools are the model-facing surface for reading estimates and
# handling corrections/new categories.
# ---------------------------------------------------------------------------

def duration_get_estimate(args: dict) -> str:
    result = duration_manager.get_estimate(args.get("query", ""))
    return json.dumps(result, ensure_ascii=False)


def duration_correct_entry(args: dict) -> str:
    try:
        return duration_manager.correct_entry(args.get("task", ""), args.get("value", ""))
    except DurationError as e:
        return f"Error: {e}"


def duration_confirm_new_category(args: dict) -> str:
    try:
        return duration_manager.confirm_new_category(args.get("name", ""))
    except DurationError as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Attire tracking tools (attire_manager.py). One global record per character
# (not per chat), structured slots only, current state only - see
# attire_manager.py's module docstring for the full design rationale.
# ---------------------------------------------------------------------------

def attire_manager_add_item(args: dict) -> str:
    try:
        record, added = attire_manager.add_item(
            args.get("character_name", ""),
            args.get("slot", ""),
            args.get("item", ""),
        )
        if not added:
            return f"No change needed - {record['name']}'s {args.get('slot')} already includes that."
        return f"Added to {record['name']}'s {args.get('slot')}."
    except AttireManagerError as e:
        return f"Error: {e}"


def attire_manager_remove_item(args: dict) -> str:
    try:
        record, removed = attire_manager.remove_item(
            args.get("character_name", ""),
            args.get("slot", ""),
            args.get("item_hint", ""),
        )
        if removed is None:
            return (
                f"No change made - couldn't confidently match '{args.get('item_hint')}' "
                f"to exactly one item in {record['name']}'s {args.get('slot')}."
            )
        return f"Removed '{removed}' from {record['name']}'s {args.get('slot')}."
    except AttireManagerError as e:
        return f"Error: {e}"


def attire_manager_replace_slot(args: dict) -> str:
    try:
        record, changed = attire_manager.replace_slot(
            args.get("character_name", ""),
            args.get("slot", ""),
            args.get("items", ""),
        )
        if not changed:
            return f"No change needed - {record['name']}'s {args.get('slot')} already matches that."
        return f"Replaced {record['name']}'s {args.get('slot')}."
    except AttireManagerError as e:
        return f"Error: {e}"


def attire_manager_get(args: dict) -> str:
    return attire_manager.get_attire_text(args.get("character_name", ""))


# Holds the currently-running (or just-finished) post-turn attire pass, if
# any. Set by _spawn_attire_subagent() at the end of a turn; consumed
# (awaited-with-timeout, then cleared) at the top of the NEXT chat_completions
# call. Single-user local server - one slot is enough, no queue needed.
_attire_subagent_task: asyncio.Task | None = None


def _spawn_attire_subagent(user_text: str, assistant_text: str) -> None:
    """Fire-and-forget: kicks off the background attire pass and stores the
    task so the next turn can wait on it if it isn't done yet."""
    global _attire_subagent_task
    _attire_subagent_task = asyncio.create_task(
        attire_subagent.run_attire_subagent(user_text, assistant_text)
    )


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
            "description": "Save a new, durable fact about the user (preferences, ongoing projects, etc.) for future recall. Skip trivial small talk. If this corrects/replaces an existing memory, use update_memory instead. For identity (name+pronouns, one combined slot), occupation, or location, pass `slot` - makes it always-visible every turn instead of only when relevant, and overwrites that slot instead of duplicating. For the `identity` slot, always include BOTH name and pronouns even if only one changed - it's one field, so a partial update silently drops whichever part you omit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember, written as a standalone sentence."},
                    "slot": {
                        "type": "string",
                        "enum": list(MEMORY_IDENTITY_SLOTS),
                        "description": "Optional. Set only when the fact is the user's identity (name and/or pronouns), occupation, or location - upserts into that slot instead of creating a new freeform memory. Omit for anything else.",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Overwrite an existing memory in place when the user corrects or changes a previously-stored fact (e.g. 'I'm actually a nurse now, not a teacher'). Requires the exact id of the memory being replaced - get it from a memory shown in the 'Relevant things you remember' context, or from list_memories/search_memories if you don't already have it. Never guess an id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The id of the existing memory to overwrite."},
                    "new_text": {"type": "string", "description": "The corrected fact, written as a standalone sentence."}
                },
                "required": ["id", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "Permanently remove a memory when the user asks you to forget something, with no replacement fact. Requires the exact id - get it from context, list_memories, or search_memories. Never guess an id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The id of the memory to delete."}
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List every fact currently saved in long-term memory, each with its id. Use this to find the id of a memory to update or delete when it wasn't already shown in context, or when the user asks what you remember about them.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pin_memory",
            "description": "Mark an existing freeform memory (one saved without a slot) as always-shown, so it appears in every future conversation instead of only when it's semantically relevant to what's being discussed. Use for facts worth always knowing that don't fit the fixed identity/occupation/location slots (e.g. a standing dietary restriction or strong preference). There's a cap on how many freeform memories can be pinned at once - if you hit it, the tool result will say so.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The id of the existing freeform memory to pin."}
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unpin_memory",
            "description": "Undo pin_memory - the memory goes back to only surfacing when it's semantically relevant, instead of always. Has no effect on slotted memories (identity/occupation/location are always shown by design and can't be unpinned - use delete_memory to remove one of those instead).",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The id of the pinned freeform memory to unpin."}
                },
                "required": ["id"],
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
            "description": "List events of the default calendar in a date range (or a specific calendar if calendar_name is given). Use for questions like 'what's on my calendar' or 'what do I have this week'. Defaults to the next 14 days if no range is given. Read-only, executes immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start of range, 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'. Defaults to now. For a single specific day, pass the corresponding date for start and the date for the next day for end - that returns the whole day."},
                    "end": {"type": "string", "description": "End of range, same format. Defaults to 14 days after start."},
                    "calendar_name": {"type": "string", "description": "Optional: a specific named calendar."},
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
                    "calendar_name": {"type": "string", "description": "Optional: a specific named calendar."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Stage creating a new REAL event on the user's iCloud calendar (an actual appointment, meeting, or plan - not an in-character/roleplay scheduled action). See the calendar write convention in your system context for the confirm step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title."},
                    "start": {"type": "string", "description": "Start time, 'YYYY-MM-DD HH:MM'."},
                    "end": {"type": "string", "description": "End time, same format. Defaults to 1 hour after start."},
                    "location": {"type": "string", "description": "Optional location."},
                    "description": {"type": "string", "description": "Optional notes/description."},
                    "calendar_name": {"type": "string", "description": "Optional: a specific named calendar."},
                },
                "required": ["title", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_events_batch",
            "description": (
                "Stage MULTIPLE new calendar events at once as a SINGLE pending change (e.g. a "
                "proposed schedule of task blocks) - use instead of repeated calendar_create_event "
                "calls; calendar_confirm_pending then creates all of them in one go. Validates "
                "against existing events and against each other, rejecting the whole batch with a "
                "clear reason if anything overlaps, so you can adjust and retry. See the calendar "
                "write convention in your system context for the confirm step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array", "minItems": 1, "maxItems": 12,
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "start": {"type": "string", "description": "'YYYY-MM-DD HH:MM'."},
                                "end": {"type": "string", "description": "'YYYY-MM-DD HH:MM'. Optional - defaults to a 30-minute block."},
                                "location": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["title", "start"],
                        },
                    },
                    "calendar_name": {"type": "string", "description": "Optional: a specific named calendar."},
                },
                "required": ["events"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_edit_event",
            "description": "Stage editing an existing iCloud calendar event by its UID (from calendar_list_events/calendar_search_events - never guess a UID). See the calendar write convention in your system context for the confirm step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_uid": {"type": "string", "description": "Exact UID of the event to edit, from a prior list/search result."},
                    "title": {"type": "string", "description": "New title, if changing it."},
                    "start": {"type": "string", "description": "New start time 'YYYY-MM-DD HH:MM', if changing it."},
                    "end": {"type": "string", "description": "New end time, if changing it."},
                    "location": {"type": "string", "description": "New location, if changing it."},
                    "description": {"type": "string", "description": "New description/notes, if changing it."},
                    "calendar_name": {"type": "string", "description": "Optional: a specific named calendar."},
                },
                "required": ["event_uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_delete_event",
            "description": "Stage permanently deleting an existing iCloud calendar event by its UID (from calendar_list_events/calendar_search_events - never guess a UID). See the calendar write convention in your system context for the confirm step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_uid": {"type": "string", "description": "Exact UID of the event to delete, from a prior list/search result."},
                    "calendar_name": {"type": "string", "description": "Optional: a specific named calendar."},
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
            "name": "calendar_check_availability",
            "description": "Read-only: check whether a time range is free or conflicts with something on any of the user's calendars. Never stages or changes anything. Ranges in roughly the next two weeks typically answer from a background cache that refreshes every few minutes, so the answer may be a little stale for very recent changes; ranges further out do a live calendar lookup instead. Give either a 'when' phrase (e.g. 'tomorrow', 'next Friday') or explicit start/end - not both.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start of the range to check, e.g. '2026-08-20 14:00'. Omit if using 'when'."},
                    "end": {"type": "string", "description": "End of the range to check. Omit to default to 1 hour after start."},
                    "when": {"type": "string", "description": "Natural date phrase like 'tomorrow' or 'next Friday', checked as a full day. Mutually exclusive with start/end."},
                },
                "required": [],
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
                    "project": {"type": "string", "description": "Project ID, short code, or name. Omit to use the focused project."},
                    "title": {"type": "string", "description": "Short, concrete, verb-led task title."},
                    "notes": {"type": "string", "description": "Optional. To record a duration estimate, effort level, or preferred time window so they show up automatically next to the task, put recognized tag lines at the very top, one per line: 'dur: 45m' (also '1h', '1h30m', '90'), 'effort: low'/'medium'/'high', 'when: morning'/'afternoon'/'evening' (optionally + 'weekday'/'weekend'). Any text after the tag lines is kept as freeform notes."},
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
                    "project": {"type": "string", "description": "Project ID, short code, or name. Omit to use the focused project."},
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
            "description": "Replace, append to, or clear the notes of an existing task when the user explicitly asks. Use append for additional context and replace only when the user wants existing notes overwritten. To record a duration estimate, effort level, or preferred time window so they show up automatically next to the task (not just when asked), put recognized tag lines at the very top of the text, one per line: 'dur: 45m' (also accepts '1h', '1h30m', '90'), 'effort: low' / 'effort: medium' / 'effort: high', and 'when: morning' / 'afternoon' / 'evening', optionally followed by 'weekday' or 'weekend' (e.g. 'when: afternoon weekend'). Any text after the tag lines is kept as freeform notes. Appending new tag lines only updates those specific tags and leaves the rest of the note (including other existing tags) intact - no need to replace the whole note to change one tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project ID, short code, or name. Omit to use the focused project."},
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
                    "project": {"type": "string", "description": "Project ID, short code, or name. Omit to use the focused project."},
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
                    "project": {"type": "string", "description": "Project ID, short code, or name. Omit to use the focused project."},
                    "operations": {
                        "type": "array", "minItems": 1, "maxItems": 25,
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["create_task", "update_task_status", "update_task_notes"]},
                                "title": {"type": "string", "description": "For create_task."},
                                "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                                "notes": {"type": "string", "description": "For create_task. Optional tag lines at the top ('dur: 45m', 'effort: medium', 'when: afternoon weekend') show up automatically next to the task - see project_manager_update_task_notes for the exact syntax."},
                                "task": {"type": "string", "description": "For status or note updates: task ID, short ID, or unambiguous title."},
                                "status": {"type": "string", "enum": ["pending", "active", "blocked", "done", "cancelled"]},
                                "mode": {"type": "string", "enum": ["replace", "append", "clear"]},
                                "text": {"type": "string", "description": "For update_task_notes. Same 'dur:'/'effort:'/'when:' tag syntax as project_manager_update_task_notes applies here."},
                            },
                            "required": ["type"],
                        },
                    },
                },
                "required": ["operations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duration_get_estimate",
            "description": (
                "Get a data-grounded duration estimate for a task/category, based on the "
                "user's own logged history - use this INSTEAD of guessing a duration yourself "
                "whenever the user asks how long something will take. Returns JSON with "
                "'resolved' (false if no matching category exists yet), and if resolved: "
                "'confidence' (insufficient/rough/confident), 'median_minutes', 'mad_minutes', "
                "and 'n' (entry count). If confidence is 'insufficient' or not resolved, tell "
                "the user there isn't enough personal history yet rather than inventing a number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The task or category to estimate, e.g. 'writing a report' or 'email'."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duration_correct_entry",
            "description": (
                "Correct the most recently logged duration for a task or category, when the "
                "user says the auto-logged time was off (e.g. 'that email actually took like "
                "20 minutes'). Loose shorthand is fine for value ('~20min', '1h', '90')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task title or category to identify which entry to correct. Omit to correct the single most recent entry."},
                    "value": {"type": "string", "description": "The corrected duration, loose shorthand OK, e.g. '~20min', '1.5h', '90'."},
                },
                "required": ["value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duration_confirm_new_category",
            "description": (
                "Confirm creating a new tracked duration category, ONLY after you asked the "
                "user (following an 'uncategorized' duration-logging flag) and they explicitly "
                "agreed to a specific category name. Never call this speculatively or without "
                "an explicit yes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The new category name, in the user's own words is fine."}
                },
                "required": ["name"],
            },
        },
    },
]

# Extracted (not inlined into TOOLS above) so attire_subagent.py's separate
# one-shot completion call can reuse these exact schemas instead of a
# second hand-copied version that could silently drift from this one.
ATTIRE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "attire_manager_update",
            "description": (
                "Record a change in the character's attire. Call this the moment the narrative describes ANY change to what a character is wearing"
                " - full outfit changes as well as subtle/partial ones (loosening or removing a tie, unbuttoning a shirt, taking off shoes or an accessory,"
                " a jacket coming off, one item being swapped for another). Only pass the slot(s) that changed; omit everything else. Pass an empty string for"
                " a slot to mean it is now bare/nothing (e.g. shoes removed). If a slot already describes one or more items (e.g. feet: 'white socks'), and an "
                "item is added (e.g. 'black sneakers are put on feet'), you MUST restate the FULL corrected value for that slot (e.g. feet: 'black sneakers and white socks')."
                " If a slot shown above already describes more than one item (e.g. feet: 'black sneakers and white socks'), and only PART of that changes (e.g. just the socks come off),"
                " you MUST restate the FULL corrected value for that slot (feet: 'black sneakers')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "The character whose attire changed, exactly as you refer to them."},
                    "head": {"type": "string", "description": "Headwear, e.g. 'wide-brimmed hat'. Omit if unchanged; empty string to clear."},
                    "top": {"type": "string", "description": "Upper-body garment, e.g. 'black leather jacket'. Omit if unchanged; empty string to clear."},
                    "bottom": {"type": "string", "description": "Lower-body garment, e.g. 'ripped jeans'. Omit if unchanged; empty string to clear."},
                    "feet": {"type": "string", "description": "Footwear, e.g. 'combat boots'. Omit if unchanged; empty string to clear."},
                    "accessories": {"type": "string", "description": "Comma-separated list of accessories, e.g. 'silver necklace, fingerless gloves'. Omit if unchanged; empty string to clear all."},
                },
                "required": ["character_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attire_manager_get",
            "description": (
                "Look up a character's current attire. Usually unnecessary since a tracked "
                "character's current attire is already shown to you automatically at the top "
                "of the conversation - use this only if that wasn't shown (e.g. a brand-new "
                "character never updated before) or you need to double-check before narrating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "The character to look up, exactly as you refer to them."},
                },
                "required": ["character_name"],
            },
        },
    },
]
# Deliberately NOT merged into TOOLS: attire tracking is handled entirely
# by the post-turn attire_subagent.py pass, not by the main agent
# mid-conversation. ATTIRE_TOOL_SCHEMAS stays defined here purely so
# attire_subagent.py can import and reuse these exact schemas for its own
# separate completion call - never added to what the main agent sees.
#
# v2 (see brainstorm-layered-clothing.md): three verbs instead of one
# full-value update, so adding a layered item structurally cannot erase
# anything else already in the slot - see attire_manager.py's docstring.
ATTIRE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "attire_add_item",
            "description": (
                "Add one item to a slot WITHOUT touching anything else already there. Use this "
                "whenever something is put on, layered, or added - e.g. shoes going on over socks, "
                "a jacket going on over a shirt, a ring being put on. Never use this to describe an "
                "item being removed or swapped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "The character whose attire changed, exactly as you refer to them."},
                    "slot": {"type": "string", "enum": list(attire_manager.ATTIRE_SLOTS), "description": "Which slot the item goes in."},
                    "item": {"type": "string", "description": "The item being added, e.g. 'black leather jacket'."},
                },
                "required": ["character_name", "slot", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attire_remove_item",
            "description": (
                "Remove one item from a slot. Use this when the narrative describes an item coming "
                "off - taken off, removed, shrugged off, kicked off, etc. Only removes the ONE item "
                "described; anything else in that slot is left untouched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "The character whose attire changed, exactly as you refer to them."},
                    "slot": {"type": "string", "enum": list(attire_manager.ATTIRE_SLOTS), "description": "Which slot to remove from."},
                    "item_hint": {"type": "string", "description": "The item being removed, as described in the narrative, e.g. 'the jacket' or 'her shoes'."},
                },
                "required": ["character_name", "slot", "item_hint"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attire_replace_slot",
            "description": (
                "Wipe a slot and set it to a brand new value. ONLY use this for a genuine full "
                "change - e.g. a character changes into a whole new outfit, or the scene explicitly "
                "resets what someone is wearing. Do NOT use this to describe a single item coming "
                "on or off - that silently deletes anything else in the slot. Use attire_add_item "
                "or attire_remove_item for anything short of a full change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "The character whose attire changed, exactly as you refer to them."},
                    "slot": {"type": "string", "enum": list(attire_manager.ATTIRE_SLOTS), "description": "Which slot to replace."},
                    "items": {"type": "string", "description": "Comma-separated full new contents of the slot, e.g. 'red sundress, sun hat'. Empty string clears it entirely."},
                },
                "required": ["character_name", "slot", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "attire_manager_get",
            "description": (
                "Look up a character's current attire. Usually unnecessary since a tracked "
                "character's current attire is already shown to you automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {"type": "string", "description": "The character to look up, exactly as you refer to them."},
                },
                "required": ["character_name"],
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
    "update_memory": update_memory_tool,
    "delete_memory": delete_memory_tool,
    "pin_memory": pin_memory_tool,
    "unpin_memory": unpin_memory_tool,
    "list_memories": list_memories_tool,
    "search_documents": search_documents_tool,
    "write_file": write_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "calendar_list_calendars": calendar_list_calendars,
    "calendar_list_events": calendar_list_events,
    "calendar_search_events": calendar_search_events,
    "calendar_create_event": calendar_create_event,
    "calendar_create_events_batch": calendar_create_events_batch,
    "calendar_edit_event": calendar_edit_event,
    "calendar_delete_event": calendar_delete_event,
    "calendar_confirm_pending": calendar_confirm_pending,
    "calendar_cancel_pending": calendar_cancel_pending,
    "calendar_check_availability": calendar_check_availability,
    "project_manager_get_overview": project_manager_get_overview,
    "project_manager_create_task": project_manager_create_task,
    "project_manager_update_task_status": project_manager_update_task_status,
    "project_manager_update_task_notes": project_manager_update_task_notes,
    "project_manager_set_all_tasks_status": project_manager_set_all_tasks_status,
    "project_manager_batch_update": project_manager_batch_update,
    "duration_get_estimate": duration_get_estimate,
    "duration_correct_entry": duration_correct_entry,
    "duration_confirm_new_category": duration_confirm_new_category,
}

# ---------------------------------------------------------------------------
# Dynamic tool selection: instead of sending all tool schemas + all usage
# instructions on every request, embed each tool once at startup and score
# it against the user's latest message per request. Only tools that clear
# TOOL_SELECTION_MIN_SCORE (plus a small always-on core set, plus anything
# force-included for state reasons, e.g. a pending calendar change) get
# sent. Falls back to sending everything if too few tools match - this can
# only ever make a request SMALLER when confident, never break a request
# that would have worked before this feature existed.
# ---------------------------------------------------------------------------

TOOL_GROUPS = {
    "get_current_time": "utility", "calculate": "utility", "web_search": "utility",
    "get_weather": "utility", "run_python": "utility",
    "list_files": "files", "read_file": "files", "write_file": "files",
    "edit_file": "files", "delete_file": "files", "search_documents": "files",
    "save_memory": "memory", "update_memory": "memory", "delete_memory": "memory",
    "list_memories": "memory", "pin_memory": "memory", "unpin_memory": "memory",
    "calendar_list_calendars": "calendar", "calendar_list_events": "calendar",
    "calendar_search_events": "calendar", "calendar_create_event": "calendar",
    "calendar_create_events_batch": "calendar", "calendar_edit_event": "calendar",
    "calendar_delete_event": "calendar", "calendar_confirm_pending": "calendar",
    "calendar_cancel_pending": "calendar", "calendar_check_availability": "calendar",
    "project_manager_get_overview": "project", "project_manager_create_task": "project",
    "project_manager_update_task_status": "project", "project_manager_update_task_notes": "project",
    "project_manager_set_all_tasks_status": "project", "project_manager_batch_update": "project",
    "duration_get_estimate": "duration", "duration_correct_entry": "duration",
    "duration_confirm_new_category": "duration",
}
# Stable order so the assembled instruction text reads the same way (and
# hits the same prompt-cache prefix) whenever the same group set is chosen.
_GROUP_ORDER = ["utility", "files", "memory", "calendar", "project", "duration"]

GROUP_INSTRUCTIONS = {
    "utility": (
        "For weather/temperature questions, always use get_weather, never "
        "web_search. Use calculate for simple arithmetic, or run_python for "
        "anything needing actual code logic."
    ),
    "files": (
        "Use search_documents (not read_file) when looking for specific "
        "information inside long or multiple documents. Use write_file when "
        "the user asks you to create, save, write out, or update a .txt or "
        ".md file - use mode 'overwrite' to replace a file's contents (or "
        "create a new one) and mode 'append' to add to the end of an "
        "existing file without erasing it. Use edit_file for a small "
        "targeted change inside an existing file instead of rewriting the "
        "whole thing with write_file. Only call delete_file when the user "
        "clearly and explicitly asks to delete a specific named file - "
        "never as a side effect of another request, and never guess the "
        "filename if it's ambiguous; ask the user to confirm instead."
    ),
    "memory": (
        "Call save_memory when the user shares a durable fact about "
        "themselves worth remembering - not for small talk. If the fact is "
        "their identity (name and/or pronouns - combined into one slot), "
        "occupation, or location, always pass the matching `slot` argument "
        "rather than leaving it plain - this makes it always visible to you "
        "in every conversation, not just when it happens to match what's "
        "being discussed, and safely overwrites the old value instead of "
        "creating a duplicate if that slot is already filled. If the user "
        "corrects or changes a fact you already remember about them (e.g. a "
        "job, name, or preference that's now different) and it's NOT one of "
        "the three slots, call update_memory with that memory's id and the "
        "corrected text - do NOT call save_memory again, since that would "
        "leave both the old and new fact stored side by side and confuse "
        "future recall. The id is usually already visible in the '[id: "
        "...]' tag next to a fact shown to you in the 'Core facts you "
        "always know about this user' or 'Relevant things you remember "
        "about this user' context; only call list_memories or "
        "search_memories to look one up if it isn't already visible. Never "
        "guess an id. Call delete_memory (with an id, same rule) only when "
        "the user explicitly asks you to forget something, with no "
        "replacement fact - this also works on a slotted memory, which just "
        "empties that slot. If a save_memory result includes a note that "
        "the new fact looks similar to an existing memory, check whether "
        "it's really the same fact restated - if so, use update_memory or "
        "delete_memory to reconcile them instead of leaving both. Use "
        "pin_memory on a freeform memory (one saved without a slot) when a "
        "fact is worth always knowing but doesn't fit the three fixed slots "
        "(e.g. a standing dietary restriction or strong preference) - "
        "pinned memories, like slots, are always shown to you rather than "
        "only when relevant. Use unpin_memory to undo that; it has no "
        "effect on slotted memories, which are always shown by design - use "
        "delete_memory on those instead."
    ),
    "calendar": (
        "Use calendar_list_events or calendar_search_events freely to read "
        "the user's iCloud calendar - these are read-only, need no "
        "confirmation, and search across all of the user's calendars by "
        "default. If the user asks what calendars they have, or expected "
        "events aren't showing up, call calendar_list_calendars to see the "
        "actual calendar names. "
        "CRITICAL: a system message near the top of this conversation, "
        "labeled [CURRENT DATE/TIME], gives you today's real date and time "
        "on every single request - use it to compute any relative date "
        "phrase ('this week', 'today', 'tomorrow', 'next month', etc.) for "
        "calendar_list_events, calendar_search_events, calendar_create_event, "
        "and calendar_edit_event. You do NOT need to call get_current_time "
        "for this - the date is already provided fresh every turn. NEVER "
        "guess or assume a date/year from memory or training data for a "
        "calendar call - a wrong year will silently return the wrong "
        "(usually empty) results instead of erroring, so this mistake is "
        "easy to make and easy to miss. If you don't need a specific range, "
        "you may also omit start/end entirely and let the tool default to "
        "today onward. "
        "calendar_create_event, calendar_edit_event, and "
        "calendar_delete_event NEVER change the real calendar by themselves "
        "- they only stage a proposed change and return a description of "
        "it. After calling one, tell the user exactly what will happen and "
        "wait for their reply. Only call calendar_confirm_pending as your "
        "very next tool call if the user's following message clearly and "
        "explicitly confirms (e.g. 'yes', 'confirm', 'go ahead') - never "
        "call it speculatively, preemptively, or on the same turn as "
        "staging the change. If the user declines or wants something "
        "different, call calendar_cancel_pending instead. Always look up an "
        "event's UID with calendar_list_events or calendar_search_events "
        "before editing or deleting it - never guess a UID."
    ),
    "project": (
        "Use the project_manager_* tools only when the user clearly states "
        "a concrete project/task action (create, start, block, complete, "
        "cancel, or annotate a task) - never from hypotheticals or vague "
        "wishes. Call project_manager_get_overview first if which project "
        "or task is meant isn't already clear from the persistent project "
        "state below. When one message contains several concrete changes "
        "for the same project, use project_manager_batch_update once "
        "instead of separate calls; for 'mark everything as done'-style "
        "requests, use project_manager_set_all_tasks_status once instead of "
        "enumerating tasks. Project-manager changes apply immediately - "
        "there is no separate confirmation step, so only call these tools "
        "when the user's intent is unambiguous."
    ),
    "duration": (
        "Use duration_get_estimate whenever the user asks how long a "
        "task/category will take - never guess a duration yourself, since "
        "the whole point of this tool is grounding the answer in the "
        "user's own logged history instead of a generic guess. If "
        "confidence comes back 'insufficient' or resolved is false, say so "
        "plainly instead of presenting a number anyway. Task completions "
        "automatically log a rough duration anchor in the background - you "
        "don't need to call any tool for that. But if the tool result from "
        "marking a task done includes a duration line (e.g. '~N min logged "
        "for ... category: ...'), always relay that line to the user in "
        "your reply, verbatim or close to it - don't silently drop it. This "
        "matters most when the category is 'uncategorized', since that "
        "line is asking the user whether to create a tracked category for "
        "it. If the user corrects a duration you just reported, or "
        "references a past task's duration being wrong, use "
        "duration_correct_entry. Only call duration_confirm_new_category "
        "after the user explicitly agrees to a specific new category name "
        "you proposed following an 'uncategorized' flag - never on your "
        "own initiative."
    ),
}

_TOOL_INSTRUCTION_CLOSING = (
    "IMPORTANT for multi-step questions: if answering fully requires "
    "several pieces of information, call tools one at a time in sequence, "
    "using each result to decide your next step, before giving your final "
    "answer. Do not stop after one tool call if the question isn't fully "
    "answered yet. Never guess or invent facts, dates, statistics, or "
    "search results that a tool could actually check for you. If a tool "
    "returns no useful result, say so honestly instead of making something "
    "up. If a tool result begins with 'Error:', the tool call FAILED - you "
    "must tell the user it failed and relay the reason (e.g. 'Docker isn't "
    "running'). Never substitute your own guessed numbers, facts, or "
    "output in place of a failed tool's result, even if you present it as "
    "an example or hypothetical. When you decide to call a tool, call it "
    "directly - do not write any explanation, plan, or commentary before "
    "or alongside the tool call. Save your explanation, if any, for your "
    "final answer after the tool result comes back."
)

# Built once at server startup: one embedding per tool, from "name: description".
_TOOL_EMBEDDINGS = {
    t["function"]["name"]: np.array(
        memory.embed(f'{t["function"]["name"]}: {t["function"].get("description", "")}')
    )
    for t in TOOLS
}


def _expand_to_groups(names: set[str]) -> set[str]:
    """Given tool names that individually cleared a confidence threshold,
    expand to every tool sharing a TOOL_GROUPS bucket with any of them.
    ...
    Deliberately NOT applied to the rescue tier (see select_tools) - rescue
    exists for weak/ambiguous signals and stays narrow on purpose, or it
    would reintroduce the "chit-chat drags in a whole tool group" problem
    the tiered design exists to prevent.
    """
    groups_hit = {TOOL_GROUPS[n] for n in names if n in TOOL_GROUPS}
    if not groups_hit:
        return set(names)
    return set(names) | {n for n, g in TOOL_GROUPS.items() if g in groups_hit}

def select_tools(user_text: str, force_names: set | None = None, prior_assistant_text: str = ""):
    """Return (selected_tools, scores, tier).

    Three-tier matching:
      1. "direct"           - tools clearing TOOL_SELECTION_MIN_SCORE
                               against the current message alone. Unchanged
                               from before this session.
      2. "context_widened"  - only tried when (1) finds nothing AND the
                               current message isn't a bare closing remark
                               (_CLOSING_REMARK_RE). Re-scores [prior
                               assistant reply + current message] against
                               the same MIN_SCORE. Rescues elliptical
                               follow-ups ("and also change the time to
                               3pm") that carry no tool-relevant signal on
                               their own but clearly continue a
                               tool-relevant prior turn - without this,
                               such messages fell all the way to
                               core-only tools and sent the model into a
                               tool-call loop with nothing useful to call.
      3. "rescue"/"core_only" - existing loose-threshold top-K fallback,
                               unchanged from before this session. Fires
                               when even the widened query finds nothing -
                               genuine plain chit-chat, or a closing remark
                               that never reached tier 2 at all.
    """
    force_names = set(force_names or ())

    if not TOOL_SELECTION_ENABLED:
        return TOOLS, {}, "disabled"

    text = (user_text or "").strip()
    if not text:
        selected_names = set(TOOL_SELECTION_ALWAYS_INCLUDE) | force_names
        selected_tools = [t for t in TOOLS if t["function"]["name"] in selected_names]
        return selected_tools, {}, "empty"

    query_vec = np.array(memory.embed(text))
    scores = {name: float(query_vec @ vec) for name, vec in _TOOL_EMBEDDINGS.items()}

    confident = {n for n, s in scores.items() if s >= TOOL_SELECTION_MIN_SCORE}

    if confident:
        selected_names = _expand_to_groups(confident)   # was: selected_names = confident
        tier = "direct"
    else:
        widened = set()
        prior_text = (prior_assistant_text or "").strip()
        if prior_text and not _CLOSING_REMARK_RE.match(text):
            # Tail, not head - the embedder's own truncation on a long
            # string keeps the start, but the part of a prior reply most
            # relevant to a follow-up is usually its end (e.g. a trailing
            # clarifying question). See config.py's comment on this constant.
            prior_tail = prior_text[-TOOL_SELECTION_CONTEXT_CHAR_LIMIT:]
            combined_vec = np.array(memory.embed(f"{prior_tail} {text}"))
            combined_scores = {name: float(combined_vec @ vec) for name, vec in _TOOL_EMBEDDINGS.items()}
            widened = {n for n, s in combined_scores.items() if s >= TOOL_SELECTION_MIN_SCORE}
            if widened:
                # Debug log reflects the query that actually decided this,
                # not the (lower, inconclusive) single-message scores.
                scores = combined_scores

        if widened:
            selected_names = _expand_to_groups(widened)  # was: selected_names = widened
            tier = "context_widened"
        else:
            # Nothing confidently matched, even widened - don't fall back
            # to everything. Rescue only the handful of tools that are at
            # least weakly related, in case this is an ambiguous real
            # request rather than genuine chit-chat (which correctly
            # rescues nothing at all).
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            selected_names = {
                n for n, s in ranked[:TOOL_SELECTION_RESCUE_TOP_K]
                if s >= TOOL_SELECTION_RESCUE_SCORE
            }
            tier = "rescue" if selected_names else "core_only"

    selected_names |= set(TOOL_SELECTION_ALWAYS_INCLUDE)
    selected_names |= force_names

    selected_tools = [t for t in TOOLS if t["function"]["name"] in selected_names]
    return selected_tools, scores, tier


def build_tool_instruction(selected_tools: list, client_tool_names: set) -> str:
    """Assemble the system-message instruction text from only the groups
    actually represented in selected_tools, instead of always including
    every group's usage rules regardless of which tools are even present."""
    names = [t["function"]["name"] for t in selected_tools]
    groups_present = {TOOL_GROUPS.get(n) for n in names}

    parts = [
        f"You have tools available: {', '.join(names)}. You MUST call the "
        "relevant tool whenever the user asks about current events, "
        "real-time facts, dates/times, weather, exact arithmetic, or "
        "anything you are not fully certain of from memory."
    ]
    for group in _GROUP_ORDER:
        if group in groups_present:
            parts.append(GROUP_INSTRUCTIONS[group])
    if client_tool_names:
        parts.append(
            "Additional tools provided by the connected frontend are also "
            f"available this turn: {', '.join(sorted(client_tool_names))}. "
            "Call them normally, following their own descriptions, and "
            "only when they clearly fit the user's request."
        )
    parts.append(_TOOL_INSTRUCTION_CLOSING)
    return " ".join(parts)


# ---------------------------------------------------------------------------

def verify_api_key(authorization: str = Header(default="")):
    """
    Gatekeeper for the two SillyTavern-facing endpoints. SillyTavern's
    Custom OpenAI-compatible connection sends its "API Key" field as a
    standard 'Authorization: Bearer <key>' header - compared here against
    AGENT_API_KEY from .env. Deliberately NOT applied to /projects (see
    handover-21) - only the model-facing chat endpoints are gated for now.
    If AGENT_API_KEY is unset, auth is skipped with a console warning
    instead of hard-failing, so a fresh clone still boots.
    """
    if not AGENT_API_KEY:
        print("[AUTH] WARNING: AGENT_API_KEY not set in .env - endpoint is unprotected.")
        return
    if authorization != f"Bearer {AGENT_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {AGENT_API_KEY}"}) as client:
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


# ---------------------------------------------------------------------------
# Memory browser HTTP API - plain REST for the /memory-browser page. Same
# pattern as /projects above: the UI drives the exact same memory.py
# functions the model's tools use, so they can never disagree about state.
# ---------------------------------------------------------------------------

@app.get("/memories")
async def api_list_memories():
    memories = await asyncio.to_thread(memory.list_memories_full)
    return {"memories": memories}


@app.patch("/memories/{memory_id}")
async def api_update_memory(memory_id: str, request: Request):
    body = await request.json()
    all_memories = await asyncio.to_thread(memory._load_memories)
    if not any(m["id"] == memory_id for m in all_memories):
        raise HTTPException(status_code=404, detail="Memory not found.")

    message = None
    if "text" in body:
        message = await asyncio.to_thread(memory.update_memory, memory_id, body["text"])
    if "pinned" in body:
        fn = memory.pin_memory if body["pinned"] else memory.unpin_memory
        message = await asyncio.to_thread(fn, memory_id)

    memories = await asyncio.to_thread(memory.list_memories_full)
    updated = next((m for m in memories if m["id"] == memory_id), None)
    return {"memory": updated, "message": message}


@app.delete("/memories/{memory_id}")
async def api_delete_memory(memory_id: str):
    all_memories = await asyncio.to_thread(memory._load_memories)
    if not any(m["id"] == memory_id for m in all_memories):
        raise HTTPException(status_code=404, detail="Memory not found.")
    await asyncio.to_thread(memory.delete_memory, memory_id)
    return {"deleted": memory_id}


@app.get("/memory-browser")
async def memory_browser_page():
    html_path = Path(__file__).parent / "web" / "memory_browser.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Prompt-log viewer HTTP API (/prompt-logs*, /notes/search, and the viewer
# page itself) now lives in prompt_log_engine.py, mounted above via
# app.include_router(prompt_log_router).
# ---------------------------------------------------------------------------


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


async def _count_tokens(client: httpx.AsyncClient, text: str) -> int | None:
    """Ask llama-server for the exact token count of a piece of text via
    its /tokenize endpoint. Returns None (never raises) on any failure -
    this is a diagnostic-only feature and should never block a real chat
    turn if llama-server is briefly unreachable for it."""
    if not text:
        return 0
    try:
        resp = await client.post(f"{LLAMA_SERVER_URL}/tokenize", json={"content": text})
        resp.raise_for_status()
        return len(resp.json().get("tokens", []))
    except Exception as e:
        alog(f"[AGENT] Token count lookup failed: {e}")
        return None


async def agent_loop(upstream_body: dict, section_labels: list[str] | None = None, tool_scores: dict | None = None, tool_tier: str | None = None):
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
    async with httpx.AsyncClient(timeout=None, headers={"Authorization": f"Bearer {AGENT_API_KEY}"}) as client:
        for iteration in range(MAX_TOOL_ITERATIONS):
            content = ""
            thinking = ""
            tool_calls = {}
            mode = None  # becomes "content" or "tool_calls" once known

            log_prompt(upstream_body, iteration, section_labels or [], tool_scores, tool_tier)
            try:
                async for chunk in _stream_chat(client, upstream_body):
                    delta = chunk["choices"][0].get("delta", {})

                    # Only populated when llama-server is run with
                    # --reasoning-format (see config.py) AND the loaded
                    # model actually produces a reasoning block. Doesn't
                    # affect `mode` - a reasoning model still ends up in
                    # "content" or "tool_calls" mode same as any other.
                    delta_thinking = delta.get("reasoning_content")
                    if delta_thinking:
                        thinking += delta_thinking

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
                alog(f"[AGENT] {e}")
                log_console(iteration, {"kind": "error", "text": str(e)}, thinking=thinking)
                yield ("delta", f"⚠️ {e}")
                yield ("done", {"role": "assistant", "content": str(e)})
                return

            if mode == "tool_calls" and tool_calls:
                calls = [tool_calls[i] for i in sorted(tool_calls)]
                message = {"role": "assistant", "content": content or None, "tool_calls": calls}

                unknown = [c for c in calls if c["function"]["name"] not in TOOL_FUNCTIONS]
                if unknown:
                    names = [c["function"]["name"] for c in unknown]
                    alog(f"[AGENT] Handing off {len(unknown)} client-owned tool call(s) to SillyTavern: {names}")
                    log_console(iteration, {"kind": "handoff", "text": json.dumps(calls, indent=2)}, thinking=thinking)
                    yield ("handoff", message)
                    return

                alog(f"[AGENT] Model requested {len(calls)} tool call(s)")
                upstream_body["messages"].append(message)

                names_in_batch = {c["function"]["name"] for c in calls}
                blocked_uid_writes = (
                    names_in_batch & _CALENDAR_UID_WRITE_TOOLS
                    if names_in_batch & _CALENDAR_READ_TOOLS
                    else set()
                )

                pending_calendar_change = calendar_manager.has_pending_change()
                last_user_text = next(
                    (m.get("content") for m in reversed(upstream_body["messages"]) if m.get("role") == "user"),
                    "",
                )
                looks_like_bare_confirmation = bool(
                    _BARE_CONFIRMATION_RE.match(str(last_user_text or "").strip())
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
                    elif name in _CALENDAR_APPLY_TOOLS and (names_in_batch & _CALENDAR_STAGE_TOOLS):
                        # confirm_pending requested in the same response as
                        # a stage call - block it so the user always sees
                        # the real staged description first, in its own
                        # turn, before anything can be applied.
                        result = (
                            "Error: calendar_confirm_pending was NOT called. It was requested "
                            "in the same response as a calendar create/edit/delete call, before "
                            "the user could see and confirm the actual staged description. Stop "
                            "here: relay the staged change's description to the user and wait "
                            "for their explicit confirmation in a SEPARATE following message "
                            "before calling calendar_confirm_pending."
                        )
                    elif name in _CALENDAR_STAGE_TOOLS and pending_calendar_change and looks_like_bare_confirmation:
                        # A change is already staged, and the user's last
                        # message reads as a plain confirmation of it, not a
                        # new/different request - re-staging now would just
                        # loop forever without ever applying anything.
                        result = (
                            f"Error: {name} was NOT called. A calendar change is already "
                            f"staged, and the user's last message ('{str(last_user_text).strip()}') "
                            f"looks like a confirmation of it, not a new or different request. "
                            f"Call calendar_confirm_pending now instead to actually apply the "
                            f"staged change. If the user meant something different, call "
                            f"calendar_cancel_pending first, then propose the new change."
                        )
                    else:
                        # Run in a thread so a slow web search doesn't freeze the server.
                        result = await asyncio.to_thread(TOOL_FUNCTIONS[name], args)

                    alog(f"[AGENT] {name}({args}) ->")
                    alog(f"[AGENT]   {str(result)[:400]}")

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
                log_console(iteration, {"kind": "tool_calls", "text": json.dumps(calls, indent=2)}, thinking=thinking)
                continue  # loop again so the model can use the tool result

            # No tool call -> normally the final answer. But if the user's
            # last message read as a bare confirmation AND a calendar
            # change is still staged, the model must not be allowed to
            # tell the user it succeeded without ever having called
            # calendar_confirm_pending. Live testing showed exactly this:
            # the model was correctly blocked from re-staging (see the
            # guard above), then simply wrote a false "successfully
            # deleted" answer here instead of calling
            # calendar_confirm_pending as the blocked-call error told it
            # to. Content is fully buffered above (never streamed early),
            # so it's safe to discard this answer and force another
            # iteration instead of returning it.
            last_user_text_for_check = next(
                (m.get("content") for m in reversed(upstream_body["messages"]) if m.get("role") == "user"),
                "",
            )
            unresolved_confirmation = (
                calendar_manager.has_pending_change()
                and bool(_BARE_CONFIRMATION_RE.match(str(last_user_text_for_check or "").strip()))
            )
            if unresolved_confirmation:
                alog("[AGENT] Model claimed a calendar change is resolved but never called "
                     "confirm/cancel - forcing it to actually do so instead of returning "
                     "the false answer.")
                upstream_body["messages"].append({"role": "assistant", "content": content or None})
                upstream_body["messages"].append({
                    "role": "system",
                    "content": (
                        "STOP: nothing has actually been applied to the calendar yet - a "
                        "change is still staged and waiting. Your previous message must NOT "
                        "have told the user it was done, because it wasn't - that was "
                        "incorrect and the user did not see it. Call calendar_confirm_pending "
                        "now to actually apply the staged change (the user's last message "
                        "confirmed it), or calendar_cancel_pending if that's not correct. Do "
                        "not write another direct answer claiming success without calling one "
                        "of these tools first."
                    ),
                })
                log_console(iteration, {"kind": "content_discarded", "text": content or ""}, thinking=thinking)
                continue

            alog("[AGENT] Model answered directly, without calling any tool.")
            log_console(iteration, {"kind": "content", "text": content or ""}, thinking=thinking)
            if content:
                yield ("delta", content)
            yield ("done", {"role": "assistant", "content": content})
            return

        # Hit MAX_TOOL_ITERATIONS without a final answer - bail out safely.
        bail_message = "(Agent stopped: too many tool calls in a row.)"
        alog(f"[AGENT] Bailed out after {MAX_TOOL_ITERATIONS} tool-call iterations without a final answer.")
        log_console(MAX_TOOL_ITERATIONS - 1, {"kind": "bailout", "text": bail_message}, thinking=thinking)
        yield ("delta", bail_message)
        yield ("done", {"role": "assistant", "content": bail_message})


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
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

    # Dynamic tool selection: only send the tools (and matching usage
    # instructions) relevant to what the user actually asked, instead of
    # every tool this server owns on every request. A pending calendar
    # change forces the whole calendar group in regardless of similarity
    # score, since a bare "yes" confirming it won't semantically match
    # "calendar" at all - see select_tools()/TOOL_GROUPS above.
    last_user_text = next(
        (m.get("content") for m in reversed(body.get("messages", [])) if m.get("role") == "user"),
        "",
    )
    # SillyTavern resends the full transcript as plain user/assistant text
    # every turn (tool_calls/tool-role messages never leave agent_loop, see
    # its docstring) - so the immediately preceding assistant reply is just
    # the most recent "assistant" entry in that same list. Feeds select_tools()'s
    # context-widening tier; see its docstring for why.
    prior_assistant_text = next(
        (m.get("content") for m in reversed(body.get("messages", [])) if m.get("role") == "assistant"),
        "",
    )

    # SillyTavern's own leading system message, if any (the character
    # card). Extracted here - earlier than it used to be - because
    # force_tool_names below needs it; the per-section token-count
    # diagnostic further down reuses this same variable rather than
    # re-extracting it.
    character_card_text = None
    if upstream_body["messages"] and upstream_body["messages"][0].get("role") == "system":
        character_card_text = upstream_body["messages"][0].get("content") or ""

    force_tool_names = set()
    if calendar_manager.has_pending_change():
        force_tool_names |= {name for name, group in TOOL_GROUPS.items() if group == "calendar"}

    # Post-turn attire sub-agent from the PREVIOUS turn: give it up to
    # ATTIRE_SUBAGENT_TIMEOUT_SECONDS to finish before reading attire state
    # below. asyncio.shield() means a timeout here doesn't cancel the task -
    # it keeps running and will still write attire.json when it's done, just
    # too late to be reflected in THIS turn's context.
    global _attire_subagent_task
    if _attire_subagent_task is not None:
        pending_task, _attire_subagent_task = _attire_subagent_task, None
        try:
            await asyncio.wait_for(asyncio.shield(pending_task), timeout=ATTIRE_SUBAGENT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            alog(f"[ATTIRE-SUBAGENT] Still running after {ATTIRE_SUBAGENT_TIMEOUT_SECONDS}s - "
                 f"proceeding with existing attire state for this turn.")
        except Exception as e:
            alog(f"[ATTIRE-SUBAGENT] Previous turn's pass raised: {e}")

    # Still needed below (NOT for tool selection anymore) purely for the
    # informational [PERSISTENT ATTIRE STATE] context block - attire is no
    # longer a main-agent-selectable tool group, so there's nothing here
    # to force in. Seeding a brand-new character from the card is also
    # deliberately not attempted here anymore: attire_subagent.py only
    # reacts to explicit changes in the last exchange, so an untracked
    # character stays untracked until their outfit actually changes -
    # accepted tradeoff, not a bug.
    attire_state_now = attire_manager._load()
    known_attire_match = attire_manager.find_character_names_in_text(
        attire_state_now, character_card_text or ""
    )

    selected_tools, tool_scores, tool_tier = select_tools(
        str(last_user_text or ""), force_tool_names, str(prior_assistant_text or "")
    )
    upstream_body["tools"] = selected_tools + client_tools
    upstream_body["tool_choice"] = "auto"

    tool_instruction = {
        "role": "system",
        "content": build_tool_instruction(selected_tools, client_tool_names),
    }
    messages_to_prepend = [tool_instruction]
    prepend_sections = [("tool_instruction", tool_instruction["content"])]

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
    prepend_sections.append(("current_datetime", current_datetime_text))

    # Server-side equivalent of the old extension's setExtensionPrompt():
    # inject the focused project's state directly, no tool call needed.
    project_state_text = await asyncio.to_thread(
        lambda: project_manager.build_context_text(project_manager._load())
    )
    if project_state_text:
        messages_to_prepend.append({"role": "system", "content": project_state_text})
        prepend_sections.append(("project_state", project_state_text))
    alog(f"[AGENT] Project state text sent to model:\n{project_state_text or '(none)'}")

    # Attire state for any already-tracked character mentioned in the
    # character card - reuses attire_state_now/known_attire_match computed
    # earlier for force_tool_names rather than reloading and re-scanning.
    # Best-effort: zero or multiple name matches both just skip injection
    # (see find_character_names_in_text's docstring) rather than guessing.
    attire_state_text = attire_manager.build_context_text_for_ids(
        attire_state_now, known_attire_match
    )
    if attire_state_text:
        messages_to_prepend.append({"role": "system", "content": attire_state_text})
        prepend_sections.append(("attire_state", attire_state_text))
    alog(f"[AGENT] Attire state text sent to model:\n{attire_state_text or '(none)'}")

    # Read-only, local-file-only - see calendar_manager.get_cached_context()
    # docstring for why this never triggers a live CalDAV call.
    calendar_context_text = await asyncio.to_thread(calendar_manager.get_cached_context)
    if calendar_context_text:
        messages_to_prepend.append({"role": "system", "content": calendar_context_text})
        prepend_sections.append(("calendar_cache", calendar_context_text))
    alog(f"[AGENT] Calendar cache text sent to model:\n{calendar_context_text or '(none)'}")

    # Shared explanation of the calendar staging convention (propose -> confirm/cancel),
    # sent once here instead of being repeated near-verbatim inside four separate
    # tool descriptions (calendar_create_event, calendar_create_events_batch,
    # calendar_edit_event, calendar_delete_event) - same information, sent once
    # per request instead of four times.
    calendar_staging_text = (
        "Calendar write convention: calendar_create_event, "
        "calendar_create_events_batch, calendar_edit_event, and "
        "calendar_delete_event never apply immediately - each only stages a "
        "pending change and returns a description of exactly what would "
        "happen. Relay that description to the user and wait for their "
        "explicit confirmation in a SEPARATE message before calling "
        "calendar_confirm_pending - never call calendar_confirm_pending in "
        "the same response as a staging call. If the user declines or wants "
        "something different, call calendar_cancel_pending instead."
    )
    messages_to_prepend.append({"role": "system", "content": calendar_staging_text})
    prepend_sections.append(("calendar_staging_convention", calendar_staging_text))

    # Auto-recall, part 1: pinned memories (identity slots + freeform pins)
    # are always shown, every turn, regardless of what's being discussed -
    # unlike the query-based search below, this isn't conditional on
    # last_user_msg existing or matching anything semantically.
    pinned = await asyncio.to_thread(memory.get_pinned_memories)
    pinned_ids = {m["id"] for m in pinned}
    if pinned:
        alog(f"[AGENT] {len(pinned)} pinned memory item(s) always shown")
        pinned_text = (
            "Core facts you always know about this user (id shown so you can "
            "call update_memory/delete_memory/unpin_memory directly if one of "
            "these needs correcting or removing - never guess an id):\n"
            + "\n".join(
                f"- [id: {m['id']}]" + (f" (slot: {m['slot']})" if m["slot"] else "") + f" {m['text']}"
                for m in pinned
            )
        )
        messages_to_prepend.append({"role": "system", "content": pinned_text})
        prepend_sections.append(("memory_pinned", pinned_text))

    # Auto-recall, part 2: silently check if any OTHER saved memories are
    # relevant to what the user just said, and inject them - no tool call
    # needed for this part. Pinned memories are excluded here since they're
    # already always shown above; this is only for everything else.
    last_user_msg = next(
        (m["content"] for m in reversed(body["messages"]) if m.get("role") == "user"),
        None,
    )
    if last_user_msg:
        relevant = await asyncio.to_thread(memory.search_memories, last_user_msg)
        relevant = [m for m in relevant if m["id"] not in pinned_ids]
        if relevant:
            alog(f"[AGENT] Recalled {len(relevant)} relevant memory item(s)")
            memory_recall_text = (
                "Relevant things you remember about this user from past "
                "conversations (id shown so you can call update_memory/delete_memory "
                "directly if one of these needs correcting - never guess an id):\n"
                + "\n".join(f"- [id: {m['id']}] {m['text']}" for m in relevant)
            )
            messages_to_prepend.append({"role": "system", "content": memory_recall_text})
            prepend_sections.append(("memory_recall", memory_recall_text))

    # Best-effort exact per-section token counts via llama-server's
    # /tokenize endpoint - purely diagnostic, printed to console to help
    # spot which system-prompt section is worth trimming. character_card_text
    # (SillyTavern's own leading system message, if any) was already
    # extracted earlier in this function for force_tool_names - reused here
    # rather than re-extracted, inspected only for visibility, never
    # modified. A tokenize failure just skips this log line; it never
    # blocks the actual turn.

    # The tools schema (this server's own TOOLS plus any client-registered
    # ones) is sent as JSON on every single request and rendered into the
    # prompt by the chat template - easy to overlook since it isn't a
    # system message like the others, but it's often the single largest
    # fixed cost in the whole prompt.
    tools_schema_text = json.dumps(upstream_body["tools"])

    # Positional labels aligned to upstream_body["messages"] indices, used
    # only by _log_prompt (see prompt-log-viewer). Order matches exactly
    # how messages_to_prepend + the original messages get concatenated
    # below - see that line for why this ordering is safe to rely on.
    message_section_labels = [name for name, _ in prepend_sections]
    if character_card_text is not None:
        message_section_labels.append("character_card")

    sections = [
        ("character_card", character_card_text),
        ("tools_schema", tools_schema_text),
    ] + prepend_sections

    async with httpx.AsyncClient(timeout=10.0, headers={"Authorization": f"Bearer {AGENT_API_KEY}"}) as tokenize_client:
        counts = await asyncio.gather(
            *(_count_tokens(tokenize_client, text) for _, text in sections)
        )
    breakdown = ", ".join(
        f"{name}={count if count is not None else '?'}"
        for (name, _), count in zip(sections, counts)
    )
    known_total = sum(c for c in counts if c is not None)
    alog(
        f"[AGENT] System prompt section sizes (tokens): {breakdown} "
        f"| known total: {known_total} / {LLAMA_CONTEXT} context"
    )

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
        async for kind, payload in agent_loop(upstream_body, message_section_labels, tool_scores, tool_tier):
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
        if finish_reason == "stop":
            _spawn_attire_subagent(str(last_user_text or ""), final_message.get("content") or "")
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
        async for kind, payload in agent_loop(upstream_body, message_section_labels, tool_scores, tool_tier):
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

            if kind == "done":
                _spawn_attire_subagent(str(last_user_text or ""), payload.get("content") or "")
                continue
            # "done" carries the full message for the non-streaming path only;
            # its content has already been sent as deltas above.

        yield sse({"index": 0, "delta": {}, "finish_reason": "stop"})
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
