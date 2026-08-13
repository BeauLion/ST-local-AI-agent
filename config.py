"""
config.py — single source of truth for every tunable value in the project.

Change a setting HERE, then restart start.py. Nothing else in the codebase
should contain a hardcoded port, path, threshold, or launch flag anymore —
if you find one, it belongs in this file instead.
"""

import os

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

# Root of this repo (folder this config.py file lives in). Everything else
# below is built relative to this, so the project stays portable.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Full path to your llama-server.exe (llama.cpp build). Update this if you
# ever move or reinstall llama.cpp.
LLAMA_SERVER_EXE = os.path.join(PROJECT_ROOT, "llama.cpp", "llama-server.exe")

# Sandboxed folder the write_file/read_file/list_files/search_documents
# tools are restricted to. Relative to PROJECT_ROOT.
SAFE_FILES_DIR = os.path.join(PROJECT_ROOT, "agent_files")

# Where memory.py stores its JSON files (memories.json, doc_index.json).
MEMORY_DATA_DIR = os.path.join(PROJECT_ROOT, "memory_data")

# Where project_manager.py stores its JSON file (projects.json).
PROJECT_DATA_DIR = os.path.join(PROJECT_ROOT, "project_data")


# ─────────────────────────────────────────────────────────────
# llama-server (the local inference engine, port 8080)
# ─────────────────────────────────────────────────────────────

LLAMA_SERVER_HOST = "0.0.0.0"
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_URL = f"http://localhost:{LLAMA_SERVER_PORT}"

# The exact model to pull/run via llama.cpp's -hf shorthand.
LLAMA_MODEL_REPO = "Qwen/Qwen2.5-14B-Instruct-GGUF:Q4_K_M"

LLAMA_NGL = 99          # -ngl: layers offloaded to GPU (99 = full offload)
LLAMA_CONTEXT = 8192    # -c: context window size
LLAMA_TEMP = 0.5        # --temp: lowered from default: fixed the malformed
                        #   tool-call bug (handover item 15). Raise this
                        #   again only if you're prepared to re-test that
                        #   fix, or re-add the salvage function.
LLAMA_FLASH_ATTENTION = 'on'   # -fa: flash attention, speed optimization [on|off|auto]
LLAMA_USE_JINJA = True         # --jinja: required for structured tool-call output


def build_llama_server_command() -> list[str]:
    """
    Builds the full llama-server launch command as a list of arguments,
    ready to pass to subprocess.Popen(). Mirrors exactly what you'd type
    by hand in PowerShell, just assembled from the settings above.
    """
    cmd = [
        LLAMA_SERVER_EXE,
        "-hf", LLAMA_MODEL_REPO,
        "-ngl", str(LLAMA_NGL),
        "-c", str(LLAMA_CONTEXT),
        "--temp", str(LLAMA_TEMP),
        "--host", LLAMA_SERVER_HOST,
        "--port", str(LLAMA_SERVER_PORT),
        "-fa", str(LLAMA_FLASH_ATTENTION),
    ]
    if LLAMA_USE_JINJA:
        cmd.append("--jinja")
    return cmd


# ─────────────────────────────────────────────────────────────
# Agent server (FastAPI app, port 8100 — talks to SillyTavern)
# ─────────────────────────────────────────────────────────────

AGENT_SERVER_HOST = "0.0.0.0"
AGENT_SERVER_PORT = 8100

# Ceiling on how many tool-call round-trips the agent will do before
# forcing a final answer, to prevent infinite tool-calling loops.
MAX_TOOL_ITERATIONS = 8


# ─────────────────────────────────────────────────────────────
# Memory / RAG (memory.py)
# ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Minimum cosine-similarity score for a saved memory to be considered
# relevant and injected into context. Lower = more permissive recall.
MEMORY_SIMILARITY_THRESHOLD = 0.15

# Stricter threshold for document RAG (search_documents), since documents
# are larger/noisier than short personal-memory facts.
DOCUMENT_SIMILARITY_THRESHOLD = 0.3


# ─────────────────────────────────────────────────────────────
# write_file tool
# ─────────────────────────────────────────────────────────────

WRITE_FILE_ALLOWED_EXTENSIONS = (".txt", ".md")
WRITE_FILE_MAX_CHARS = 20_000

# delete_file tool - slightly broader than write_file's extensions, since
# agent_files can also contain RAG-uploaded PDFs the user may want removed.
DELETE_FILE_ALLOWED_EXTENSIONS = (".txt", ".md", ".pdf")


# ─────────────────────────────────────────────────────────────
# Project manager (project_manager.py) — formerly the SillyTavern
# "Lightweight Project Manager" extension. State, validation, and the
# model tools now live entirely on the backend; the extension is a thin
# UI that reads/writes this server over HTTP. See handover notes.
# ─────────────────────────────────────────────────────────────

MAX_PROJECT_NAME_LENGTH = 120
MAX_TASK_TITLE_LENGTH = 240
MAX_PROJECT_CODE_LENGTH = 6
MAX_TASK_NOTE_LENGTH = 2000
MAX_BATCH_OPERATIONS = 25

# How many of the focused project's tasks get injected into the model's
# system prompt on every request (see main.py's project context block).
MAX_TASKS_IN_CONTEXT = 8

PROJECT_STATUSES = ("active", "paused", "completed")
TASK_STATUSES = ("pending", "active", "blocked", "done", "cancelled")
TASK_PRIORITIES = ("low", "normal", "high")

# Origins allowed to call the agent server's HTTP API from the browser.
# SillyTavern's own page (wherever it's hosted) needs to be listed here for
# the project-manager extension's fetch() calls to be allowed by CORS.
# "*" is fine for a single-user local setup; tighten this if you ever expose
# the agent server beyond localhost.
CORS_ALLOWED_ORIGINS = ["*"]


# ─────────────────────────────────────────────────────────────
# Calendar (calendar_manager.py) — iCloud via CalDAV
# ─────────────────────────────────────────────────────────────

# iCloud's CalDAV entry point. Same URL regardless of which calendar(s)
# your account has - principal discovery happens from here.
ICLOUD_CALDAV_URL = "https://caldav.icloud.com"

# IANA timezone name used to localize any naive date/time the model writes
# to the calendar (create/edit). A named zone (not a fixed UTC offset) is
# required so CEST/CET DST transitions are handled automatically. Needs
# the `tzdata` package on Windows (stdlib zoneinfo has no built-in tz
# database there) - see requirements.txt.
CALENDAR_TIMEZONE = "Europe/Amsterdam"

# Per-request timeout (seconds) for all CalDAV calls to iCloud. Previously
# unset, which let a single stalled request hang on whatever the caldav
# library's internal default is (~120s) with no way to recover from it.
CALDAV_TIMEOUT_SECONDS = 30

# Extra attempts (beyond the first) confirm_pending() makes if writing a
# staged change to iCloud fails, with a short delay between attempts.
CALENDAR_WRITE_RETRIES = 2

# Calendar to default to when no calendar_name is given. Must exactly
# match (or uniquely partially match) one of your real iCloud calendar
# names - check with calendar_list_calendars if unsure. Falls back to
# iCloud's first-returned calendar if this name doesn't match anything.
CALENDAR_DEFAULT_NAME = "Home"

# Names of the environment variables calendar_manager.py reads credentials
# from (via a .env file in PROJECT_ROOT - see .env.example). Secrets never
# go in this file.
ICLOUD_USERNAME_ENV_VAR = "ICLOUD_USERNAME"
ICLOUD_APP_PASSWORD_ENV_VAR = "ICLOUD_APP_PASSWORD"

# How many days ahead calendar_list_events looks by default when the model
# doesn't specify an end date.
CALENDAR_DEFAULT_LOOKAHEAD_DAYS = 14

# How many days back/forward calendar_search_events looks by default.
CALENDAR_SEARCH_LOOKBACK_DAYS = 7
CALENDAR_SEARCH_LOOKAHEAD_DAYS = 90

# Staged (unconfirmed) create/edit/delete changes expire after this many
# minutes, so a "confirm" typed much later in an unrelated part of the
# conversation can't accidentally apply an old, stale proposed change.
CALENDAR_PENDING_CHANGE_TTL_MINUTES = 10

# How wide a date_search() window to scan when resolving an event by UID
# for edit/delete (calendar_manager._get_event_by_uid_via_search). caldav's
# own event_by_uid() proved unreliable against iCloud even with a genuinely
# correct UID (likely guesses a resource URL internally rather than really
# searching) - date_search()+filter is the same mechanism list/search
# already use successfully, so it's used here too instead.
CALENDAR_UID_LOOKUP_LOOKBACK_DAYS = 365
CALENDAR_UID_LOOKUP_LOOKAHEAD_DAYS = 365

# calendar_create_events_batch: stages MULTIPLE events (e.g. a proposed
# morning schedule from project tasks) as one pending change. Safety
# ceiling on batch size, and the default block length used when an event
# in the batch doesn't specify an explicit end time (shorter than the
# single-event default of 1 hour, since this is aimed at task blocks).
CALENDAR_BATCH_MAX_EVENTS = 12
CALENDAR_BATCH_DEFAULT_DURATION_MINUTES = 30


# ─────────────────────────────────────────────────────────────
# run_python tool (Docker sandbox)
# ─────────────────────────────────────────────────────────────

DOCKER_IMAGE = "python:3.12-slim"
DOCKER_NETWORK_DISABLED = True
DOCKER_MEM_LIMIT = "256m"
DOCKER_CPU_COUNT = 1
DOCKER_TIMEOUT_SECONDS = 10