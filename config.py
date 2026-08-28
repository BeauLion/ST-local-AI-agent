"""
config.py — single source of truth for every tunable value in the project.

Change a setting HERE, then restart start.py. Nothing else in the codebase
should contain a hardcoded port, path, threshold, or launch flag anymore —
if you find one, it belongs in this file instead.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

# Root of this repo (folder this config.py file lives in). Everything else
# below is built relative to this, so the project stays portable.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load .env here too (not just in calendar_manager.py) - CORS_ALLOWED_ORIGINS
# below is built at module-load time, so the env vars it reads need to
# already be in os.environ by the time this file runs, not just by the time
# some other module happens to get imported later. Calling load_dotenv()
# twice (here and in calendar_manager.py) is harmless - it's idempotent.
load_dotenv(Path(PROJECT_ROOT) / ".env")

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
#LLAMA_MODEL_REPO = "Qwen/Qwen2.5-14B-Instruct-GGUF:Q4_K_M"
#LLAMA_MODEL_REPO = "Qwen/Qwen3-14B-GGUF:Q4_K_M"
LLAMA_MODEL_REPO = "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced"

LLAMA_NGL = 99          # -ngl: layers offloaded to GPU (99 = full offload)
LLAMA_CONTEXT = 32768    # -c: context window size
LLAMA_TEMP = 0.6
LLAMA_TOP_P = 0.95
LLAMA_TOP_K = 20
LLAMA_MIN_P = 0.0   # new constant - see command builder change below
LLAMA_FLASH_ATTENTION = 'on'   # -fa: flash attention, speed optimization [on|off|auto]
LLAMA_USE_JINJA = True         # --jinja: required for structured tool-call output
#LLAMA_REASONING_FORMAT = "deepseek"
LLAMA_MD = "./llama.cpp/mtp-gemma-4-12B-it.gguf"
LLAMA_SPEC_TYPE = "draft-mtp"
#LLAMA_MMPROJ = "./llama.cpp/mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf"

def build_llama_server_command() -> list[str]:
    """
    Builds the full llama-server launch command as a list of arguments,
    ready to pass to subprocess.Popen(). Mirrors exactly what you'd type
    by hand in PowerShell, just assembled from the settings above.
    """
    cmd = [
        LLAMA_SERVER_EXE,
        "-hf", LLAMA_MODEL_REPO,
        "-md", LLAMA_MD,
        "--spec-type", LLAMA_SPEC_TYPE,
        #"--mmproj", LLAMA_MMPROJ,
        "-ngl", str(LLAMA_NGL),
        "-c", str(LLAMA_CONTEXT),
        "--temp", str(LLAMA_TEMP),
        "--host", LLAMA_SERVER_HOST,
        "--port", str(LLAMA_SERVER_PORT),
        "-fa", str(LLAMA_FLASH_ATTENTION),
        "--top-p", str(LLAMA_TOP_P),
        "--top-k", str(LLAMA_TOP_K),
        "--min-p", str(LLAMA_MIN_P),
        #"--reasoning-format", LLAMA_REASONING_FORMAT,
    ]
    if AGENT_API_KEY:
        cmd += ["--api-key", AGENT_API_KEY]
    if LLAMA_USE_JINJA:
        cmd.append("--jinja")
    return cmd


# ─────────────────────────────────────────────────────────────
# Agent server (FastAPI app, port 8100 — talks to SillyTavern)
# ─────────────────────────────────────────────────────────────

AGENT_SERVER_HOST = "0.0.0.0"
AGENT_SERVER_PORT = 8100

# Shared secret SillyTavern must send as its connection's "API Key" to reach
# /v1/chat/completions and /v1/models. Loaded from .env (see .env.example).
# If left empty, auth is skipped entirely (with a console warning) so a
# fresh clone still boots without extra setup - see verify_api_key() in
# main.py. Deliberately NOT applied to /projects - see handover-21.
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")

# Ceiling on how many tool-call round-trips the agent will do before
# forcing a final answer, to prevent infinite tool-calling loops.
MAX_TOOL_ITERATIONS = 8


# ─────────────────────────────────────────────────────────────
# Prompt inspection logging
# ─────────────────────────────────────────────────────────────

# Folder where the exact request body sent to llama-server gets logged.
PROMPT_LOG_DIR = os.path.join(PROJECT_ROOT, "prompt_logs")

# One log file is created per server run (a "session"). Every request to
# llama-server appends one entry - full messages array, tools schema, and
# sampling params, exactly as sent. Set False to disable without touching
# main.py. Off cost to be aware of: the tools schema alone is several KB,
# and it's re-logged on every single entry, so long sessions produce large
# files - that's expected, this is a debug tool, not meant to run forever.
PROMPT_LOG_ENABLED = True


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

# Fixed, singleton identity facts. Always injected into every system
# prompt regardless of query (see main.py's auto-recall block); saved via
# save_memory's optional `slot` param, which upserts instead of appending.
# See brainstorm-memory-structure-and-dedupe.md for the full design.
MEMORY_IDENTITY_SLOTS = ("identity", "occupation", "location")

# Cap on freeform (non-slot) pinned memories, toggled via pin_memory/
# unpin_memory. Slots don't count against this - they're bounded
# separately by MEMORY_IDENTITY_SLOTS itself (max 4).
MEMORY_MAX_FREEFORM_PINS = 10

# Similarity threshold for save_memory's dedupe nudge - deliberately
# higher than MEMORY_SIMILARITY_THRESHOLD (recall) since this needs to
# avoid flagging merely-related facts as duplicates, only near-identical
# or contradicting ones. Starting point only - needs live tuning against
# real memories once this is in use.
MEMORY_DEDUPE_SIMILARITY_THRESHOLD = 0.65


# ─────────────────────────────────────────────────────────────
# Dynamic tool selection (main.py's select_tools) — sends only the tool
# groups relevant to the user's last message instead of this server's full
# ~28-tool schema on every request, to cut prompt tokens. Scored the same
# way as memory recall: embed the query, cosine-similarity against a short
# description of each tool group (see TOOL_GROUP_DESCRIPTIONS in main.py).
# ─────────────────────────────────────────────────────────────

# Master switch - False sends every tool on every request (the old
# behavior), useful for A/B-ing prompt size against answer quality.
TOOL_SELECTION_ENABLED = True

# Minimum cosine-similarity score for a tool group to be included. Same
# scale/reasoning as MEMORY_SIMILARITY_THRESHOLD above - needs live tuning;
# the tool_selection_debug block _log_prompt() writes when PROMPT_LOG_ENABLED
# is on (see main.py) shows every group's actual score, which is the
# intended way to tune this from real traffic instead of guessing blind.
TOOL_SELECTION_MIN_SCORE = 0.2

# Secondary, much looser threshold used ONLY when nothing clears
# TOOL_SELECTION_MIN_SCORE - rescues genuinely ambiguous tool requests
# (e.g. oddly-phrased ones) without dragging in the full 34-tool list the
# way an unconditional fallback would. Plain conversation, where nothing
# clears even this, correctly gets just the core always-include set.
TOOL_SELECTION_RESCUE_SCORE = 0.18

# Cap on how many tools the rescue tier can add, so a weak/ambiguous match
# still stays small rather than ballooning back toward "everything".
TOOL_SELECTION_RESCUE_TOP_K = 3

# Individual tool names always sent regardless of score - cheap, frequently
# needed across unrelated requests, and their absence is confusing to a
# user who expects e.g. "what's 12*7" to always work.
TOOL_SELECTION_ALWAYS_INCLUDE = ("get_current_time", "calculate", "run_python")

# Context-widening tier (select_tools() in main.py) - only tried when the
# current user message alone clears neither TOOL_SELECTION_MIN_SCORE nor
# is a bare closing remark ("thanks!", "cool"). Re-scores [prior assistant
# reply + current message] at the same MIN_SCORE before falling through to
# the loose rescue tier below - rescues elliptical follow-ups like "and
# also change the time to 3pm" that carry no tool signal on their own but
# clearly continue a tool-relevant prior turn. Prior assistant text is
# truncated to its LAST N characters (not the first N) before embedding,
# since the embedding model's own truncation keeps the start of a long
# string - and the most relevant part of a prior reply for a follow-up is
# usually its tail (e.g. a trailing clarifying question), not its opening.
TOOL_SELECTION_CONTEXT_CHAR_LIMIT = 400


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

# Lightweight "key: value" tag syntax recognized only at the very top of a
# task's notes (front-matter style - stops at the first unrecognized line)
# and rendered compactly next to the task in the project context block.
# See project_manager.py's _parse_note_tags()/_format_tags_inline().
# Recognized keys: "dur" (reuses duration_manager.parse_duration_minutes),
# "effort" (below), and "when" (TASK_NOTE_WHEN_TIMES, optionally followed
# by a TASK_NOTE_WHEN_MODIFIERS word, e.g. "when: afternoon weekend").
TASK_NOTE_EFFORT_ALIASES = {
    "low": "low", "lo": "low",
    "medium": "medium", "med": "medium", "normal": "medium",
    "high": "high", "hi": "high",
}
TASK_NOTE_WHEN_TIMES = ("morning", "afternoon", "evening")
TASK_NOTE_WHEN_MODIFIERS = ("weekday", "weekend")

# localhost is always allowed below. Anything else (Tailscale IPs, LAN IPs,
# etc.) goes in .env as EXTRA_CORS_ORIGINS - a comma-separated list - so
# real IPs never end up committed to the repo. See .env.example for the
# expected format.
_extra_origins = os.getenv("EXTRA_CORS_ORIGINS", "")
_extra_origins_list = [origin.strip() for origin in _extra_origins.split(",") if origin.strip()]

CORS_ALLOWED_ORIGINS = ["http://localhost:8000"] + _extra_origins_list

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
CALDAV_TIMEOUT_SECONDS = 45

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

# Background cache for ambient context injection (main.py's system
# prompt) — NOT used by any read/write tool, which stay live. See
# calendar_manager.refresh_cache()/get_cached_context() and handover-17
# for why this boundary matters.
CALENDAR_DATA_DIR = os.path.join(PROJECT_ROOT, "calendar_data")
CALENDAR_CACHE_FILE = os.path.join(CALENDAR_DATA_DIR, "context_cache.json")

# How often the background timer refreshes the cache, and how far ahead
# it looks each time.
CALENDAR_CACHE_REFRESH_MINUTES = 5
CALENDAR_CACHE_LOOKAHEAD_DAYS = 14

# How many cached events get injected into the system prompt (same idea
# as MAX_TASKS_IN_CONTEXT above, for the calendar side).
CALENDAR_CACHE_MAX_EVENTS_IN_CONTEXT = 15


# ─────────────────────────────────────────────────────────────
# Task duration tracking (duration_manager.py)
# ─────────────────────────────────────────────────────────────

DURATION_DATA_DIR = os.path.join(PROJECT_ROOT, "duration_data")

# Confidence-state thresholds by entry count per category (see
# brainstorm-task-duration-tracking.md - three states, not a hard cutoff).
MIN_ENTRIES_FOR_ESTIMATE = 5
MIN_ENTRIES_FOR_CONFIDENT_ESTIMATE = 10

# Minimum cosine-similarity score for a task title to match an existing
# category by meaning rather than exact text/alias. Same pattern as
# MEMORY_SIMILARITY_THRESHOLD/DOCUMENT_SIMILARITY_THRESHOLD above.
DURATION_CATEGORY_SIMILARITY_THRESHOLD = 0.5

# Seeded canonical category list - deliberately NOT allowed to grow on its
# own (see brainstorm doc §5). Personalized for academic/research work as
# the primary focus, plus work admin/communication and household/personal
# admin. New categories beyond this only ever get added via explicit
# confirmation (duration_confirm_new_category), never silently.
DURATION_CANONICAL_CATEGORIES = (
    "reading", "writing", "data_analysis", "research_admin",
    "email", "meetings", "planning", "household",
)

# Small hand-maintained alias table for obvious synonyms, checked before
# the embedding fallback. Extend as you notice fragmentation.
DURATION_CATEGORY_ALIASES = {
    "literature review": "reading", "lit review": "reading", "paper reading": "reading",
    "drafting": "writing", "thesis": "writing", "thesis writing": "writing",
    "analysis": "data_analysis", "data analysis": "data_analysis", "coding": "data_analysis",
    "grant": "research_admin", "grant application": "research_admin",
    "funding": "research_admin", "paperwork": "research_admin",
    "e-mail": "email", "mail": "email", "correspondence": "email",
    "call": "meetings", "calls": "meetings", "meeting": "meetings", "supervisor": "meetings",
    "chores": "household", "errands": "household", "cleaning": "household", "shopping": "household",
}


# ─────────────────────────────────────────────────────────────
# Attire manager (attire_manager.py)
# ─────────────────────────────────────────────────────────────

ATTIRE_DATA_DIR = os.path.join(PROJECT_ROOT, "attire_data")

# "accessories" is the one multi-item (list) slot; everything else is a
# single string-or-None. See attire_manager.py's module docstring.
ATTIRE_SLOTS = ("head", "top", "bottom", "feet", "accessories")

MAX_CHARACTER_NAME_LENGTH = 200
MAX_ATTIRE_ITEM_LENGTH = 200

# How many assistant turns (from the start of the chat) the attire tool
# group gets force-included for an untracked character whose card
# describes an outfit, giving the model a window to seed it. See
# main.py's force_tool_names attire logic.
ATTIRE_SEED_ASSISTANT_TURN_LIMIT = 3

# Post-turn attire sub-agent (attire_subagent.py) - a separate, one-shot
# completion call against the same llama-server, run after every finished
# turn and decoupled entirely from the main agent's own tool selection.
# See main.py's chat_completions: the NEXT request waits on this (up to
# this many seconds) before reading attire state for injection, but never
# blocks longer than that - a slow/hung pass just means one turn of stale
# state, not a frozen agent.
ATTIRE_SUBAGENT_TIMEOUT_SECONDS = 15


# ─────────────────────────────────────────────────────────────
# run_python tool (Docker sandbox)
# ─────────────────────────────────────────────────────────────

DOCKER_IMAGE = "python:3.12-slim"
DOCKER_NETWORK_DISABLED = True
DOCKER_MEM_LIMIT = "256m"
DOCKER_CPU_COUNT = 1
DOCKER_TIMEOUT_SECONDS = 10