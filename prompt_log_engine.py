"""
Prompt-log engine: everything related to the prompt/console session logs
and the prompt_log_viewer.html page that reads them, isolated out of
main.py for easier maintenance. This module owns:

  - Writing session log files (log_prompt / log_console), called from
    main.py's agent_loop on every iteration.
  - Reading them back and serving them as JSON (/prompt-logs*).
  - Per-file note annotations, keyed by group/chunk position - see the
    comment above _annotations_path() for why position rather than
    iteration number.
  - Cross-session note search (/notes/search), which needs its own
    Python port of the viewer JS's buildGroups()/groupChunks() pairing
    logic to resolve a note's key back to the iteration/chunk it belongs
    to without asking the browser to do it.
  - Serving the viewer page itself (prompt_log_viewer.html/.css/.js).

main.py wires this in with:

    from prompt_log_engine import log_prompt, log_console, router as prompt_log_router
    app.include_router(prompt_log_router)

and calls log_prompt(...) / log_console(...) at the same call sites that
used to call the underscore-prefixed versions directly.
"""

import json
import threading
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from config import PROMPT_LOG_DIR, PROMPT_LOG_ENABLED
from console_log import alog, flush as flush_console

# ---------------------------------------------------------------------------
# Prompt-inspection logging: writes the EXACT JSON body this server sends to
# llama-server (real, final prompt - system messages, full history, tools
# schema, sampling params) to one file per server run. This is ground truth
# of what the model actually saw - different from the per-section token-
# count diagnostic in main.py, which only measures size, not content.
# ---------------------------------------------------------------------------
_PROMPT_LOG_DIR = Path(PROMPT_LOG_DIR)
_PROMPT_LOG_DIR.mkdir(exist_ok=True)
SESSION_LOG_PATH = _PROMPT_LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.log"
_prompt_log_lock = threading.Lock()

# One annotations JSON file per log file, holding free-text notes the user
# attaches from prompt_log_viewer.html. Keyed by group index (the log
# entry's position after prompt/console pairing - see buildGroups() in the
# viewer's JS, mirrored below by _build_groups()) rather than iteration
# number, since iteration numbers aren't guaranteed unique/contiguous
# (e.g. the orphaned-console edge case) while position in the file is
# always stable once written.
_annotations_lock = threading.Lock()


def _get_git_commit() -> str | None:
    """Best-effort short commit hash for the currently checked-out code.
    Returns None if git isn't available or this isn't a repo (e.g. a
    zipped copy) - callers must handle that, never raise."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# Computed once per process, same lifetime as SESSION_LOG_PATH - the
# commit a session ran under doesn't change mid-run.
GIT_COMMIT = _get_git_commit()


def _annotations_path(filename: str) -> Path:
    safe_name = Path(filename).name  # strips any path components - blocks traversal
    return _PROMPT_LOG_DIR / f"{safe_name}.annotations.json"


def log_prompt(upstream_body: dict, iteration: int, section_labels: list[str], tool_scores: dict | None = None, tool_tier: str | None = None) -> None:
    """Append one JSON object (one line) to this run's session log file,
    describing the exact request about to be POSTed to llama-server. Each
    message is tagged with a `section` label so the log viewer page can
    filter by chunk type, not just role. section_labels is positional -
    section_labels[i] describes upstream_body["messages"][i]; anything
    beyond that list's length is ordinary conversation history/tool-call
    round-trip messages, which grow between iterations. Best-effort: a
    write failure is printed but never blocks the actual turn."""
    if not PROMPT_LOG_ENABLED:
        return
    try:
        chunks = []
        for i, msg in enumerate(upstream_body["messages"]):
            section = section_labels[i] if i < len(section_labels) else "conversation"
            content = msg.get("content")
            if content is None and msg.get("tool_calls"):
                content = json.dumps(msg["tool_calls"])
            chunks.append({
                "role": msg.get("role", "unknown"),
                "section": section,
                "content": content or "",
            })
        if upstream_body.get("tools"):
            chunks.append({
                "role": "tools",
                "section": "tools_schema",
                "content": json.dumps(upstream_body["tools"], indent=2),
            })
        if tool_scores:
            included = {t["function"]["name"] for t in upstream_body.get("tools", [])}
            debug_chunk = {
                "role": "tools",
                "section": "tool_selection_debug",
                "content": json.dumps(
                    {n: {"score": round(s, 3), "included": n in included}
                     for n, s in sorted(tool_scores.items(), key=lambda kv: -kv[1])},
                    indent=2,
                ),
            }
            # Additive only - existing "content" shape is untouched, so the
            # current log viewer keeps working even if it doesn't render
            # this field. Shows which of select_tools()'s tiers fired:
            # "direct" / "context_widened" / "rescue" / "core_only" /
            # "disabled" / "empty" - see main.py's select_tools() docstring.
            if tool_tier:
                debug_chunk["tier"] = tool_tier
            chunks.append(debug_chunk)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "prompt",
            "iteration": iteration,
            "model": upstream_body.get("model"),
            "commit": GIT_COMMIT,
            "chunks": chunks,
        }
        with _prompt_log_lock, open(SESSION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[AGENT] Prompt log write failed: {e}")


def log_console(iteration: int, response: dict | None = None, thinking: str | None = None) -> None:
    """Flushes everything buffered via console_log.alog() since the last
    flush and appends it as its own JSONL entry, immediately after the
    prompt entry it belongs to - prompt_log_viewer.html pairs them by
    position. `response`, if given, is {"kind": ..., "text": ...} - the
    model's actual output for this iteration. `thinking`, if given, is the
    model's reasoning/chain-of-thought text for this iteration (only ever
    populated when llama-server is run with --reasoning-format and the
    loaded model actually emits one). Writes nothing if there's nothing at
    all to record, so quiet iterations don't clutter the log."""
    if not PROMPT_LOG_ENABLED:
        flush_console()  # still drain it so it can't leak into a later request
        return
    lines = flush_console()
    if not lines and not response and not thinking:
        return
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "console",
            "iteration": iteration,
            "lines": lines,
        }
        if response:
            entry["response"] = response
        if thinking:
            entry["thinking"] = thinking
        with _prompt_log_lock, open(SESSION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[AGENT] Console log write failed: {e}")


# ---------------------------------------------------------------------------
# Python port of prompt_log_viewer.js's buildGroups()/groupChunks()/groupRef()
# - used only by the /notes/search endpoint below, to resolve a note's
# position-based key back to the iteration/chunk it's attached to without
# a round trip to the browser. Keep in sync with the JS versions if either
# changes; they must agree on how prompt/console entries pair up.
# ---------------------------------------------------------------------------

def _entry_type(e: dict) -> str:
    return e.get("type") or ("prompt" if e.get("chunks") is not None else "console")


def _build_groups(entries: list[dict]) -> list[dict]:
    out = []
    i, n = 0, len(entries)
    while i < n:
        e = entries[i]
        if _entry_type(e) == "console":
            out.append({"prompt": None, "console": e})
            i += 1
            continue
        group = {"prompt": e, "console": None}
        nxt = entries[i + 1] if i + 1 < n else None
        if nxt and _entry_type(nxt) == "console":
            group["console"] = nxt
            i += 2
        else:
            i += 1
        out.append(group)
    return out


def _group_chunks(group: dict) -> list[dict]:
    chunks = list(group["prompt"]["chunks"]) if group["prompt"] else []
    console = group["console"]
    if console:
        if console.get("lines"):
            chunks.append({
                "role": "console",
                "section": "console_output",
                "content": "\n".join(console["lines"]),
            })
        if console.get("thinking"):
            chunks.append({
                "role": "thinking",
                "section": "reasoning",
                "content": console["thinking"],
            })
        if console.get("response"):
            r = console["response"]
            chunks.append({
                "role": "response",
                "section": r.get("kind") or "response",
                "content": r.get("text") or "",
            })
    return chunks


def _group_ref(group: dict) -> dict:
    return group["prompt"] or group["console"]


def _read_log_entries(path: Path, filename_for_errors: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[AGENT] Skipping malformed log line {line_num} in {filename_for_errors}")
    return entries


# ---------------------------------------------------------------------------
# Prompt-log viewer HTTP API - plain REST for the /prompt-log-viewer page,
# reading the JSONL files log_prompt() writes to PROMPT_LOG_DIR.
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/prompt-logs")
async def list_prompt_logs():
    """Lists available session log files, most recent first."""
    files = sorted(_PROMPT_LOG_DIR.glob("session_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        }
        for f in files
    ]


@router.get("/prompt-logs/{filename}")
async def get_prompt_log(filename: str):
    """Returns one session log file's entries, parsed from JSONL."""
    safe_name = Path(filename).name  # strips any path components - blocks traversal
    path = _PROMPT_LOG_DIR / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Log file not found")
    return _read_log_entries(path, filename)


@router.get("/prompt-logs/{filename}/annotations")
async def get_prompt_log_annotations(filename: str):
    """Returns the saved notes for one log file, or an empty shape if the
    file has never been annotated."""
    path = _annotations_path(filename)
    if not path.is_file():
        return {"iteration_notes": {}, "chunk_notes": {}}
    try:
        with _annotations_lock, open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[AGENT] Annotations read failed for {filename}: {e}")
        return {"iteration_notes": {}, "chunk_notes": {}}


@router.put("/prompt-logs/{filename}/annotations")
async def put_prompt_log_annotations(filename: str, request: Request):
    """Full-replace save of one log file's notes. Body is the whole
    {"iteration_notes": {...}, "chunk_notes": {...}} shape - the viewer
    always sends the complete, current set rather than a partial patch."""
    body = await request.json()
    if not isinstance(body, dict) or "iteration_notes" not in body or "chunk_notes" not in body:
        raise HTTPException(status_code=400, detail="Expected {iteration_notes, chunk_notes}")
    path = _annotations_path(filename)
    try:
        with _annotations_lock, open(path, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save annotations: {e}")
    return {"saved": True}


@router.get("/notes/search")
async def search_notes(q: str):
    """Searches note text (both iteration- and chunk-level) across every
    session log's annotations file, resolving each match to the
    iteration/chunk it's attached to. Skips parsing a session log entirely
    unless that session actually has a matching note, so this stays cheap
    even as the number of sessions grows - the annotations files are much
    smaller than the logs themselves."""
    query = q.strip().lower()
    if not query:
        return {"results": []}

    results = []
    for ann_path in sorted(_PROMPT_LOG_DIR.glob("*.annotations.json")):
        try:
            with _annotations_lock, open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        matches = [
            ("iteration", k, t) for k, t in (ann.get("iteration_notes") or {}).items()
            if query in t.lower()
        ]
        matches += [
            ("chunk", k, t) for k, t in (ann.get("chunk_notes") or {}).items()
            if query in t.lower()
        ]
        if not matches:
            continue

        log_filename = ann_path.name.removesuffix(".annotations.json")
        log_path = _PROMPT_LOG_DIR / log_filename
        if not log_path.is_file():
            continue  # annotations file survived a deleted/renamed log - nothing to resolve against

        entries = _read_log_entries(log_path, log_filename)
        groups = _build_groups(entries)

        for kind, key, note_text in matches:
            try:
                group_index = int(key.split(":")[0])
            except ValueError:
                continue
            if group_index < 0 or group_index >= len(groups):
                continue
            group = groups[group_index]
            ref = _group_ref(group)
            chunk, chunk_index = None, None
            if kind == "chunk":
                try:
                    chunk_index = int(key.split(":")[1])
                except (IndexError, ValueError):
                    continue
                group_chunks = _group_chunks(group)
                if chunk_index < 0 or chunk_index >= len(group_chunks):
                    continue
                chunk = group_chunks[chunk_index]

            results.append({
                "filename": log_filename,
                "group_index": group_index,
                "chunk_index": chunk_index,
                "kind": kind,
                "iteration": ref.get("iteration"),
                "timestamp": ref.get("timestamp"),
                "model": group["prompt"].get("model") if group["prompt"] else None,
                "note_text": note_text,
                "chunk": chunk,
            })

    results.sort(key=lambda r: r["timestamp"] or "", reverse=True)
    return {"results": results}


@router.get("/prompt-log-viewer")
async def prompt_log_viewer_page():
    html_path = Path(__file__).parent / "web" / "prompt_log_viewer.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get("/prompt_log_viewer.css")
async def prompt_log_viewer_css():
    css_path = Path(__file__).parent / "web" / "prompt_log_viewer.css"
    return Response(content=css_path.read_text(encoding="utf-8"), media_type="text/css")


@router.get("/prompt_log_viewer.js")
async def prompt_log_viewer_js():
    js_path = Path(__file__).parent / "web" / "prompt_log_viewer.js"
    return Response(content=js_path.read_text(encoding="utf-8"), media_type="application/javascript")
