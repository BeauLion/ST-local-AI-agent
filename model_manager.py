"""
model_manager.py — owns the llama-server subprocess's entire lifecycle,
so the agent server (this same process) can swap the running model on
request instead of requiring you to manually stop/edit config.py/restart
start.py.

Ownership moved here from start.py: main.py's lifespan calls startup()/
shutdown() below, and start.py itself no longer launches llama-server at
all - see the comment at the top of the new start.py.

State persists to model_state.json (PROJECT_ROOT) so a full restart of
start.py comes back up on whatever model you last swapped to, not
silently back to config.py's LLAMA_MODEL_REPO. POST /model/reset goes
back to config.py's own values explicitly.

SAFETY: swap() always health-checks the new process before committing to
it. If the new model fails to come up in time (bad repo string, OOM on a
16GB card, etc.), it automatically rolls back to the previous
known-good model rather than leaving the agent dead - see swap()'s
docstring for the one case where that isn't possible.
"""

import json
import subprocess
import threading
import time
from pathlib import Path

import httpx

import config
from console_log import alog

_STATE_PATH = Path(config.PROJECT_ROOT) / "model_state.json"
_LOCK = threading.Lock()

_proc: subprocess.Popen | None = None

# phase: "stopped" | "starting" | "ready" | "swapping" | "error"
_state = {
    "phase": "stopped",
    "repo": config.LLAMA_MODEL_REPO,
    "ngl": config.LLAMA_NGL,
    "context": config.LLAMA_CONTEXT,
    "md": config.LLAMA_MD,
    "spec_type": config.LLAMA_SPEC_TYPE,
    "last_error": None,
    "started_at": None,
}


class SwapError(Exception):
    """Raised only when a swap can't even be attempted (e.g. one already
    running) - NOT raised for a swap that fails and successfully rolls
    back, since that's a handled, reported outcome, not an exception."""


def _load_persisted():
    if _STATE_PATH.exists():
        try:
            saved = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            for key in ("repo", "ngl", "context", "md", "spec_type"):
                if key in saved:
                    _state[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass


def _save_persisted():
    to_save = {k: _state[k] for k in ("repo", "ngl", "context", "md", "spec_type")}
    _STATE_PATH.write_text(json.dumps(to_save, indent=2), encoding="utf-8")


def _stream_output(proc: subprocess.Popen):
    for line in proc.stdout:
        print(f"[llama] {line}", end="")


def _launch(repo, ngl, context, md, spec_type) -> subprocess.Popen:
    cmd = config.build_llama_server_command(
        model_repo=repo, ngl=ngl, context=context, md=md, spec_type=spec_type,
    )
    alog(f"[MODEL] Launching llama-server:\n  {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    threading.Thread(target=_stream_output, args=(proc,), daemon=True).start()
    return proc


def _wait_ready(timeout_seconds: int) -> bool:
    url = f"{config.LLAMA_SERVER_URL}/health"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(1)
    return False


def _stop_current(timeout_seconds: int):
    global _proc
    if _proc is None:
        return
    alog("[MODEL] Stopping current llama-server process...")
    _proc.terminate()
    try:
        _proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        alog("[MODEL] llama-server didn't exit in time - killing it.")
        _proc.kill()
        _proc.wait(timeout=5)
    _proc = None


def startup():
    """Call once from main.py's lifespan on agent-server boot (blocking -
    wrap in asyncio.to_thread from async code)."""
    global _proc
    with _LOCK:
        _load_persisted()
        _state["phase"] = "starting"
        target = {k: _state[k] for k in ("repo", "ngl", "context", "md", "spec_type")}
    proc = _launch(**target)
    with _LOCK:
        _proc = proc
    ready = _wait_ready(config.LLAMA_SWAP_TIMEOUT_SECONDS)
    with _LOCK:
        if ready:
            _state["phase"] = "ready"
            _state["started_at"] = time.time()
            alog("[MODEL] llama-server is ready.")
        else:
            _state["phase"] = "error"
            _state["last_error"] = "llama-server did not become ready in time on startup."
            alog("[MODEL] ERROR: " + _state["last_error"])


def shutdown():
    """Call once from main.py's lifespan on agent-server shutdown (blocking)."""
    with _LOCK:
        _stop_current(config.LLAMA_STOP_TIMEOUT_SECONDS)
        _state["phase"] = "stopped"


def status() -> dict:
    with _LOCK:
        return dict(_state)


def is_ready() -> bool:
    with _LOCK:
        return _state["phase"] == "ready"


def swap(repo: str | None = None, ngl: int | None = None, context: int | None = None,
         md: str | None = None, spec_type: str | None = None) -> dict:
    """
    Blocking (run via asyncio.to_thread from the endpoint). Stops the
    current llama-server, starts the requested one, and waits for it to
    become healthy. On failure, rolls back to the previous known-good
    model automatically - swap() only leaves the agent in "error" phase
    (needing a real manual restart) if the ROLLBACK itself also fails,
    which points at something more serious than a bad model choice
    (e.g. llama-server.exe itself is broken).
    """
    global _proc
    with _LOCK:
        if _state["phase"] == "swapping":
            raise SwapError("A model swap is already in progress.")
        previous = {k: _state[k] for k in ("repo", "ngl", "context", "md", "spec_type")}
        candidate = {
            "repo": repo or previous["repo"],
            "ngl": ngl if ngl is not None else previous["ngl"],
            "context": context if context is not None else previous["context"],
            "md": md or previous["md"],
            "spec_type": spec_type or previous["spec_type"],
        }
        _state["phase"] = "swapping"
        _state["last_error"] = None
        _stop_current(config.LLAMA_STOP_TIMEOUT_SECONDS)

    proc = _launch(**candidate)
    with _LOCK:
        _proc = proc
    ready = _wait_ready(config.LLAMA_SWAP_TIMEOUT_SECONDS)  # lock NOT held - status() stays pollable

    if ready:
        with _LOCK:
            _state.update(candidate)
            _state["phase"] = "ready"
            _state["started_at"] = time.time()
        _save_persisted()
        alog(f"[MODEL] Swapped to {candidate['repo']} successfully.")
        return status()

    alog(f"[MODEL] Swap to {candidate['repo']} failed health check - rolling back to {previous['repo']}.")
    with _LOCK:
        _stop_current(config.LLAMA_STOP_TIMEOUT_SECONDS)

    proc = _launch(**previous)
    with _LOCK:
        _proc = proc
    rollback_ready = _wait_ready(config.LLAMA_SWAP_TIMEOUT_SECONDS)

    with _LOCK:
        _state.update(previous)
        if rollback_ready:
            _state["phase"] = "ready"
            _state["started_at"] = time.time()
            _state["last_error"] = (
                f"Swap to {candidate['repo']} failed (did not become healthy in time). "
                f"Rolled back to {previous['repo']}."
            )
        else:
            _state["phase"] = "error"
            _state["last_error"] = (
                f"Swap to {candidate['repo']} failed AND rollback to {previous['repo']} also failed. "
                "Check the console output and restart start.py manually."
            )
        return dict(_state)
