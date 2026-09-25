"""
model_manager.py — owns the llama-server subprocess's entire lifecycle,
so the agent server (this same process) can swap the running model on
request instead of requiring you to manually stop/edit config.py/restart
start.py.

Ownership moved here from start.py: main.py's lifespan calls startup()/
shutdown() below, and start.py itself no longer launches llama-server at
all - see the comment at the top of the new start.py.

State persists to model_state.json (DATA_ROOT) so a full restart of
start.py comes back up on whatever model you last swapped to, not
silently back to config.py's LLAMA_MODEL_REPO. POST /model/reset goes
back to config.py's own values explicitly.

SAFETY: swap() always health-checks the new process before committing to
it. If the new model fails to come up in time (bad repo string, OOM on a
16GB card, etc.), it automatically rolls back to the previous
known-good model rather than leaving the agent dead - see swap()'s
docstring for the one case where that isn't possible.

TWO BACKENDS (config.LLAMA_BACKEND):
  "local"      - everything above: this process launches/stops llama-server.
  "llama-swap" - llama-server runs on another machine behind llama-swap,
                 which loads a model on demand from the request's "model"
                 field. Nothing is launched or killed here: the "current
                 model" is just a llama-swap model name that main.py and
                 attire_subagent.py put in every request body (via
                 request_model()), a swap asks llama-swap to load the new
                 one and only commits to it once it answers /health, and a
                 background poller keeps phase in sync with whether the
                 remote machine is reachable at all ("unreachable" while
                 it's down - the agent itself stays up and serving the
                 dashboard/projects/calendar, only chat 503s).
The public functions (startup/shutdown/status/swap/reset) dispatch to one
backend or the other; callers never need to know which is active.
"""

import json
import subprocess
import threading
import time
from pathlib import Path

import httpx

import config
from console_log import alog

_STATE_PATH = Path(config.DATA_ROOT) / "model_state.json"
# Separate file on purpose: a llama-swap model NAME persisted into
# model_state.json would later be passed to a local launch as an -hf repo
# (or vice versa) if you ever switch LLAMA_BACKEND back.
_REMOTE_STATE_PATH = Path(config.DATA_ROOT) / "llama_swap_state.json"
_LOCK = threading.Lock()

_proc: subprocess.Popen | None = None

# phase: "stopped" | "starting" | "ready" | "swapping" | "error"
#        (+ "unreachable" - llama-swap backend only)
_state = {
    "phase": "stopped",
    "repo": config.LLAMA_MODEL_REPO,
    "ngl": config.LLAMA_NGL,
    "context": config.LLAMA_CONTEXT,
    "md": config.LLAMA_MD,
    "spec_type": config.LLAMA_SPEC_TYPE,
    "last_error": None,
    "started_at": None,
    "backend": config.LLAMA_BACKEND,
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


def _local_startup():
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


def _local_shutdown():
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


def _local_swap(repo: str | None = None, ngl: int | None = None, context: int | None = None,
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


# ─────────────────────────────────────────────────────────────
# "llama-swap" backend - see the module docstring.
# ─────────────────────────────────────────────────────────────

_poller_stop = threading.Event()
_poller: threading.Thread | None = None


def _is_remote() -> bool:
    return config.LLAMA_BACKEND == "llama-swap"


def llama_headers() -> dict:
    """Headers for every request the agent makes to llama-server/llama-swap.
    Sent even with an empty key, as it always was - both ignore it when no
    key is configured on their end."""
    return {"Authorization": f"Bearer {config.LLAMA_API_KEY}"}


def request_model() -> str | None:
    """The "model" value to put in a chat-completion body, or None to leave
    the body's own value alone ("local": llama-server ignores it anyway,
    there's only ever one model loaded)."""
    if not _is_remote():
        return None
    with _LOCK:
        return _state["repo"]


def llama_url(path: str) -> str:
    """URL for a llama-server-native endpoint (/tokenize, /props, ...).
    llama-swap only routes the OpenAI-style /v1/* endpoints by model name;
    anything else has to go through /upstream/<model>/ to reach the right
    llama-server."""
    model = request_model()
    if model:
        return f"{config.LLAMA_SERVER_URL}/upstream/{model}{path}"
    return f"{config.LLAMA_SERVER_URL}{path}"


def list_remote_models() -> list[str]:
    """Model names configured in llama-swap (blocking). Raises
    httpx.HTTPError / ValueError / KeyError if llama-swap can't be reached
    or answers with something unexpected."""
    resp = httpx.get(f"{config.LLAMA_SERVER_URL}/v1/models", headers=llama_headers(), timeout=5)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def _remote_context(model: str) -> int | None:
    """Best-effort n_ctx of a loaded model, for status display only."""
    try:
        resp = httpx.get(f"{config.LLAMA_SERVER_URL}/upstream/{model}/props", headers=llama_headers(), timeout=5)
        resp.raise_for_status()
        return resp.json().get("default_generation_settings", {}).get("n_ctx")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


def _load_remote_persisted():
    if _REMOTE_STATE_PATH.exists():
        try:
            saved = json.loads(_REMOTE_STATE_PATH.read_text(encoding="utf-8"))
            _state["repo"] = saved.get("model")
        except (json.JSONDecodeError, OSError):
            pass


def _save_remote_persisted():
    _REMOTE_STATE_PATH.write_text(json.dumps({"model": _state["repo"]}, indent=2), encoding="utf-8")


def _refresh_remote():
    """One reachability check. Never touches state mid-swap - swap() owns
    phase for its duration."""
    try:
        models = list_remote_models()
    except (httpx.HTTPError, ValueError, KeyError) as e:
        with _LOCK:
            if _state["phase"] != "swapping":
                if _state["phase"] != "unreachable":
                    alog(f"[MODEL] llama-swap at {config.LLAMA_SERVER_URL} is unreachable: {e}")
                _state["phase"] = "unreachable"
                _state["last_error"] = f"Can't reach llama-swap at {config.LLAMA_SERVER_URL} ({e})."
        return

    with _LOCK:
        if _state["phase"] == "swapping":
            return
        if _state["phase"] == "unreachable":
            _state["last_error"] = None
        if _state["repo"] not in models:
            if config.LLAMA_SWAP_DEFAULT_MODEL in models:
                chosen = config.LLAMA_SWAP_DEFAULT_MODEL
            elif models:
                chosen = models[0]
            else:
                _state["phase"] = "error"
                _state["last_error"] = "llama-swap is reachable but has no models configured."
                return
            if _state["repo"]:
                _state["last_error"] = (
                    f"Model '{_state['repo']}' isn't configured in llama-swap any more - using '{chosen}'."
                )
            _state["repo"] = chosen
            _save_remote_persisted()
        if _state["phase"] != "ready":
            alog(f"[MODEL] llama-swap is reachable - using model '{_state['repo']}'.")
            _state["phase"] = "ready"
            _state["started_at"] = time.time()


def _poll_remote():
    while not _poller_stop.wait(config.LLAMA_SWAP_POLL_SECONDS):
        _refresh_remote()


def _remote_startup():
    """Effectively non-blocking (one short request): the agent boots even
    if the GPU machine is off, and the poller picks it up once it's back."""
    global _poller
    with _LOCK:
        _state.update({"repo": None, "ngl": None, "context": None, "md": None, "spec_type": None})
        _load_remote_persisted()
        _state["phase"] = "starting"
    alog(f"[MODEL] Using llama-swap at {config.LLAMA_SERVER_URL} (no local llama-server).")
    _refresh_remote()
    _poller_stop.clear()
    _poller = threading.Thread(target=_poll_remote, daemon=True)
    _poller.start()


def _remote_shutdown():
    _poller_stop.set()
    with _LOCK:
        _state["phase"] = "stopped"


def _remote_swap(repo: str | None = None) -> dict:
    """
    Blocking. Asks llama-swap to load `repo` (a model name from its
    config.yaml) and only switches over once that model's llama-server
    answers /health. ngl/context/md/spec_type don't apply - those are set
    per model in llama-swap's config on the GPU machine.

    "Rollback" is free here: nothing on this side was stopped, so a failed
    swap just leaves the previous name selected. (llama-swap may have
    unloaded the previous model to make room - it reloads it on the next
    request that names it.)
    """
    with _LOCK:
        if _state["phase"] == "swapping":
            raise SwapError("A model swap is already in progress.")
        previous = _state["repo"]
        _state["phase"] = "swapping"
        _state["last_error"] = None

    try:
        models = list_remote_models()
        if repo not in models:
            raise ValueError(
                f"'{repo}' isn't a model in llama-swap's config (available: {', '.join(models) or 'none'})"
            )
        # llama-swap starts the model's llama-server on the first request
        # routed to it and holds that request until the server is healthy,
        # so this one call IS the load + health check.
        resp = httpx.get(
            f"{config.LLAMA_SERVER_URL}/upstream/{repo}/health",
            headers=llama_headers(), timeout=config.LLAMA_SWAP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except (httpx.HTTPError, ValueError, KeyError) as e:
        alog(f"[MODEL] Swap to '{repo}' failed: {e}")
        with _LOCK:
            _state["phase"] = "ready"
            _state["last_error"] = f"Swap to '{repo}' failed ({e}). Still using '{previous}'."
        _refresh_remote()  # flips to "unreachable" if that's what the failure was
        return status()

    context = _remote_context(repo)
    with _LOCK:
        _state.update({"repo": repo, "context": context, "phase": "ready", "started_at": time.time()})
        _save_remote_persisted()
    alog(f"[MODEL] Swapped to '{repo}' on llama-swap.")
    return status()


# ─────────────────────────────────────────────────────────────
# Public API - dispatches to whichever backend is configured.
# ─────────────────────────────────────────────────────────────

def startup():
    """Call once from main.py's lifespan on agent-server boot (blocking -
    wrap in asyncio.to_thread from async code)."""
    return _remote_startup() if _is_remote() else _local_startup()


def shutdown():
    """Call once from main.py's lifespan on agent-server shutdown (blocking)."""
    return _remote_shutdown() if _is_remote() else _local_shutdown()


def swap(repo: str | None = None, ngl: int | None = None, context: int | None = None,
         md: str | None = None, spec_type: str | None = None) -> dict:
    """Blocking (run via asyncio.to_thread). See _local_swap/_remote_swap."""
    if _is_remote():
        return _remote_swap(repo=repo)
    return _local_swap(repo=repo, ngl=ngl, context=context, md=md, spec_type=spec_type)


def reset() -> dict:
    """Blocking. Back to the configured default model: config.py's LLAMA_*
    values ("local") or LLAMA_SWAP_DEFAULT_MODEL ("llama-swap", falling back
    to the first model llama-swap lists)."""
    if _is_remote():
        target = config.LLAMA_SWAP_DEFAULT_MODEL
        if not target:
            try:
                target = next(iter(list_remote_models()), None)
            except (httpx.HTTPError, ValueError, KeyError):
                target = None
        return _remote_swap(repo=target)
    return _local_swap(
        repo=config.LLAMA_MODEL_REPO, ngl=config.LLAMA_NGL, context=config.LLAMA_CONTEXT,
        md=config.LLAMA_MD, spec_type=config.LLAMA_SPEC_TYPE,
    )
