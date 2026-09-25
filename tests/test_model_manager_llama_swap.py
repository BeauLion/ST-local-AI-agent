"""
Tests for model_manager.py's "llama-swap" backend (config.LLAMA_BACKEND):
the agent talks to llama-swap on another machine and never launches a
process itself. Covers boot-time model selection, surviving the remote
machine being down, swap success/failure (and that a failed swap keeps
the previous model selected), reset, and the URL/model helpers main.py
and attire_subagent.py route their requests through.

httpx.get is replaced with a small URL-keyed fake - no real network, no
real llama-swap. The background reachability poller is never started
except in the startup tests, which shut it down again immediately.
"""
import copy
import json

import httpx
import pytest

import config
import model_manager as mm

BASE = "http://gpu-box:8080"


class _FakeRemote:
    """Answers httpx.get() by URL. `models` is what /v1/models lists;
    `down` makes every request fail like an unreachable host; per-URL
    overrides go in `status_by_url` (an int status) or `json_by_url`."""

    def __init__(self):
        self.models = ["gemma-12b", "qwen3-14b"]
        self.down = False
        self.status_by_url: dict[str, int] = {}
        self.json_by_url: dict[str, dict] = {}
        self.calls: list[str] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        request = httpx.Request("GET", url)
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        status = self.status_by_url.get(url, 200)
        if url == f"{BASE}/v1/models":
            body = {"data": [{"id": m} for m in self.models]}
        else:
            body = self.json_by_url.get(url, {})
        return httpx.Response(status, json=body, request=request)


@pytest.fixture
def remote(monkeypatch, tmp_path):
    fake = _FakeRemote()
    monkeypatch.setattr(config, "LLAMA_BACKEND", "llama-swap")
    monkeypatch.setattr(config, "LLAMA_SERVER_URL", BASE)
    monkeypatch.setattr(config, "LLAMA_SWAP_DEFAULT_MODEL", "")
    monkeypatch.setattr(config, "LLAMA_SWAP_POLL_SECONDS", 3600)
    monkeypatch.setattr(mm, "_REMOTE_STATE_PATH", tmp_path / "llama_swap_state.json")
    monkeypatch.setattr(mm.httpx, "get", fake.get)
    saved_state = copy.deepcopy(mm._state)
    yield fake
    mm._poller_stop.set()
    mm._state.clear()
    mm._state.update(saved_state)


def _boot():
    mm.startup()
    mm._poller_stop.set()  # stop the poller but keep phase inspectable (shutdown() would set "stopped")
    return mm._state


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def test_startup_selects_the_first_listed_model_when_no_default_is_set(remote):
    mm.startup()
    try:
        st = mm.status()
        assert st["phase"] == "ready"
        assert st["repo"] == "gemma-12b"
    finally:
        mm.shutdown()


def test_startup_prefers_the_configured_default_model(remote, monkeypatch):
    monkeypatch.setattr(config, "LLAMA_SWAP_DEFAULT_MODEL", "qwen3-14b")
    assert _boot()["repo"] == "qwen3-14b"


def test_startup_restores_the_persisted_model(remote):
    mm._REMOTE_STATE_PATH.write_text(json.dumps({"model": "qwen3-14b"}), encoding="utf-8")
    assert _boot()["repo"] == "qwen3-14b"


def test_startup_falls_back_when_the_persisted_model_is_gone(remote):
    mm._REMOTE_STATE_PATH.write_text(json.dumps({"model": "deleted-model"}), encoding="utf-8")
    state = _boot()
    assert state["repo"] == "gemma-12b"
    assert "deleted-model" in state["last_error"]


def test_startup_never_launches_a_local_process(remote, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not launch llama-server on the llama-swap backend")
    monkeypatch.setattr(mm, "_launch", _boom)
    _boot()


def test_startup_survives_the_remote_machine_being_down(remote):
    remote.down = True
    state = _boot()
    assert state["phase"] == "unreachable"
    assert BASE in state["last_error"]


def test_refresh_recovers_once_the_remote_machine_is_back(remote):
    remote.down = True
    _boot()
    remote.down = False
    mm._refresh_remote()
    assert mm.status()["phase"] == "ready"
    assert mm.status()["last_error"] is None


def test_reachable_but_empty_model_list_is_an_error(remote):
    remote.models = []
    assert _boot()["phase"] == "error"


# ---------------------------------------------------------------------------
# swap / reset
# ---------------------------------------------------------------------------

def test_swap_loads_via_upstream_health_and_commits(remote):
    _boot()
    remote.json_by_url[f"{BASE}/upstream/qwen3-14b/props"] = {"default_generation_settings": {"n_ctx": 16384}}
    st = mm.swap(repo="qwen3-14b")
    assert f"{BASE}/upstream/qwen3-14b/health" in remote.calls
    assert st["phase"] == "ready"
    assert st["repo"] == "qwen3-14b"
    assert st["context"] == 16384
    assert json.loads(mm._REMOTE_STATE_PATH.read_text(encoding="utf-8")) == {"model": "qwen3-14b"}


def test_swap_to_an_unknown_model_keeps_the_previous_one(remote):
    _boot()
    st = mm.swap(repo="nope")
    assert st["repo"] == "gemma-12b"
    assert st["phase"] == "ready"
    assert "nope" in st["last_error"]


def test_swap_that_fails_to_load_keeps_the_previous_one(remote):
    _boot()
    remote.status_by_url[f"{BASE}/upstream/qwen3-14b/health"] = 502
    st = mm.swap(repo="qwen3-14b")
    assert st["repo"] == "gemma-12b"
    assert "qwen3-14b" in st["last_error"]


def test_swap_ignores_local_only_arguments(remote):
    _boot()
    st = mm.swap(repo="qwen3-14b", ngl=10, context=999, md="x.gguf", spec_type="y")
    assert st["repo"] == "qwen3-14b"
    assert st["ngl"] is None


def test_swap_while_swapping_is_rejected(remote):
    _boot()
    mm._state["phase"] = "swapping"
    with pytest.raises(mm.SwapError):
        mm.swap(repo="qwen3-14b")


def test_reset_goes_to_the_configured_default(remote, monkeypatch):
    _boot()
    mm.swap(repo="qwen3-14b")
    monkeypatch.setattr(config, "LLAMA_SWAP_DEFAULT_MODEL", "gemma-12b")
    assert mm.reset()["repo"] == "gemma-12b"


def test_reset_without_a_default_goes_to_the_first_listed_model(remote):
    _boot()
    mm.swap(repo="qwen3-14b")
    assert mm.reset()["repo"] == "gemma-12b"


# ---------------------------------------------------------------------------
# request helpers
# ---------------------------------------------------------------------------

def test_request_model_and_llama_url_route_through_the_active_model(remote):
    _boot()
    assert mm.request_model() == "gemma-12b"
    assert mm.llama_url("/tokenize") == f"{BASE}/upstream/gemma-12b/tokenize"


def test_local_backend_leaves_model_and_urls_alone(monkeypatch):
    monkeypatch.setattr(config, "LLAMA_BACKEND", "local")
    monkeypatch.setattr(config, "LLAMA_SERVER_URL", "http://localhost:8080")
    assert mm.request_model() is None
    assert mm.llama_url("/tokenize") == "http://localhost:8080/tokenize"


def test_llama_headers_use_the_llama_api_key(monkeypatch):
    monkeypatch.setattr(config, "LLAMA_API_KEY", "secret")
    assert mm.llama_headers() == {"Authorization": "Bearer secret"}
