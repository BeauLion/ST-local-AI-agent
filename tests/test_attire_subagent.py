"""
Tests for attire_subagent.py: the post-turn one-shot completion call,
how it builds the request (system prompt, current-state summary, the
last exchange, the tool schemas borrowed from main.py), and how it
dispatches/handles whatever tool_calls come back - including every
fail-open path (network failure, non-2xx response, malformed tool
arguments, an unmatched/ignored tool name), since this module is
explicitly designed to never raise back into a real conversation turn.

Two things are faked rather than exercised for real:

- httpx.AsyncClient: replaced with _FakeAsyncClient, an in-memory async
  context manager that records every .post() call and returns whatever
  response (or exception) a test configures - no real network call, no
  real llama-server needed to run this suite.
- The `main` module: run_attire_subagent() does a DEFERRED `import main
  as agent_main` inside the function body specifically to avoid a
  circular import with the real main.py (see that module's docstring).
  Importing the real main.py here would mean building the whole FastAPI
  app, its embeddings-based tool selector, etc. just to test this one
  module's own dispatch logic - instead, tests install a lightweight
  fake module object into sys.modules["main"] via monkeypatch, exposing
  just the four names attire_subagent.py actually touches
  (ATTIRE_TOOL_SCHEMAS and the three mutating wrapper functions), and
  recording every wrapper call for assertions.

No pytest-asyncio dependency: run_attire_subagent is invoked directly
via asyncio.run() inside otherwise-ordinary sync test functions.
"""
import asyncio
import json
import sys
import types

import httpx
import pytest

import attire_manager as am
import attire_subagent as sa
import console_log
from config import AGENT_API_KEY, LLAMA_SERVER_URL


# ---------------------------------------------------------------------------
# fakes / fixtures
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Stands in for an httpx.Response - just enough surface for
    run_attire_subagent's .raise_for_status() / .json() calls."""

    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}", request=None, response=self
            )

    def json(self):
        return self._json_data


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient as an async context manager.
    Records every .post() call as (url, json_body) in .calls, and every
    construction's kwargs in .init_kwargs - so a test can assert on the
    Authorization header without a real HTTP stack. Configure the
    response via set_response(), or force the request to raise via
    set_exception()."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.init_kwargs: dict = {}
        self._response = _FakeResponse({"choices": [{"message": {}}]})
        self._exception = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None):
        self.calls.append((url, json))
        if self._exception is not None:
            raise self._exception
        return self._response

    def set_response(self, json_data, status_code=200):
        self._response = _FakeResponse(json_data, status_code)

    def set_exception(self, exc):
        self._exception = exc


@pytest.fixture
def fake_llama_client(monkeypatch):
    """Points attire_subagent's httpx.AsyncClient(...) constructor at a
    single shared _FakeAsyncClient instance for the test, capturing
    whatever kwargs it was constructed with (timeout/headers) onto that
    same instance."""
    client = _FakeAsyncClient()

    def _factory(*args, **kwargs):
        client.init_kwargs = kwargs
        return client

    monkeypatch.setattr(sa.httpx, "AsyncClient", _factory)
    return client


@pytest.fixture
def fake_main_module(monkeypatch):
    """Installs a lightweight stand-in for main.py into sys.modules, so
    run_attire_subagent's deferred `import main as agent_main` picks
    this up instead of importing (and building the FastAPI app inside)
    the real main.py. Records every wrapper call as (name, args) in
    .calls; each wrapper returns a fixed, inspectable string."""
    fake = types.ModuleType("main")
    calls: list[tuple] = []

    def _make_wrapper(name):
        def _wrapper(args):
            calls.append((name, args))
            return f"{name} applied"
        return _wrapper

    fake.ATTIRE_TOOL_SCHEMAS = ["fake-attire-tool-schema"]
    fake.attire_manager_add_item = _make_wrapper("attire_manager_add_item")
    fake.attire_manager_remove_item = _make_wrapper("attire_manager_remove_item")
    fake.attire_manager_replace_slot = _make_wrapper("attire_manager_replace_slot")
    fake.calls = calls

    monkeypatch.setitem(sys.modules, "main", fake)
    return fake


@pytest.fixture(autouse=True)
def fake_prompt_log(monkeypatch):
    """Records every log_prompt()/log_console() call attire_subagent.py
    makes, without touching the real prompt-log file on disk. Autouse:
    every test in this file gets this for free, since otherwise every
    single test (not just the ones specifically about logging) would hit
    the real prompt_log_engine and its on-disk session file. This suite
    isn't re-testing prompt_log_engine's own file I/O (that's its own
    module's concern) - just confirming attire_subagent.py calls it with
    sensible, inspectable arguments, the same way fake_main_module stands
    in for main.py rather than re-testing main.py itself."""
    calls = {"prompt": [], "console": []}

    def _log_prompt(*args, **kwargs):
        calls["prompt"].append((args, kwargs))

    def _log_console(*args, **kwargs):
        calls["console"].append((args, kwargs))

    monkeypatch.setattr(sa, "log_prompt", _log_prompt)
    monkeypatch.setattr(sa, "log_console", _log_console)
    return calls


def _tool_call(name: str, arguments) -> dict:
    """Builds one OpenAI-shaped tool_call dict. `arguments` may be a
    dict (auto-serialized to JSON, the normal case) or a raw string
    (for deliberately-malformed-JSON tests)."""
    args_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"function": {"name": name, "arguments": args_str}}


def _run(user_text: str, assistant_text: str):
    return asyncio.run(sa.run_attire_subagent(user_text, assistant_text))


# ---------------------------------------------------------------------------
# early return - nothing to look at
# ---------------------------------------------------------------------------

def test_makes_no_request_when_both_texts_are_blank(fake_llama_client, fake_main_module):
    _run("", "")
    assert fake_llama_client.calls == []


def test_whitespace_only_text_is_treated_as_blank(fake_llama_client, fake_main_module):
    _run("   ", "\n\t  ")
    assert fake_llama_client.calls == []


def test_proceeds_when_only_one_side_has_content(tmp_attire_file, fake_llama_client, fake_main_module):
    """A blank user turn (e.g. a slash command with no narrative) with a
    real assistant reply should still be scanned - only BOTH being blank
    should short-circuit."""
    _run("", "He shrugs off his jacket.")
    assert len(fake_llama_client.calls) == 1


# ---------------------------------------------------------------------------
# request construction
# ---------------------------------------------------------------------------

def test_sends_exactly_one_request(tmp_attire_file, fake_llama_client, fake_main_module):
    _run("hello", "hi there")
    assert len(fake_llama_client.calls) == 1


def test_posts_to_the_configured_llama_server_chat_endpoint(tmp_attire_file, fake_llama_client, fake_main_module):
    _run("hello", "hi there")
    url, _ = fake_llama_client.calls[0]
    assert url == f"{LLAMA_SERVER_URL}/v1/chat/completions"


def test_sends_the_configured_api_key_as_a_bearer_header(tmp_attire_file, fake_llama_client, fake_main_module):
    _run("hello", "hi there")
    assert fake_llama_client.init_kwargs["headers"]["Authorization"] == f"Bearer {AGENT_API_KEY}"


def test_request_is_non_streaming_with_auto_tool_choice(tmp_attire_file, fake_llama_client, fake_main_module):
    _run("hello", "hi there")
    _, body = fake_llama_client.calls[0]
    assert body["stream"] is False
    assert body["tool_choice"] == "auto"


def test_request_uses_the_tool_schemas_from_the_main_module(tmp_attire_file, fake_llama_client, fake_main_module):
    """Confirms the sub-agent reuses main.py's ATTIRE_TOOL_SCHEMAS rather
    than a second hand-copied schema that could drift."""
    _run("hello", "hi there")
    _, body = fake_llama_client.calls[0]
    assert body["tools"] == fake_main_module.ATTIRE_TOOL_SCHEMAS


def test_messages_include_system_prompt_then_state_then_user_then_assistant(
    tmp_attire_file, fake_llama_client, fake_main_module
):
    _run("He takes off his shoes.", "You watch him kick them aside.")
    _, body = fake_llama_client.calls[0]
    messages = body["messages"]

    assert messages[0]["role"] == "system"
    assert "continuity tracker" in messages[0]["content"]
    assert messages[1]["role"] == "system"
    assert "[CURRENT ATTIRE STATE]" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "He takes off his shoes."}
    assert messages[3] == {"role": "assistant", "content": "You watch him kick them aside."}


def test_state_summary_reports_no_tracked_characters_when_state_is_empty(
    tmp_attire_file, fake_llama_client, fake_main_module
):
    _run("hello", "hi there")
    _, body = fake_llama_client.calls[0]
    assert "No characters are currently tracked." in body["messages"][1]["content"]


def test_state_summary_includes_a_tracked_characters_current_slots(
    tmp_attire_file, fake_llama_client, fake_main_module
):
    am.add_item("Aria", "feet", "yellow socks")

    _run("He puts black shoes on her feet.", "She wiggles her toes in them.")

    _, body = fake_llama_client.calls[0]
    assert "Aria" in body["messages"][1]["content"]
    assert "feet: yellow socks" in body["messages"][1]["content"]


# ---------------------------------------------------------------------------
# no tool calls - the common, expected outcome
# ---------------------------------------------------------------------------

def test_no_tool_calls_applies_nothing(tmp_attire_file, fake_llama_client, fake_main_module):
    fake_llama_client.set_response({"choices": [{"message": {}}]})

    _run("The weather was nice today.", "It really was.")

    assert fake_main_module.calls == []


def test_no_tool_calls_logs_that_nothing_changed(tmp_attire_file, fake_llama_client, fake_main_module):
    fake_llama_client.set_response({"choices": [{"message": {}}]})
    console_log._lines.clear()

    _run("The weather was nice today.", "It really was.")

    assert any("No attire change detected" in line for line in console_log._lines)


# ---------------------------------------------------------------------------
# dispatch - routing a tool call to the right wrapper
# ---------------------------------------------------------------------------

def test_dispatches_add_item_call_to_the_add_item_wrapper(tmp_attire_file, fake_llama_client, fake_main_module):
    args = {"character_name": "Aria", "slot": "feet", "item": "black shoes"}
    fake_llama_client.set_response({
        "choices": [{"message": {"tool_calls": [_tool_call("attire_add_item", args)]}}]
    })

    _run("She puts on black shoes.", "The shoes click on the floor.")

    assert fake_main_module.calls == [("attire_manager_add_item", args)]


def test_dispatches_remove_item_call_to_the_remove_item_wrapper(tmp_attire_file, fake_llama_client, fake_main_module):
    args = {"character_name": "Aria", "slot": "top", "item_hint": "jacket"}
    fake_llama_client.set_response({
        "choices": [{"message": {"tool_calls": [_tool_call("attire_remove_item", args)]}}]
    })

    _run("She shrugs off her jacket.", "It falls to the floor.")

    assert fake_main_module.calls == [("attire_manager_remove_item", args)]


def test_dispatches_replace_slot_call_to_the_replace_slot_wrapper(tmp_attire_file, fake_llama_client, fake_main_module):
    args = {"character_name": "Aria", "slot": "top", "items": "red sundress"}
    fake_llama_client.set_response({
        "choices": [{"message": {"tool_calls": [_tool_call("attire_replace_slot", args)]}}]
    })

    _run("She changes into a red sundress.", "It suits her.")

    assert fake_main_module.calls == [("attire_manager_replace_slot", args)]


def test_applies_multiple_tool_calls_from_one_response_in_order(tmp_attire_file, fake_llama_client, fake_main_module):
    add_args = {"character_name": "Aria", "slot": "feet", "item": "black shoes"}
    remove_args = {"character_name": "Aria", "slot": "feet", "item_hint": "sandals"}
    fake_llama_client.set_response({
        "choices": [{"message": {"tool_calls": [
            _tool_call("attire_remove_item", remove_args),
            _tool_call("attire_add_item", add_args),
        ]}}]
    })

    _run("She kicks off her sandals and puts on black shoes.", "Much better.")

    assert fake_main_module.calls == [
        ("attire_manager_remove_item", remove_args),
        ("attire_manager_add_item", add_args),
    ]


def test_ignores_attire_manager_get_tool_call(tmp_attire_file, fake_llama_client, fake_main_module):
    """attire_manager_get is read-only - even if the model calls it, it
    must never be dispatched (there's no wrapper for it in fake_main
    at all, so a dispatch attempt would raise a KeyError if this
    weren't filtered out first)."""
    args = {"character_name": "Aria"}
    fake_llama_client.set_response({
        "choices": [{"message": {"tool_calls": [_tool_call("attire_manager_get", args)]}}]
    })

    _run("What is she wearing?", "Let me check.")

    assert fake_main_module.calls == []


def test_ignores_an_unrecognized_tool_name(tmp_attire_file, fake_llama_client, fake_main_module):
    fake_llama_client.set_response({
        "choices": [{"message": {"tool_calls": [_tool_call("some_other_tool", {})]}}]
    })

    _run("hello", "hi there")

    assert fake_main_module.calls == []


# ---------------------------------------------------------------------------
# fail-open behavior
# ---------------------------------------------------------------------------

def test_malformed_arguments_are_skipped_without_crashing_the_rest(
    tmp_attire_file, fake_llama_client, fake_main_module
):
    """One tool call with unparseable JSON arguments must not stop a
    later, well-formed tool call in the same response from applying."""
    good_args = {"character_name": "Aria", "slot": "feet", "item": "black shoes"}
    fake_llama_client.set_response({
        "choices": [{"message": {"tool_calls": [
            _tool_call("attire_add_item", "{not valid json"),
            _tool_call("attire_add_item", good_args),
        ]}}]
    })

    _run("hello", "hi there")

    assert fake_main_module.calls == [("attire_manager_add_item", good_args)]


def test_malformed_arguments_are_logged(tmp_attire_file, fake_llama_client, fake_main_module):
    console_log._lines.clear()
    fake_llama_client.set_response({
        "choices": [{"message": {"tool_calls": [_tool_call("attire_add_item", "{not valid json")]}}]
    })

    _run("hello", "hi there")

    assert any("Malformed tool arguments" in line for line in console_log._lines)


def test_network_failure_is_caught_and_applies_nothing(tmp_attire_file, fake_llama_client, fake_main_module):
    fake_llama_client.set_exception(httpx.ConnectError("connection refused"))

    _run("hello", "hi there")  # must not raise

    assert fake_main_module.calls == []


def test_network_failure_is_logged(tmp_attire_file, fake_llama_client, fake_main_module):
    console_log._lines.clear()
    fake_llama_client.set_exception(httpx.ConnectError("connection refused"))

    _run("hello", "hi there")

    assert any("Request to llama-server failed" in line for line in console_log._lines)


def test_non_2xx_response_is_caught_and_applies_nothing(tmp_attire_file, fake_llama_client, fake_main_module):
    fake_llama_client.set_response({"error": "internal error"}, status_code=500)

    _run("hello", "hi there")  # must not raise

    assert fake_main_module.calls == []


# ---------------------------------------------------------------------------
# prompt-log-viewer integration
# ---------------------------------------------------------------------------

def test_does_not_log_anything_when_returning_early_for_blank_input(
    tmp_attire_file, fake_llama_client, fake_main_module, fake_prompt_log
):
    _run("", "")
    assert fake_prompt_log["prompt"] == []
    assert fake_prompt_log["console"] == []


def test_logs_the_outgoing_request_with_four_section_labels(
    tmp_attire_file, fake_llama_client, fake_main_module, fake_prompt_log
):
    """The four labels must line up positionally with the four messages
    this module always sends (system prompt, current state, user,
    assistant) - see log_prompt()'s section_labels contract."""
    _run("He takes off his shoes.", "You watch him kick them aside.")

    assert len(fake_prompt_log["prompt"]) == 1
    (body, iteration, section_labels), _ = fake_prompt_log["prompt"][0]
    assert iteration == 0
    assert section_labels == [
        "attire_subagent_system",
        "attire_subagent_current_state",
        "attire_subagent_user_turn",
        "attire_subagent_assistant_turn",
    ]
    assert len(section_labels) == len(body["messages"])


def test_logs_the_exact_body_that_was_sent_to_llama_server(
    tmp_attire_file, fake_llama_client, fake_main_module, fake_prompt_log
):
    _run("hello", "hi there")

    (logged_body, _, _), _ = fake_prompt_log["prompt"][0]
    _, posted_body = fake_llama_client.calls[0]
    assert logged_body is posted_body


def test_logs_no_change_outcome_with_kind_content(
    tmp_attire_file, fake_llama_client, fake_main_module, fake_prompt_log
):
    fake_llama_client.set_response({"choices": [{"message": {}}]})

    _run("The weather was nice today.", "It really was.")

    (iteration, response), kwargs = fake_prompt_log["console"][0]
    assert iteration == 0
    assert response["kind"] == "content"


def test_logs_tool_calls_outcome_with_kind_tool_calls_and_raw_payload(
    tmp_attire_file, fake_llama_client, fake_main_module, fake_prompt_log
):
    args = {"character_name": "Aria", "slot": "feet", "item": "black shoes"}
    calls = [_tool_call("attire_add_item", args)]
    fake_llama_client.set_response({"choices": [{"message": {"tool_calls": calls}}]})

    _run("She puts on black shoes.", "The shoes click on the floor.")

    (iteration, response), kwargs = fake_prompt_log["console"][0]
    assert iteration == 0
    assert response["kind"] == "tool_calls"
    assert json.loads(response["text"]) == calls


def test_logs_network_failure_with_kind_error(
    tmp_attire_file, fake_llama_client, fake_main_module, fake_prompt_log
):
    fake_llama_client.set_exception(httpx.ConnectError("connection refused"))

    _run("hello", "hi there")

    assert fake_prompt_log["prompt"], "the request should still be logged even though it then failed"
    (iteration, response), kwargs = fake_prompt_log["console"][0]
    assert iteration == 0
    assert response["kind"] == "error"
    assert "connection refused" in response["text"]


def test_passes_reasoning_content_through_as_thinking_when_present(
    tmp_attire_file, fake_llama_client, fake_main_module, fake_prompt_log
):
    fake_llama_client.set_response({
        "choices": [{"message": {"reasoning_content": "she still has socks on underneath"}}]
    })

    _run("She puts on black shoes.", "The shoes click on the floor.")

    (_, _), kwargs = fake_prompt_log["console"][0]
    assert kwargs.get("thinking") == "she still has socks on underneath"
