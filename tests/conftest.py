"""
Shared fixtures for testing calendar_manager.py, duration_manager.py,
project_manager.py, console_log.py, memory.py, and attire_manager.py.

CRITICAL ORDERING:
- install_fake_memory_module() MUST run before duration_manager is
  imported anywhere - duration_manager does `import memory`, and the
  real memory.py loads an actual SentenceTransformer model at module
  IMPORT time, not lazily. Since conftest.py is always collected by
  pytest before test modules, doing the fake-install here, before the
  `import duration_manager` line below, guarantees every test file's
  own `import duration_manager as dm` transparently gets the fake.
- install_fake_sentence_transformers_module() MUST run before the real
  memory.py (imported below as plain `import memory`, for the
  memory.py test suite itself) is imported anywhere - memory.py does
  `from sentence_transformers import SentenceTransformer` and
  instantiates it at module IMPORT time (`_model =
  SentenceTransformer(...)`), not lazily. install_fake_memory_module()
  above only fakes the top-level `memory` module (for duration_manager
  and project_manager's benefit); it doesn't touch
  sentence_transformers, so this is still needed before the real
  memory.py loads.

Nothing in this file (or in any test that uses these fixtures) makes a
real network call. calendar_manager.py's one seam for reaching iCloud is
_get_calendars() - every read/write function goes through it - so
that's the single point the calendar fixtures patch.
"""
import sys
from pathlib import Path

import pytest

# Make the project root importable when pytest is run from anywhere
# (e.g. `pytest` from the repo root, or from inside tests/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import calendar_manager as cm  # noqa: E402  (import after sys.path fix, deliberately)
import attire_manager as am  # noqa: E402
import console_log  # noqa: E402

from tests.fakes.fake_caldav import FakeCalendar  # noqa: E402
from tests.fakes.fake_memory import install_fake_memory_module  # noqa: E402

install_fake_memory_module()

import duration_manager as dm  # noqa: E402

# Safe to import for real at this point: memory is already faked above,
# and by now sys.modules["duration_manager"] is the genuine module (the
# line above), so project_manager's own `import duration_manager` inside
# project_manager.py resolves to that real module too - harmless, since
# project_manager's tests override the `duration_manager` ATTRIBUTE on
# this already-imported module directly (see fake_duration fixture
# below), not via sys.modules. See fake_duration_manager.py's docstring
# for why a second sys.modules swap isn't used here.
import project_manager as pm  # noqa: E402

from tests.fakes.fake_duration_manager import FakeDurationModule  # noqa: E402
from tests.fakes.fake_sentence_transformers import install_fake_sentence_transformers_module  # noqa: E402

# sys.modules["memory"] is currently the FAKE module installed by
# install_fake_memory_module() above (for duration_manager/
# project_manager). The memory.py test suite needs the REAL module, so
# fake sentence_transformers first, then drop the fake memory module
# from sys.modules and import the real one fresh under the same name.
# duration_manager/project_manager have already bound their own
# `memory` reference (dm.memory, pm's indirect use via duration_manager)
# to the fake, so this swap doesn't affect them.
install_fake_sentence_transformers_module()
del sys.modules["memory"]
import memory  # noqa: E402

import prompt_log_engine as pe  # noqa: E402

# ---------------------------------------------------------------------------
# calendar_manager fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_calendar_state():
    """Runs before AND after every test automatically (autouse=True, no
    test needs to request it by name). Without this, calendar_manager's
    module-level globals - the cached client, the cached calendar list,
    the currently staged pending change, the last list/search results -
    would leak from one test into the next, since they're plain module
    attributes, not something recreated per-test on their own."""
    def _clear():
        cm._cached_client = None
        cm._cached_calendars = None
        cm._pending_change = None
        with cm._last_results_lock:
            cm._last_results.clear()

    _clear()
    yield
    _clear()


@pytest.fixture
def fake_calendars(monkeypatch):
    """The main fixture most calendar tests use. Provides a list
    containing one FakeCalendar named 'Home', and monkeypatches
    calendar_manager._get_calendars to return it - so any function under
    test (list_events, stage_create_event, etc.) thinks it's talking to
    a real iCloud account.

    Returns the list itself so a test can add more calendars, or reach
    into calendars[0].seed_event(...) to set up existing data before
    calling the function under test.
    """
    calendars = [FakeCalendar(name="Home")]

    def _get_calendars(refresh: bool = False):
        return calendars

    monkeypatch.setattr(cm, "_get_calendars", _get_calendars)
    return calendars


@pytest.fixture
def tmp_cache_file(tmp_path, monkeypatch):
    """Points calendar_manager's on-disk context cache at a throwaway
    file under pytest's own tmp_path, so refresh_cache()/get_cached_context()/
    check_availability()'s cache-hit path never read or write the real
    calendar_data/context_cache.json on this machine."""
    fake_path = tmp_path / "context_cache.json"
    monkeypatch.setattr(cm, "_CACHE_FILE", fake_path)
    return fake_path


# ---------------------------------------------------------------------------
# duration_manager fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_duration_state():
    """Runs before AND after every test. Clears the module-level category
    embedding cache (dict keyed by category name -> vector) so a fake
    vector configured in one test can't leak into the next, and resets
    the fake embedder's configured vectors/default back to a clean
    slate."""
    def _clear():
        dm._category_embedding_cache.clear()
        dm.memory._embedder.reset()

    _clear()
    yield
    _clear()


@pytest.fixture
def fake_embedder():
    """Direct access to the fake embed() function's controls - use
    .set(text, vector) to pin what memory.embed() returns for a specific
    string, or .set_default(vector) to change the fallback."""
    return dm.memory._embedder


@pytest.fixture
def tmp_duration_file(tmp_path, monkeypatch):
    """Points duration_manager's on-disk state file at a throwaway file
    under pytest's own tmp_path, so no test ever reads or writes the
    real duration_data/durations.json on this machine."""
    fake_path = tmp_path / "durations.json"
    monkeypatch.setattr(dm, "STATE_FILE", fake_path)
    return fake_path


# ---------------------------------------------------------------------------
# project_manager fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project_file(tmp_path, monkeypatch):
    """Points project_manager's on-disk state file at a throwaway file
    under pytest's own tmp_path, so no test ever reads or writes the
    real project_data/projects.json on this machine."""
    fake_path = tmp_path / "projects.json"
    monkeypatch.setattr(pm, "STATE_FILE", fake_path)
    return fake_path


@pytest.fixture
def fake_duration(monkeypatch):
    """Replaces project_manager.duration_manager (the module ATTRIBUTE,
    not sys.modules - see fake_duration_manager.py) with a fresh
    FakeDurationModule for the duration of one test. Returns the
    controller object so a test can inspect .calls or configure
    .set_done_return(...)."""
    fake_module = FakeDurationModule()
    monkeypatch.setattr(pm, "duration_manager", fake_module)
    return fake_module.controller


# ---------------------------------------------------------------------------
# console_log fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_console_log_buffer():
    """console_log._lines is a plain module-level list, not something
    recreated per-test on its own - without this, buffered lines from one
    test would leak into the next test's flush()."""
    console_log._lines.clear()
    yield
    console_log._lines.clear()


# ---------------------------------------------------------------------------
# memory.py fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_memory_embedder():
    """Direct access to memory.py's (fake) model instance - use
    .set(text, vector) to pin what encode() returns for a specific
    string, or .set_default(vector) to change the fallback. Reset
    automatically between tests by reset_memory_state below.

    Named distinctly from the duration_manager fake_embedder fixture
    above - that one controls dm.memory._embedder (the fake `memory`
    module duration_manager imports), this one controls the real
    memory.py's own fake SentenceTransformer instance. They are
    unrelated fakes for unrelated modules and must not share a name."""
    return memory._model


@pytest.fixture(autouse=True)
def reset_memory_state():
    """Resets the fake embedder's configured vectors between tests, same
    reasoning as duration_manager's reset_duration_state fixture."""
    memory._model.reset()
    yield
    memory._model.reset()


@pytest.fixture
def tmp_memory_file(tmp_path, monkeypatch):
    """Points memory.py's on-disk personal-memory store at a throwaway
    file under pytest's own tmp_path, so no test ever reads or writes the
    real memory_data/memories.json on this machine."""
    fake_path = tmp_path / "memories.json"
    monkeypatch.setattr(memory, "MEMORY_FILE", fake_path)
    return fake_path


@pytest.fixture
def tmp_doc_index_file(tmp_path, monkeypatch):
    """Points memory.py's on-disk document index at a throwaway file
    under pytest's own tmp_path, so no test ever reads or writes the
    real memory_data/doc_index.json on this machine."""
    fake_path = tmp_path / "doc_index.json"
    monkeypatch.setattr(memory, "DOC_INDEX_FILE", fake_path)
    return fake_path


# ---------------------------------------------------------------------------
# prompt_log_engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_log_dir(tmp_path, monkeypatch):
    """Redirects every prompt_log_engine function at a throwaway
    directory, and points SESSION_LOG_PATH at a fixed, known filename
    inside it - so tests never read/write the real prompt_logs/ folder
    or depend on the real wall-clock session filename. Both
    _PROMPT_LOG_DIR and SESSION_LOG_PATH are plain module-level globals
    referenced fresh inside every function that uses them (not captured
    in a closure at def time), so monkeypatching the attribute here
    redirects every function correctly."""
    monkeypatch.setattr(pe, "_PROMPT_LOG_DIR", tmp_path)
    monkeypatch.setattr(pe, "SESSION_LOG_PATH", tmp_path / "session_test.log")
    return tmp_path


@pytest.fixture
def prompt_log_enabled(monkeypatch):
    """Forces PROMPT_LOG_ENABLED on for a test, regardless of config.py's
    real value. Monkeypatches the ATTRIBUTE on prompt_log_engine, not
    config - PROMPT_LOG_ENABLED was pulled in via `from config import
    PROMPT_LOG_ENABLED`, a one-time name binding at import, not a live
    reference back to config.py."""
    monkeypatch.setattr(pe, "PROMPT_LOG_ENABLED", True)


@pytest.fixture
def prompt_log_disabled(monkeypatch):
    """Forces PROMPT_LOG_ENABLED off for a test."""
    monkeypatch.setattr(pe, "PROMPT_LOG_ENABLED", False)


@pytest.fixture
def client():
    """A FastAPI TestClient wired up with just prompt_log_engine's
    router - mirrors how main.py wires it into the real app
    (`app.include_router(prompt_log_router)`), without pulling in the
    rest of main.py's own routes/state."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(pe.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# attire_manager fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_attire_file(tmp_path, monkeypatch):
    """Points attire_manager's on-disk store (attire.json) at a throwaway
    file under pytest's own tmp_path, so tests never read or write the
    real attire_data/attire.json on this machine. Unlike calendar_manager,
    attire_manager has no other module-level state to reset between tests
    (no cached client, no pending-change lifecycle) - the file itself,
    freshly pointed at an empty tmp_path each test, is the only thing that
    needs isolating."""
    fake_path = tmp_path / "attire.json"
    monkeypatch.setattr(am, "STATE_FILE", fake_path)
    return fake_path