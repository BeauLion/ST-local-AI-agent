"""
Shared fixtures for testing calendar_manager.py.

Nothing in this file (or in any test that uses these fixtures) makes a
real network call. The one seam calendar_manager.py has for reaching
iCloud is _get_calendars() - every read/write function goes through it -
so that's the single point these fixtures patch.
"""

import sys
from pathlib import Path

import pytest

# Make the project root importable when pytest is run from anywhere
# (e.g. `pytest` from the repo root, or from inside tests/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import calendar_manager as cm  # noqa: E402  (import after sys.path fix, deliberately)

from tests.fakes.fake_caldav import FakeCalendar  # noqa: E402


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
    """The main fixture most tests will use. Provides a list containing
    one FakeCalendar named 'Home', and monkeypatches
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
