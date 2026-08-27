"""
Fake stand-in for duration_manager.py, used only by project_manager's own
tests. project_manager.py imports duration_manager directly
(on_task_active/inactive/done, parse_duration_minutes) - these tests only
need to verify project_manager CALLS duration_manager correctly (right
function, right arguments, right timing), not that duration_manager's
own logic is correct (already covered end-to-end by
tests/test_duration_*.py).

DELIBERATELY NOT a sys.modules swap (unlike fake_memory.py): by the time
project_manager's tests run in this same pytest session, conftest.py has
already done a real `import duration_manager as dm` for duration_manager's
OWN test suite - sys.modules["duration_manager"] is genuinely the real
module by then, and duration_manager's tests need it to STAY that way.
So instead, this fake is applied by directly monkeypatching the
`duration_manager` ATTRIBUTE on the already-imported project_manager
module object (`monkeypatch.setattr(pm, "duration_manager", fake)`) -
this only changes what project_manager.py itself sees when it calls
`duration_manager.whatever(...)`, and doesn't touch sys.modules or
duration_manager's own tests at all.

parse_duration_minutes is a deliberate exception in HOW it's faked, not
IN THAT it's faked: project_manager.py's "dur:" note-tag parsing
(_parse_note_tags) needs SOME real-shaped parser to decide tag validity
in its own tests. This fake ships its own small, deliberately-simplified
parser instead of reusing the genuine one, to keep project_manager's
tests from silently depending on duration_manager's exact parsing edge
cases - those are already covered in test_duration_parsing.py. This fake
covers the same basic shapes (bare number, number+unit, "1h30m"
compound) but is NOT a byte-for-byte copy of the real regexes and isn't
meant to be.
"""
import re


def fake_parse_duration_minutes(value_text) -> float | None:
    clean = re.sub(r"\s+", "", str(value_text or "").strip().lower())
    if not clean:
        return None
    compound = re.match(r"^(\d+(?:\.\d+)?)h(\d+(?:\.\d+)?)m$", clean)
    if compound:
        return float(compound.group(1)) * 60 + float(compound.group(2))
    hour = re.match(r"^(\d+(?:\.\d+)?)(?:h|hr|hrs|hour|hours)$", clean)
    if hour:
        return float(hour.group(1)) * 60
    minute = re.match(r"^(\d+(?:\.\d+)?)(?:m|min|mins|minute|minutes)?$", clean)
    if minute:
        return float(minute.group(1))
    return None


class FakeDurationManager:
    """The object a test's `fake_duration` fixture hands back for
    inspection/control. Records every on_task_active/inactive/done call
    as a tuple in .calls, and lets a test script what on_task_done
    should return next (a fixed value, or a callable for per-call
    logic) via set_done_return()."""

    def __init__(self):
        self.calls: list[tuple] = []
        self._done_return = None

    def on_task_active(self, task_id, project_id) -> None:
        self.calls.append(("active", task_id, project_id))

    def on_task_inactive(self, task_id) -> None:
        self.calls.append(("inactive", task_id))

    def on_task_done(self, task_id, project_id, title) -> str | None:
        self.calls.append(("done", task_id, project_id, title))
        if callable(self._done_return):
            return self._done_return(task_id, project_id, title)
        return self._done_return

    def set_done_return(self, value) -> None:
        self._done_return = value


class FakeDurationModule:
    """A lightweight stand-in with the same four names project_manager.py
    actually calls - passed to monkeypatch.setattr(pm, "duration_manager",
    this), NOT inserted into sys.modules."""

    def __init__(self):
        self.controller = FakeDurationManager()
        self.parse_duration_minutes = fake_parse_duration_minutes
        self.on_task_active = self.controller.on_task_active
        self.on_task_inactive = self.controller.on_task_inactive
        self.on_task_done = self.controller.on_task_done
