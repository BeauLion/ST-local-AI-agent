"""
Fake stand-ins for the caldav objects calendar_manager.py talks to.

Design choice: these fakes wrap REAL icalendar.Event objects, not plain
dicts. calendar_manager.py's read path (_ical_field) and its edit path
(stage_edit_event's apply_fn, which does
`comp["DTSTART"] = icalendar.vDDDTypes(start_dt)`) both depend on
icalendar's actual type-wrapping behavior (a datetime you .add() comes
back out of .get() as something with a `.dt` attribute). Reimplementing
that behavior by hand in a fake would risk testing the fake instead of
the real code. Only the network-facing caldav.DAVClient/Calendar layer
is faked - icalendar itself is the real library.

Known limitation: only timed (non-all-day) events are supported here.
All-day events use `date` instead of `datetime` for DTSTART/DTEND, which
would need extra handling in date_search()'s overlap check below if a
future test needs one.
"""

import uuid
from contextlib import contextmanager
from datetime import datetime

import icalendar


def _strip_tz(dt: datetime) -> datetime:
    """caldav's real date_search() accepts naive or aware datetimes; this
    fake only compares naive ones internally so tests don't have to worry
    about matching timezone-awareness between the query range and the
    seeded event data."""
    if isinstance(dt, datetime) and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


class FakeEvent:
    """Stands in for a caldav.Event. Wraps one real icalendar.Event
    component. Tracks .saved/.deleted so tests can assert on them
    directly instead of re-reading calendar state afterward."""

    def __init__(self, component: "icalendar.Event", calendar: "FakeCalendar"):
        self._component = component
        self._calendar = calendar
        self.saved = False
        self.deleted = False

    @property
    def icalendar_component(self):
        return self._component

    @contextmanager
    def edit_icalendar_component(self):
        # Real caldav's version is also just a context manager around the
        # same live component object - callers mutate it in place, no
        # extra bookkeeping needed here on exit.
        yield self._component

    def save(self):
        self.saved = True

    def delete(self):
        self.deleted = True
        if self in self._calendar.events:
            self._calendar.events.remove(self)


class FakeCalendar:
    """Stands in for a caldav.Calendar. Holds an in-memory list of
    FakeEvent and implements just the two methods calendar_manager.py
    calls on a calendar object: .add_event() and .date_search()."""

    def __init__(self, name: str = "Home"):
        self.name = name
        self.events: list[FakeEvent] = []

    def add_event(self, dtstart, dtend, summary, location=None, description=None, uid=None):
        """Mirrors caldav.Calendar.add_event()'s call signature. Used both
        by calendar_manager.py directly (stage_create_event's apply_fn)
        and by seed_event() below for test setup."""
        comp = icalendar.Event()
        comp.add("uid", uid or str(uuid.uuid4()))
        comp.add("summary", summary)
        comp.add("dtstart", dtstart)
        comp.add("dtend", dtend)
        if location:
            comp.add("location", location)
        if description:
            comp.add("description", description)
        event = FakeEvent(comp, self)
        self.events.append(event)
        return event

    def seed_event(self, title, start, end=None, location="", description="", uid=None):
        """Test-setup convenience wrapper around add_event() - clearer name
        for 'this event already exists before the test starts', and lets
        a test pin a specific uid up front for UID-lookup tests."""
        from datetime import timedelta
        end = end or start + timedelta(hours=1)
        return self.add_event(start, end, title, location, description, uid)

    def date_search(self, start, end):
        results = []
        q_start, q_end = _strip_tz(start), _strip_tz(end)
        for event in self.events:
            comp = event.icalendar_component
            ev_start = _strip_tz(comp.get("dtstart").dt)
            dtend_field = comp.get("dtend")
            ev_end = _strip_tz(dtend_field.dt) if dtend_field else ev_start
            if _overlaps(q_start, q_end, ev_start, ev_end):
                results.append(event)
        return results
