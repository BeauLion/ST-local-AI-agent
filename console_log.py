"""
console_log.py — tiny shared buffer so per-request console prints (from
main.py, memory.py, or any other module that imports this) can be captured
alongside the exact terminal output they already produce, then attached to
the matching prompt-log entry for prompt_log_viewer.html to pair up.

Deliberately has zero imports beyond nothing at all - any module can import
this without risking a circular import with main.py.
"""

_lines: list[str] = []


def alog(msg: str) -> None:
    """Print exactly as a normal print() would (so start.py's terminal
    capture and behavior are unchanged), AND remember the line for the
    next flush()."""
    print(msg)
    _lines.append(msg)


def flush() -> list[str]:
    """Returns everything buffered since the last flush() and clears the
    buffer. main.py calls this once per agent-loop iteration (and once
    more for the pre-loop context-building prints, which naturally ride
    along with iteration 0's flush since nothing drains the buffer before
    that point)."""
    global _lines
    lines, _lines = _lines, []
    return lines
