"""
Tests for prompt_log_engine.py's _read_log_entries (malformed-line
resilience) and _annotations_path (path-traversal safety).
"""
import prompt_log_engine as pe


# ---------------------------------------------------------------------------
# _read_log_entries
# ---------------------------------------------------------------------------

def test_reads_valid_jsonl_entries(tmp_path):
    log_file = tmp_path / "session.log"
    log_file.write_text('{"type": "prompt", "iteration": 1}\n{"type": "console", "iteration": 1}\n', encoding="utf-8")

    entries = pe._read_log_entries(log_file, "session.log")

    assert len(entries) == 2
    assert entries[0]["type"] == "prompt"
    assert entries[1]["type"] == "console"


def test_skips_malformed_lines_without_crashing(tmp_path, capsys):
    log_file = tmp_path / "session.log"
    log_file.write_text(
        '{"type": "prompt", "iteration": 1}\n'
        'not valid json at all\n'
        '{"type": "console", "iteration": 1}\n',
        encoding="utf-8",
    )

    entries = pe._read_log_entries(log_file, "session.log")

    assert len(entries) == 2  # the malformed line was skipped, not crashed on
    captured = capsys.readouterr()
    assert "Skipping malformed log line" in captured.out


def test_skips_blank_lines_silently(tmp_path):
    log_file = tmp_path / "session.log"
    log_file.write_text('{"type": "prompt", "iteration": 1}\n\n\n{"type": "console", "iteration": 1}\n', encoding="utf-8")

    entries = pe._read_log_entries(log_file, "session.log")
    assert len(entries) == 2


def test_empty_file_returns_empty_list(tmp_path):
    log_file = tmp_path / "session.log"
    log_file.write_text("", encoding="utf-8")
    assert pe._read_log_entries(log_file, "session.log") == []


# ---------------------------------------------------------------------------
# _annotations_path - traversal safety
# ---------------------------------------------------------------------------

def test_annotations_path_for_a_plain_filename(tmp_log_dir):
    path = pe._annotations_path("session_20260101.log")
    assert path == tmp_log_dir / "session_20260101.log.annotations.json"


def test_annotations_path_strips_directory_traversal_components(tmp_log_dir):
    path = pe._annotations_path("../../etc/passwd")
    # Path(...).name strips every path component - only "passwd" survives,
    # confined to _PROMPT_LOG_DIR regardless of how many "../" were given.
    assert path.parent == tmp_log_dir
    assert path.name == "passwd.annotations.json"
    assert ".." not in str(path)


def test_annotations_path_strips_absolute_path_components(tmp_log_dir):
    path = pe._annotations_path("/etc/passwd")
    assert path.parent == tmp_log_dir
    assert path.name == "passwd.annotations.json"
