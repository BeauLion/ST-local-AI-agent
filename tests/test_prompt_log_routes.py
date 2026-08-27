"""
Tests for prompt_log_engine.py's HTTP routes, via a TestClient wired up
with just this module's router (see the `client` fixture).

Lower-value coverage than the pairing/writing/safety tests, per the
earlier scoping discussion: most of these routes are thin wrappers over
already-tested functions (_read_log_entries, _annotations_path,
_build_groups) or over FastAPI's own request/response handling. Worth
locking in the current wiring and status codes, not expecting to find
much here.
"""
import json

import prompt_log_engine as pe


# ---------------------------------------------------------------------------
# GET /prompt-logs
# ---------------------------------------------------------------------------

def test_list_prompt_logs_returns_empty_list_when_no_logs_exist(client, tmp_log_dir):
    response = client.get("/prompt-logs")
    assert response.status_code == 200
    assert response.json() == []


def test_list_prompt_logs_returns_files_most_recent_first(client, tmp_log_dir):
    import time
    older = tmp_log_dir / "session_older.log"
    older.write_text("{}\n", encoding="utf-8")
    time.sleep(0.01)
    newer = tmp_log_dir / "session_newer.log"
    newer.write_text("{}\n", encoding="utf-8")

    response = client.get("/prompt-logs")
    filenames = [f["filename"] for f in response.json()]
    assert filenames == ["session_newer.log", "session_older.log"]


def test_list_prompt_logs_ignores_non_session_files(client, tmp_log_dir):
    (tmp_log_dir / "session_a.log").write_text("{}\n", encoding="utf-8")
    (tmp_log_dir / "session_a.log.annotations.json").write_text("{}", encoding="utf-8")

    response = client.get("/prompt-logs")
    filenames = [f["filename"] for f in response.json()]
    assert filenames == ["session_a.log"]


# ---------------------------------------------------------------------------
# GET /prompt-logs/{filename}
# ---------------------------------------------------------------------------

def test_get_prompt_log_returns_parsed_entries(client, tmp_log_dir):
    log_file = tmp_log_dir / "session_a.log"
    log_file.write_text('{"type": "prompt", "iteration": 1}\n', encoding="utf-8")

    response = client.get("/prompt-logs/session_a.log")
    assert response.status_code == 200
    assert response.json() == [{"type": "prompt", "iteration": 1}]


def test_get_prompt_log_404s_for_missing_file(client, tmp_log_dir):
    response = client.get("/prompt-logs/does_not_exist.log")
    assert response.status_code == 404


def test_get_prompt_log_traversal_attempt_is_confined_and_404s(client, tmp_log_dir):
    """A path-traversal filename should resolve to (the nonexistent)
    _PROMPT_LOG_DIR/passwd, not actually escape the directory - so this
    404s rather than exposing anything outside PROMPT_LOG_DIR."""
    response = client.get("/prompt-logs/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET/PUT /prompt-logs/{filename}/annotations
# ---------------------------------------------------------------------------

def test_get_annotations_returns_empty_shape_when_never_annotated(client, tmp_log_dir):
    response = client.get("/prompt-logs/session_a.log/annotations")
    assert response.status_code == 200
    assert response.json() == {"iteration_notes": {}, "chunk_notes": {}}


def test_put_then_get_annotations_round_trips(client, tmp_log_dir):
    body = {"iteration_notes": {"0": "a note"}, "chunk_notes": {}}
    put_response = client.put("/prompt-logs/session_a.log/annotations", json=body)
    assert put_response.status_code == 200
    assert put_response.json() == {"saved": True}

    get_response = client.get("/prompt-logs/session_a.log/annotations")
    assert get_response.json() == body


def test_put_annotations_rejects_a_malformed_body(client, tmp_log_dir):
    response = client.put("/prompt-logs/session_a.log/annotations", json={"wrong_key": {}})
    assert response.status_code == 400


def test_get_annotations_recovers_from_a_corrupt_file_on_disk(client, tmp_log_dir, capsys):
    ann_path = pe._annotations_path("session_a.log")
    ann_path.write_text("{not valid json", encoding="utf-8")

    response = client.get("/prompt-logs/session_a.log/annotations")
    assert response.status_code == 200
    assert response.json() == {"iteration_notes": {}, "chunk_notes": {}}


# ---------------------------------------------------------------------------
# GET /notes/search
# ---------------------------------------------------------------------------

def test_search_notes_empty_query_returns_no_results(client, tmp_log_dir):
    response = client.get("/notes/search", params={"q": "  "})
    assert response.json() == {"results": []}


def test_search_notes_finds_a_matching_iteration_note(client, tmp_log_dir):
    log_file = tmp_log_dir / "session_a.log"
    log_file.write_text(
        json.dumps({"type": "prompt", "iteration": 1, "model": "m", "chunks": [], "timestamp": "2026-08-27T10:00:00"}) + "\n",
        encoding="utf-8",
    )
    pe._annotations_path("session_a.log").write_text(
        json.dumps({"iteration_notes": {"0": "remember to check this"}, "chunk_notes": {}}), encoding="utf-8",
    )

    response = client.get("/notes/search", params={"q": "remember"})
    results = response.json()["results"]

    assert len(results) == 1
    assert results[0]["filename"] == "session_a.log"
    assert results[0]["kind"] == "iteration"
    assert results[0]["iteration"] == 1


def test_search_notes_is_case_insensitive(client, tmp_log_dir):
    log_file = tmp_log_dir / "session_a.log"
    log_file.write_text(
        json.dumps({"type": "prompt", "iteration": 1, "chunks": [], "timestamp": "2026-08-27T10:00:00"}) + "\n",
        encoding="utf-8",
    )
    pe._annotations_path("session_a.log").write_text(
        json.dumps({"iteration_notes": {"0": "Important Note"}, "chunk_notes": {}}), encoding="utf-8",
    )

    response = client.get("/notes/search", params={"q": "important"})
    assert len(response.json()["results"]) == 1


def test_search_notes_skips_annotations_whose_log_file_no_longer_exists(client, tmp_log_dir):
    """An annotations file surviving a deleted/renamed log - per the
    module's own comment - must be skipped, not raise."""
    pe._annotations_path("gone.log").write_text(
        json.dumps({"iteration_notes": {"0": "orphaned note"}, "chunk_notes": {}}), encoding="utf-8",
    )
    response = client.get("/notes/search", params={"q": "orphaned"})
    assert response.json() == {"results": []}


def test_search_notes_with_no_matching_text_returns_empty(client, tmp_log_dir):
    log_file = tmp_log_dir / "session_a.log"
    log_file.write_text(
        json.dumps({"type": "prompt", "iteration": 1, "chunks": [], "timestamp": "2026-08-27T10:00:00"}) + "\n",
        encoding="utf-8",
    )
    pe._annotations_path("session_a.log").write_text(
        json.dumps({"iteration_notes": {"0": "nothing relevant here"}, "chunk_notes": {}}), encoding="utf-8",
    )

    response = client.get("/notes/search", params={"q": "xyzzy"})
    assert response.json() == {"results": []}


# ---------------------------------------------------------------------------
# Static viewer pages
# ---------------------------------------------------------------------------

def test_prompt_log_viewer_page_serves_html(client):
    response = client.get("/prompt-log-viewer")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_prompt_log_viewer_css_serves_with_correct_content_type(client):
    response = client.get("/prompt_log_viewer.css")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/css; charset=utf-8"


def test_prompt_log_viewer_js_serves_with_correct_content_type(client):
    response = client.get("/prompt_log_viewer.js")
    assert response.status_code == 200
    # Starlette only auto-appends "; charset=utf-8" for "text/*" media
    # types (as CSS gets above) - "application/javascript" doesn't get
    # one, so this is a plain equality check with no charset suffix.
    assert response.headers["content-type"] == "application/javascript"
