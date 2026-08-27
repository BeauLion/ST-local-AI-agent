"""
Tests for memory.py's document-indexing/search functions: extract_text,
_chunk_text, _reindex_if_changed, search_documents.

Uses real .txt files on disk under tmp_path (no fake needed - reading
plain text files is deterministic, structural behavior, same reasoning
as calendar_manager's tests keeping the real icalendar library). PDF
extraction (extract_text's other branch) isn't covered here - pypdf
itself is a real, separately-tested library, and constructing a real PDF
fixture file is out of scope for this pass; extract_text's .txt branch
is what _reindex_if_changed and search_documents actually exercise in
these tests.
"""
import memory


# ---------------------------------------------------------------------------
# extract_text - .txt branch
# ---------------------------------------------------------------------------

def test_extract_text_reads_a_plain_text_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Hello world", encoding="utf-8")
    assert memory.extract_text(f) == "Hello world"


def test_extract_text_replaces_undecodable_bytes_instead_of_raising(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_bytes(b"Valid text \xff\xfe more text")
    result = memory.extract_text(f)  # should not raise
    assert "Valid text" in result
    assert "more text" in result


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_short_text_is_a_single_chunk():
    result = memory._chunk_text("short text", chunk_size=800, overlap=100)
    assert result == ["short text"]


def test_chunk_text_splits_long_text_into_overlapping_chunks():
    text = "x" * 1000
    result = memory._chunk_text(text, chunk_size=800, overlap=100)
    assert len(result) == 2
    assert len(result[0]) == 800
    # Second chunk starts at (chunk_size - overlap) = 700
    assert result[1] == text[700:1000]


def test_chunk_text_empty_string_returns_no_chunks():
    assert memory._chunk_text("", chunk_size=800, overlap=100) == []


# ---------------------------------------------------------------------------
# _reindex_if_changed
# ---------------------------------------------------------------------------

def test_reindex_indexes_new_txt_files(tmp_path, tmp_doc_index_file, fake_memory_embedder):
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])
    (tmp_path / "doc.txt").write_text("Some document content", encoding="utf-8")

    index = memory._reindex_if_changed(tmp_path)

    assert "doc.txt" in index["files"]
    assert len(index["chunks"]) == 1
    assert index["chunks"][0]["source"] == "doc.txt"


def test_reindex_ignores_files_with_unrecognized_extensions(tmp_path, tmp_doc_index_file, fake_memory_embedder):
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])
    (tmp_path / "doc.txt").write_text("Indexed", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not a document")

    index = memory._reindex_if_changed(tmp_path)

    assert list(index["files"].keys()) == ["doc.txt"]


def test_reindex_skips_unchanged_files_on_second_call(tmp_path, tmp_doc_index_file, fake_memory_embedder, monkeypatch):
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])
    (tmp_path / "doc.txt").write_text("Content", encoding="utf-8")
    memory._reindex_if_changed(tmp_path)

    call_count = {"n": 0}
    real_encode = fake_memory_embedder.encode

    def counting_encode(*args, **kwargs):
        call_count["n"] += 1
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(fake_memory_embedder, "encode", counting_encode)
    memory._reindex_if_changed(tmp_path)  # nothing changed - should re-embed nothing

    assert call_count["n"] == 0


def test_reindex_re_embeds_a_file_that_changed(tmp_path, tmp_doc_index_file, fake_memory_embedder):
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])
    doc = tmp_path / "doc.txt"
    doc.write_text("Original content", encoding="utf-8")
    memory._reindex_if_changed(tmp_path)

    import time
    time.sleep(0.01)
    doc.write_text("Updated content", encoding="utf-8")
    index = memory._reindex_if_changed(tmp_path)

    assert len(index["chunks"]) == 1
    assert index["chunks"][0]["text"] == "Updated content"


def test_reindex_drops_chunks_for_deleted_files(tmp_path, tmp_doc_index_file, fake_memory_embedder):
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])
    doc = tmp_path / "doc.txt"
    doc.write_text("Content", encoding="utf-8")
    memory._reindex_if_changed(tmp_path)

    doc.unlink()
    index = memory._reindex_if_changed(tmp_path)

    assert index["files"] == {}
    assert index["chunks"] == []


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------

def test_search_documents_returns_matching_chunks_above_threshold(tmp_path, tmp_doc_index_file, fake_memory_embedder):
    fake_memory_embedder.set("Content about cats", [1.0, 0.0, 0.0])
    fake_memory_embedder.set("tell me about cats", [1.0, 0.0, 0.0])
    (tmp_path / "doc.txt").write_text("Content about cats", encoding="utf-8")

    results = memory.search_documents("tell me about cats", tmp_path)

    assert len(results) == 1
    assert results[0][0] == "doc.txt"
    assert results[0][1] == "Content about cats"


def test_search_documents_excludes_matches_below_threshold(tmp_path, tmp_doc_index_file, fake_memory_embedder):
    fake_memory_embedder.set("Content about cats", [1.0, 0.0, 0.0])
    fake_memory_embedder.set("completely unrelated query", [0.0, 1.0, 0.0])
    (tmp_path / "doc.txt").write_text("Content about cats", encoding="utf-8")

    results = memory.search_documents("completely unrelated query", tmp_path)
    assert results == []


def test_search_documents_respects_top_k(tmp_path, tmp_doc_index_file, fake_memory_embedder):
    fake_memory_embedder.set_default([1.0, 0.0, 0.0])
    for i in range(6):
        (tmp_path / f"doc{i}.txt").write_text(f"Content {i}", encoding="utf-8")

    results = memory.search_documents("query", tmp_path, top_k=2)
    assert len(results) == 2
