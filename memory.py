"""
Phase 3: Local memory + document RAG.

Two separate stores, both using the same underlying tech:
  - "Personal memory": short facts the agent explicitly chose to remember
    about you (saved via the save_memory tool)
  - "Document index": chunks of your .txt files in agent_files, searchable
    by meaning rather than exact keyword match

Embeddings run on CPU via sentence-transformers - deliberately kept off the
GPU so it never competes with llama.cpp for VRAM. This model is tiny, CPU is
plenty fast for personal-scale use.

Storage is just JSON files + numpy cosine similarity - no external vector
database needed at this scale (hundreds to a few thousand chunks/memories).
"""

import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    DOCUMENT_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    MEMORY_DATA_DIR,
    MEMORY_SIMILARITY_THRESHOLD,
)

DOC_EXTENSIONS = (".txt", ".pdf")

_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

DATA_DIR = Path(MEMORY_DATA_DIR)
DATA_DIR.mkdir(exist_ok=True)

MEMORY_FILE = DATA_DIR / "memories.json"
DOC_INDEX_FILE = DATA_DIR / "doc_index.json"


def embed(text: str) -> list:
    # normalize_embeddings=True means dot product == cosine similarity,
    # which is what _cosine_search relies on below.
    return _model.encode(text, normalize_embeddings=True).tolist()


def _cosine_search(query_vec, items, top_k=3, min_score=0.05):
    if not items:
        return []
    vecs = np.array([it["embedding"] for it in items])
    q = np.array(query_vec)
    scores = vecs @ q
    top_idx = np.argsort(-scores)[:top_k]
    return [(items[i], float(scores[i])) for i in top_idx if scores[i] >= min_score]


# --------------------------- Personal memory ---------------------------

def _load_memories() -> list:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return []


def _save_memories(memories: list):
    MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")


def save_memory(text: str):
    memories = _load_memories()
    memories.append({"text": text, "embedding": embed(text)})
    _save_memories(memories)


def search_memories(query: str, top_k: int = 3) -> list:
    memories = _load_memories()
    results = _cosine_search(embed(query), memories, top_k=top_k, min_score=MEMORY_SIMILARITY_THRESHOLD)
    print(f"[MEMORY] Query: {query!r}")
    for m, score in results:
        print(f"[MEMORY]   {score:.3f}  {m['text']}")
    if not results:
        all_scored = _cosine_search(embed(query), memories, top_k=3, min_score=-1)
        for m, score in all_scored:
            print(f"[MEMORY]   (below threshold) {score:.3f}  {m['text']}")
    return [m["text"] for m, score in results]


# --------------------------- Document RAG ---------------------------

def extract_text(file: Path, max_chars: int = None) -> str:
    """Read a document's text. For PDFs, stops parsing pages once max_chars
    is reached instead of extracting the whole file just to truncate it."""
    if file.suffix.lower() == ".pdf":
        reader = PdfReader(str(file))
        parts = []
        total = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            parts.append(page_text)
            total += len(page_text)
            if max_chars is not None and total >= max_chars:
                break
        return "\n".join(parts)
    return file.read_text(encoding="utf-8", errors="replace")


def _load_doc_index() -> dict:
    if DOC_INDEX_FILE.exists():
        return json.loads(DOC_INDEX_FILE.read_text(encoding="utf-8"))
    return {"files": {}, "chunks": []}


def _save_doc_index(index: dict):
    DOC_INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _reindex_if_changed(folder: Path) -> dict:
    """Re-embed only files that are new or have changed since last time."""
    index = _load_doc_index()
    changed = False

    doc_files = [f for f in folder.iterdir() if f.suffix.lower() in DOC_EXTENSIONS]
    current_files = {f.name for f in doc_files}

    # Drop chunks for files that were deleted.
    removed = set(index["files"]) - current_files
    if removed:
        index["chunks"] = [c for c in index["chunks"] if c["source"] not in removed]
        for name in removed:
            del index["files"][name]
        changed = True

    for file in doc_files:
        mtime = file.stat().st_mtime
        if index["files"].get(file.name) == mtime:
            continue  # unchanged since last index

        index["chunks"] = [c for c in index["chunks"] if c["source"] != file.name]
        text = extract_text(file)
        for chunk in _chunk_text(text):
            index["chunks"].append({"source": file.name, "text": chunk, "embedding": embed(chunk)})
        index["files"][file.name] = mtime
        changed = True

    if changed:
        _save_doc_index(index)
    return index


def search_documents(query: str, folder: Path, top_k: int = 4) -> list:
    index = _reindex_if_changed(folder)
    results = _cosine_search(embed(query), index["chunks"], top_k=top_k, min_score=DOCUMENT_SIMILARITY_THRESHOLD)
    return [(r["source"], r["text"]) for r, score in results]
