"""
Phase 3: Local memory + document RAG.

Two separate stores, both using the same underlying tech:
  - "Personal memory": short facts the agent explicitly chose to remember
    about you (saved via the save_memory tool). Two special categories on
    top of plain freeform facts:
      - slots (MEMORY_IDENTITY_SLOTS): singleton identity facts (name,
        occupation, location, pronouns) - always injected into context,
        upserted rather than appended.
      - freeform pins: any other memory the model has explicitly pinned via
        pin_memory - also always injected, capped at MEMORY_MAX_FREEFORM_PINS.
    See brainstorm-memory-structure-and-dedupe.md for the design writeup.
  - "Document index": chunks of your .txt files in agent_files, searchable
    by meaning rather than exact keyword match

Embeddings run on CPU via sentence-transformers - deliberately kept off the
GPU so it never competes with llama.cpp for VRAM. This model is tiny, CPU is
plenty fast for personal-scale use.

Storage is just JSON files + numpy cosine similarity - no external vector
database needed at this scale (hundreds to a few thousand chunks/memories).
"""

import datetime
import json
import uuid
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    DOCUMENT_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    MEMORY_DATA_DIR,
    MEMORY_DEDUPE_SIMILARITY_THRESHOLD,
    MEMORY_IDENTITY_SLOTS,
    MEMORY_MAX_FREEFORM_PINS,
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

def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _load_memories() -> list:
    if not MEMORY_FILE.exists():
        return []
    memories = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))

    # Migration: older memories.json entries only have some subset of
    # id/created_at/updated_at/slot/pinned. Backfill in place so the file
    # only needs to be upgraded once, on first load after a given change
    # ships. Safe to run indefinitely - a no-op once every entry is current.
    migrated = False
    for m in memories:
        if "id" not in m:
            m["id"] = uuid.uuid4().hex[:8]
            migrated = True
        if "created_at" not in m:
            m["created_at"] = _now_iso()
            migrated = True
        if "updated_at" not in m:
            m["updated_at"] = m["created_at"]
            migrated = True
        if "slot" not in m:
            m["slot"] = None
            migrated = True
        if "pinned" not in m:
            m["pinned"] = m["slot"] is not None
            migrated = True
    if migrated:
        _save_memories(memories)

    return memories


def _save_memories(memories: list):
    MEMORY_FILE.write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_freeform_pins(memories: list) -> int:
    return sum(1 for m in memories if m.get("pinned") and not m.get("slot"))


def save_memory(text: str, slot: str = None) -> dict:
    """Save a memory. Two modes:

    - slot given (must be a valid MEMORY_IDENTITY_SLOTS value - callers are
      expected to validate this before calling in, same as other args):
      upsert. If an entry already occupies that slot, its text/embedding/
      updated_at are overwritten in place (no duplicate created); otherwise
      a new always-pinned entry is created for that slot. No dedupe check -
      upserting by construction can't create a duplicate.
    - slot omitted: plain freeform memory, appended as a new entry (never
      overwrites anything). Before appending, runs a non-blocking dedupe
      check against every existing memory (freeform AND slots - a freeform
      fact silently contradicting a slot is the case most worth catching).
      The memory is saved either way; if a close match is found its info is
      returned so the caller can nudge the model toward update_memory.

    Returns a dict: {"id", "action" ("created"/"updated"), "slot", "similar"}
    where "similar" is None or {"id", "text", "score"} of the closest
    existing match (freeform mode only).
    """
    memories = _load_memories()
    now = _now_iso()

    if slot:
        for m in memories:
            if m.get("slot") == slot:
                m["text"] = text
                m["embedding"] = embed(text)
                m["updated_at"] = now
                m["pinned"] = True
                _save_memories(memories)
                return {"id": m["id"], "action": "updated", "slot": slot, "similar": None}
        new_id = uuid.uuid4().hex[:8]
        memories.append({
            "id": new_id, "text": text, "embedding": embed(text),
            "created_at": now, "updated_at": now, "slot": slot, "pinned": True,
        })
        _save_memories(memories)
        return {"id": new_id, "action": "created", "slot": slot, "similar": None}

    text_vec = embed(text)
    similar = None
    dupe_hits = _cosine_search(text_vec, memories, top_k=1, min_score=MEMORY_DEDUPE_SIMILARITY_THRESHOLD)
    if dupe_hits:
        m, score = dupe_hits[0]
        similar = {"id": m["id"], "text": m["text"], "score": score}

    new_id = uuid.uuid4().hex[:8]
    memories.append({
        "id": new_id, "text": text, "embedding": text_vec,
        "created_at": now, "updated_at": now, "slot": None, "pinned": False,
    })
    _save_memories(memories)
    return {"id": new_id, "action": "created", "slot": None, "similar": similar}


def update_memory(memory_id: str, new_text: str) -> str:
    """Replace an existing memory's text in place, by id. Returns a status
    string - never raises, since this is called from a model tool wrapper.
    Leaves slot/pinned untouched - correcting a fact's value shouldn't
    unpin it or remove it from its slot."""
    memories = _load_memories()
    for m in memories:
        if m["id"] == memory_id:
            old_text = m["text"]
            m["text"] = new_text
            m["embedding"] = embed(new_text)
            m["updated_at"] = _now_iso()
            _save_memories(memories)
            return f"Updated memory {memory_id} (was: {old_text!r}) -> {new_text!r}"
    return f"No memory found with id {memory_id!r}. Use list_memories or search_memories to find the correct id."


def delete_memory(memory_id: str) -> str:
    """Remove a memory by id. Returns a status string - never raises. Works
    the same regardless of slot/pinned status - deleting a slotted memory
    just empties that slot; it can be re-saved later with the same slot."""
    memories = _load_memories()
    for m in memories:
        if m["id"] == memory_id:
            memories.remove(m)
            _save_memories(memories)
            return f"Deleted memory {memory_id}: {m['text']!r}"
    return f"No memory found with id {memory_id!r}. Use list_memories or search_memories to find the correct id."


def pin_memory(memory_id: str) -> str:
    """Pin a freeform memory so it's always injected into context, without
    needing to match a search query. Returns a status string - never raises.
    Slotted memories are already always-pinned by construction; pinning one
    again is a no-op with an explanatory message rather than an error."""
    memories = _load_memories()
    for m in memories:
        if m["id"] == memory_id:
            if m.get("slot"):
                return f"Memory {memory_id} is part of the '{m['slot']}' slot and is already always shown - no change needed."
            if m.get("pinned"):
                return f"Memory {memory_id} is already pinned."
            if _count_freeform_pins(memories) >= MEMORY_MAX_FREEFORM_PINS:
                return (
                    f"Cannot pin memory {memory_id}: freeform pin limit "
                    f"({MEMORY_MAX_FREEFORM_PINS}) reached. Unpin another memory first."
                )
            m["pinned"] = True
            m["updated_at"] = _now_iso()
            _save_memories(memories)
            return f"Pinned memory {memory_id}: {m['text']!r}"
    return f"No memory found with id {memory_id!r}. Use list_memories or search_memories to find the correct id."


def unpin_memory(memory_id: str) -> str:
    """Unpin a freeform memory. Returns a status string - never raises.
    Slotted memories can't be unpinned (they're always-shown by design);
    use delete_memory to remove one instead."""
    memories = _load_memories()
    for m in memories:
        if m["id"] == memory_id:
            if m.get("slot"):
                return f"Memory {memory_id} is part of the '{m['slot']}' slot and can't be unpinned - slots are always shown. Use delete_memory to remove it instead."
            if not m.get("pinned"):
                return f"Memory {memory_id} isn't pinned."
            m["pinned"] = False
            m["updated_at"] = _now_iso()
            _save_memories(memories)
            return f"Unpinned memory {memory_id}: {m['text']!r}"
    return f"No memory found with id {memory_id!r}. Use list_memories or search_memories to find the correct id."


def list_memories() -> list:
    """Return every saved memory as {id, text, slot, pinned}. Slots first
    (in MEMORY_IDENTITY_SLOTS order), then everything else most-recently-
    updated first."""
    memories = _load_memories()
    slot_order = {s: i for i, s in enumerate(MEMORY_IDENTITY_SLOTS)}

    # Slots first, in MEMORY_IDENTITY_SLOTS order; everything else after,
    # most-recently-updated first. Two different sort directions, so this
    # is done as two separate sorts rather than one combined key.
    slotted = sorted(
        (m for m in memories if m.get("slot") in slot_order),
        key=lambda m: slot_order[m["slot"]],
    )
    rest = sorted(
        (m for m in memories if m.get("slot") not in slot_order),
        key=lambda m: m["updated_at"], reverse=True,
    )
    ordered = slotted + rest
    return [{"id": m["id"], "text": m["text"], "slot": m.get("slot"), "pinned": m.get("pinned", False)} for m in ordered]


def list_memories_full() -> list:
    """Like list_memories(), but includes created_at/updated_at and skips
    the token-saving omission that's fine for model context but not for a
    browsing UI. Same slots-first-then-recent ordering. Excludes the
    embedding vector (never needed client-side)."""
    memories = _load_memories()
    slot_order = {s: i for i, s in enumerate(MEMORY_IDENTITY_SLOTS)}
    slotted = sorted(
        (m for m in memories if m.get("slot") in slot_order),
        key=lambda m: slot_order[m["slot"]],
    )
    rest = sorted(
        (m for m in memories if m.get("slot") not in slot_order),
        key=lambda m: m["updated_at"], reverse=True,
    )
    ordered = slotted + rest
    return [
        {
            "id": m["id"], "text": m["text"], "slot": m.get("slot"),
            "pinned": m.get("pinned", False),
            "created_at": m.get("created_at"), "updated_at": m.get("updated_at"),
        }
        for m in ordered
    ]


def get_pinned_memories() -> list:
    """Return every always-injected memory (slots + freeform pins), in the
    same slots-first-then-recent order as list_memories. Used by the
    auto-recall block so these show up every turn regardless of query."""
    return [m for m in list_memories() if m["pinned"]]


def search_memories(query: str, top_k: int = 3) -> list:
    """Return relevant memories as [{"id", "text", "slot", "pinned"}, ...],
    ranked by similarity to query only (no recency/pin weighting - the
    auto-recall caller is responsible for surfacing pinned memories
    separately and filtering them out of this result to avoid duplicates)."""
    memories = _load_memories()
    results = _cosine_search(embed(query), memories, top_k=top_k, min_score=MEMORY_SIMILARITY_THRESHOLD)
    print(f"[MEMORY] Query: {query!r}")
    for m, score in results:
        print(f"[MEMORY]   {score:.3f}  [{m['id']}] {m['text']}")
    if not results:
        all_scored = _cosine_search(embed(query), memories, top_k=3, min_score=-1)
        for m, score in all_scored:
            print(f"[MEMORY]   (below threshold) {score:.3f}  [{m['id']}] {m['text']}")
    return [{"id": m["id"], "text": m["text"], "slot": m.get("slot"), "pinned": m.get("pinned", False)} for m, score in results]


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
