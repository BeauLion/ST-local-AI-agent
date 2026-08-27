"""
Fake stand-in for memory.py's embed() - the one seam duration_manager.py
touches in that module. Installed into sys.modules BEFORE duration_manager
(or memory itself) is ever imported anywhere in the test session, so the
real memory.py - and the real SentenceTransformer model it loads at
IMPORT time, not lazily (see memory.py's module-level `_model = ...`
line) - is never touched by this test suite at all. Without this, just
importing duration_manager for testing would load a real ML model.

Why fake embed() itself, not just the network/disk path around it (unlike
calendar_manager's CalDAV fake, which faked the network layer but kept
icalendar - a structural, stable library - real): embedding output is
the model's own judgment call, not deterministic structural behavior.
A model/weights change could shift real cosine-similarity scores without
any bug in duration_manager.py's code, which would make tests built on
the real model both flaky over model versions and slow to run. This
suite instead hands resolve_category() hand-picked, controlled vectors
precise enough to deliberately target its threshold-comparison logic -
it is not this test suite's job to verify the embedding model's
semantic judgment, only duration_manager.py's own logic around it.
"""
import sys
from types import ModuleType


class FakeEmbedder:
    """Maps exact input strings to hand-picked vectors via .set(text, vector).
    Any text NOT explicitly configured falls back to a fixed default
    vector, so tests that don't care about embedding output (most - they
    exercise the alias/exact-match paths in resolve_category, which never
    call embed() at all) don't need to configure anything up front."""

    def __init__(self):
        self._vectors: dict[str, list] = {}
        self._default: list = [1.0, 0.0, 0.0]

    def set(self, text: str, vector: list) -> None:
        self._vectors[text] = vector

    def set_default(self, vector: list) -> None:
        self._default = vector

    def reset(self) -> None:
        self._vectors.clear()
        self._default = [1.0, 0.0, 0.0]

    def __call__(self, text: str) -> list:
        return self._vectors.get(text, self._default)


def install_fake_memory_module() -> ModuleType:
    """Call once, before duration_manager (or memory) is imported anywhere
    in the process. Idempotent - if a memory module (real or fake) is
    already present in sys.modules, this does nothing and just returns
    it, so calling it more than once (e.g. from multiple conftest.py
    files) is always safe."""
    if "memory" in sys.modules:
        return sys.modules["memory"]
    fake = ModuleType("memory")
    embedder = FakeEmbedder()
    fake.embed = embedder          # duration_manager.py calls memory.embed(text)
    fake._embedder = embedder      # tests reach in via duration_manager.memory._embedder
    sys.modules["memory"] = fake
    return fake
