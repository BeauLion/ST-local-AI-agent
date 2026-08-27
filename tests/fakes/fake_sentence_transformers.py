"""
Fake stand-in for sentence_transformers.SentenceTransformer - memory.py's
one real ML dependency, loaded at module IMPORT time as `_model =
SentenceTransformer(EMBEDDING_MODEL, device="cpu")`. Installed into
sys.modules BEFORE memory.py is ever imported, so the real (multi-hundred
MB, network-downloaded-on-first-use) model is never touched by this test
suite at all.

Same reasoning as duration_manager's fake_memory.py: embedding output is
the model's own judgment call, not deterministic structural behavior, so
this suite hands memory.py's cosine-similarity logic (_cosine_search,
dedupe checks, search thresholds) controlled, hand-picked vectors rather
than depending on a real model's actual semantic output.
"""
import sys
from types import ModuleType

import numpy as np


class FakeSentenceTransformer:
    """Controllable stand-in for the real SentenceTransformer. .set(text,
    vector) pins what .encode(text, ...) returns for a specific string;
    any other text falls back to a fixed default vector. .encode() must
    return something with a .tolist() method (memory.py's embed() calls
    that on the result) - a numpy array satisfies that directly."""

    def __init__(self, *args, **kwargs):
        self._vectors: dict[str, list] = {}
        self._default: list = [1.0, 0.0, 0.0]

    def set(self, text: str, vector: list) -> None:
        self._vectors[text] = vector

    def set_default(self, vector: list) -> None:
        self._default = vector

    def reset(self) -> None:
        self._vectors.clear()
        self._default = [1.0, 0.0, 0.0]

    def encode(self, text, normalize_embeddings: bool = True):
        return np.array(self._vectors.get(text, self._default))


def install_fake_sentence_transformers_module() -> ModuleType:
    """Call once, before memory.py is imported anywhere in the process.
    Idempotent - if sentence_transformers (real or fake) is already in
    sys.modules, this does nothing and returns it, so calling it more
    than once is always safe."""
    if "sentence_transformers" in sys.modules:
        return sys.modules["sentence_transformers"]
    fake = ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake
    return fake
