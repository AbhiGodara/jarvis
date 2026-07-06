"""
core/embedder.py — process-wide shared sentence-embedding model (Mk-III Phase 3).

One SentenceTransformer("all-MiniLM-L6-v2") instance serves both consumers:
the semantic router (commands/registry.semantic_best) and the RAG vector
store — the ~80 MB model loads at most once per process.

Two access patterns:
  get_embedder()      BLOCKING — load if needed, return the model (RAG's
                      startup path, tests).
  encode_if_ready()   NON-BLOCKING — the router's path. Kicks a background
                      load on the first call and returns None until the model
                      is up, so a voice turn is never stalled by model
                      loading; routing simply falls through to the Mk-II
                      tiers for those first few seconds.

main.py calls preload_async() at boot when semantic routing is enabled, so
the model is normally ready before the first real query.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_load_failed = False
_load_started = False
_load_lock = threading.Lock()


def get_embedder():
    """Load (once) and return the shared model — blocking. None if unavailable."""
    global _model, _load_failed, _load_started
    with _load_lock:
        _load_started = True
        if _model is not None or _load_failed:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model '{MODEL_NAME}' (shared, once per process)...")
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model ready.")
        except Exception as e:
            _load_failed = True
            logger.warning(
                f"Embedding model unavailable ({e}). "
                f"Semantic routing and RAG need: pip install sentence-transformers"
            )
        return _model


def preload_async() -> None:
    """Start loading in a background thread (call once at boot)."""
    global _load_started
    _load_started = True
    threading.Thread(target=get_embedder, name="embedder-preload", daemon=True).start()


def is_ready() -> bool:
    return _model is not None


def encode_if_ready(texts: list[str]) -> np.ndarray | None:
    """Encode texts to L2-normalized vectors, or None if the model isn't up.

    Never blocks: the first call kicks a background load and returns None;
    callers fall back to non-semantic behavior until the model is ready.
    """
    if _model is None:
        if not _load_started and not _load_failed:
            preload_async()
        return None
    vectors = _model.encode(
        list(texts), show_progress_bar=False, normalize_embeddings=True
    )
    return np.asarray(vectors, dtype=np.float32)
