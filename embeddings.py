"""
Embedding Pipeline Module.
Wraps HuggingFace sentence-transformers for generating text embeddings.
Uses a singleton pattern to avoid loading the model multiple times.
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ─── Module-Level Singleton ──────────────────────────────────────────────────
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: '{EMBEDDING_MODEL}'...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Embedding model loaded. Dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).

    Raises:
        ValueError: If the input list is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = _get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def get_single_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text string.

    Args:
        text: The text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    return get_embeddings([text])[0]


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding model."""
    return _get_model().get_sentence_embedding_dimension()
