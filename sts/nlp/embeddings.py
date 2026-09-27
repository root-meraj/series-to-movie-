"""A lazy sentence-transformer singleton returning normalized float32 embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from sts.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_MODEL: tuple[str, SentenceTransformer] | None = None


def _create_model(name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def _get_model(name: str) -> SentenceTransformer:
    global _MODEL
    if _MODEL is None or _MODEL[0] != name:
        logging.getLogger(__name__).info("nlp unit-progress: loading embedding model %s", name)
        _MODEL = None
        _MODEL = (name, _create_model(name))
    return _MODEL[1]


def embed(texts: list[str], *, model_name: str | None = None) -> NDArray[np.float32]:
    """Embed text in order, L2-normalizing nonblank rows and zeroing blank rows.

    An empty input returns shape (0, 0) without loading a model. Otherwise the
    configured model defines the column count. Zero rows for silent scenes avoid
    treating two absent transcripts as semantically identical dialogue.
    """
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("Embedding inputs must be strings")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    name = Settings().sbert_model if model_name is None else model_name
    if not name.strip():
        raise ValueError("Embedding model name must not be blank")
    model = _get_model(name)
    dimension = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    if dimension is None or dimension <= 0:
        raise ValueError("Embedding model must report a positive output dimension")
    result = np.zeros((len(texts), dimension), dtype=np.float32)
    indices = [index for index, text in enumerate(texts) if text.strip()]
    if not indices:
        return result
    vectors = np.asarray(
        model.encode(
            [texts[index] for index in indices],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vectors.shape != (len(indices), dimension) or not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned invalid shape or non-finite values")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector for nonblank text")
    result[indices] = (vectors / norms).astype(np.float32)
    return result
