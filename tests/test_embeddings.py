"""Embedding shape, normalization, silence handling, and model-cache behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from sts.nlp import embeddings


@dataclass
class _Model:
    calls: list[list[str]] = field(default_factory=list)
    output: NDArray[np.float32] | None = None
    dimension: int | None = 3

    def get_sentence_embedding_dimension(self) -> int | None:
        """Report a tiny deterministic vector dimension."""
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> NDArray[np.float32]:
        """Record batches and return deliberately unnormalized vectors."""
        assert kwargs == {
            "batch_size": 32,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        self.calls.append(texts)
        if self.output is not None:
            return self.output
        return np.asarray([[3.0, 4.0, 0.0] for text in texts], dtype=np.float32)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> _Model:
    """Use a deterministic local encoder for every ordinary embedding check."""
    result = _Model()
    monkeypatch.setattr(embeddings, "_create_model", lambda name: result)
    return result


def test_normalizes_vectors_and_preserves_blank_row_positions(model: _Model) -> None:
    """Silent scenes get zeros while dialogue keeps its input order and unit length."""
    result = embeddings.embed(["first", "  ", "second", ""])
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.6, 0.8, 0], [0, 0, 0], [0.6, 0.8, 0], [0, 0, 0]])
    assert model.calls == [["first", "second"]]


def test_empty_input_does_not_load_a_model() -> None:
    """An empty sequence is valid even without the configured model installed."""
    assert embeddings.embed([]).shape == (0, 0)
    assert embeddings._MODEL is None


def test_all_blank_inputs_skip_encoding(model: _Model) -> None:
    """Silence has the configured model's dimension but no semantic content."""
    np.testing.assert_array_equal(embeddings.embed(["", "\n"]), np.zeros((2, 3)))
    assert model.calls == []


def test_cache_reuses_model_and_replaces_on_configuration_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated calls reuse weights; an explicitly different model replaces the singleton."""
    constructed: list[str] = []

    def create(name: str) -> _Model:
        constructed.append(name)
        return _Model()

    monkeypatch.setattr(embeddings, "_create_model", create)
    monkeypatch.setenv("SBERT_MODEL", "configured-model")
    embeddings.embed(["one"])
    original = embeddings._MODEL
    embeddings.embed(["two"])
    assert embeddings._MODEL is original
    embeddings.embed(["three"], model_name="saved-job-model")
    assert embeddings._MODEL is not original
    assert constructed == ["configured-model", "saved-job-model"]


@pytest.mark.parametrize("output", [[[0, 0, 0]], [[float("nan"), 0, 1]], [[1, 2]]])
def test_invalid_encoder_output_is_rejected(model: _Model, output: list[list[float]]) -> None:
    """Malformed, non-finite, or zero dialogue embeddings cannot enter similarity scoring."""
    model.output = np.asarray(output, dtype=np.float32)
    with pytest.raises(ValueError, match="Embedding model returned"):
        embeddings.embed(["dialogue"])


@pytest.mark.parametrize("dimension", [0, None])
def test_invalid_dimension_is_rejected(model: _Model, dimension: int | None) -> None:
    """A model must declare a usable dimension before allocating a result."""
    model.dimension = dimension
    with pytest.raises(ValueError, match="positive output dimension"):
        embeddings.embed(["dialogue"])


def test_invalid_inputs_fail_before_model_loading() -> None:
    """Wrong text types and blank model names produce actionable input errors."""
    with pytest.raises(TypeError, match="must be strings"):
        embeddings.embed([123])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must not be blank"):
        embeddings.embed(["text"], model_name=" ")
