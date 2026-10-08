"""Embedding shape, normalization, silence handling, and model-cache behavior."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from sts.nlp import embeddings
from sts.nlp.embeddings import _create_model


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


def _install_constructor(
    monkeypatch: pytest.MonkeyPatch, constructor: Callable[..., _Model]
) -> None:
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = constructor
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embeddings, "_create_model", _create_model)


def test_cached_model_loads_without_remote_constructor_and_reuses_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete local cache skips remote metadata checks on first and later embeddings."""
    calls: list[tuple[str, dict[str, object]]] = []
    cached = _Model()

    def create(name: str, **kwargs: object) -> _Model:
        calls.append((name, kwargs))
        assert kwargs == {"local_files_only": True}
        return cached

    _install_constructor(monkeypatch, create)
    embeddings.embed(["first"], model_name="cached-model")
    identity = embeddings._MODEL
    embeddings.embed(["second"], model_name="cached-model")
    assert embeddings._MODEL is identity
    assert calls == [("cached-model", {"local_files_only": True})]


@pytest.mark.parametrize("wrapped", [False, True])
def test_cache_miss_downloads_once_then_reuses_weights(
    monkeypatch: pytest.MonkeyPatch, wrapped: bool
) -> None:
    """Hub cache misses and their Transformers wrappers permit exactly one download attempt."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[dict[str, object]] = []
    downloaded = _Model()

    def create(name: str, **kwargs: object) -> _Model:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            missing = LocalEntryNotFoundError("Required files are absent from the cache")
            if wrapped:
                raise OSError("The model could not be loaded from cached files") from missing
            raise missing
        return downloaded

    _install_constructor(monkeypatch, create)
    embeddings.embed(["first"], model_name="uncached-model")
    embeddings.embed(["second"], model_name="uncached-model")
    assert calls == [{"local_files_only": True}, {}]
    assert embeddings._MODEL is not None and embeddings._MODEL[1] is downloaded


def test_absent_cached_weight_file_allows_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transformers missing-weight errors without a nested Hub exception still repair the cache."""
    calls: list[dict[str, object]] = []

    def create(name: str, **kwargs: object) -> _Model:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError(f"{name} does not appear to have a file named model.safetensors")
        return _Model()

    _install_constructor(monkeypatch, create)
    embeddings.embed(["text"], model_name="missing-weights")
    assert calls == [{"local_files_only": True}, {}]


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("corrupt weights"),
        PermissionError("cache unreadable"),
        OSError("invalid tensor file"),
        ValueError("invalid configuration"),
    ],
)
def test_unrelated_constructor_failure_never_downloads_or_publishes_model(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Corruption, permissions, and programming/configuration failures remain explicit."""
    calls: list[dict[str, object]] = []

    def create(name: str, **kwargs: object) -> _Model:
        calls.append(kwargs)
        raise failure

    _install_constructor(monkeypatch, create)
    with pytest.raises(type(failure)) as raised:
        embeddings.embed(["text"], model_name="broken-model")
    assert raised.value is failure
    assert embeddings._MODEL is None
    assert calls == [{"local_files_only": True}]


def test_failed_cache_download_preserves_download_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed first download propagates its error instead of publishing partial weights."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[dict[str, object]] = []
    failure = OSError("Download failed")

    def create(name: str, **kwargs: object) -> _Model:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("Not cached")
        raise failure

    _install_constructor(monkeypatch, create)
    with pytest.raises(OSError) as raised:
        embeddings.embed(["text"], model_name="unavailable-model")
    assert raised.value is failure
    assert embeddings._MODEL is None
    assert calls == [{"local_files_only": True}, {}]


def test_local_model_directory_failure_cannot_trigger_hub_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit local model paths cannot be reinterpreted as downloadable Hub repositories."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    directory = tmp_path / "local-model"
    directory.mkdir()
    calls: list[dict[str, object]] = []

    def create(name: str, **kwargs: object) -> _Model:
        calls.append(kwargs)
        raise LocalEntryNotFoundError("Local directory is incomplete")

    _install_constructor(monkeypatch, create)
    with pytest.raises(LocalEntryNotFoundError):
        embeddings.embed(["text"], model_name=str(directory))
    assert embeddings._MODEL is None
    assert calls == [{"local_files_only": True}]
