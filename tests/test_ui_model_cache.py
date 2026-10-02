"""Lazy web-only model resources use primitive keys and bounded caches."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest

from sts.app import model_cache
from sts.config import Settings
from sts.nlp import embeddings, ner
from sts.pipeline import transcribe


@dataclass
class _Factories:
    whisper: list[tuple[str, str, str]] = field(default_factory=list)
    ner: list[object] = field(default_factory=list)
    sbert: list[str] = field(default_factory=list)


@pytest.fixture
def factories(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Factories]:
    """Isolate Streamlit resources and replace every trained-model constructor."""
    calls = _Factories()
    for cached in (model_cache._cached_whisper, model_cache._cached_ner, model_cache._cached_sbert):
        cached.clear()

    def whisper(cfg: Settings) -> object:
        calls.whisper.append((cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute))
        return object()

    def person_entities() -> object:
        resource = object()
        calls.ner.append(resource)
        return resource

    def sbert(model_name: str) -> object:
        calls.sbert.append(model_name)
        return object()

    monkeypatch.setattr(model_cache, "_WHISPER_FACTORY", whisper)
    monkeypatch.setattr(model_cache, "_NER_FACTORY", person_entities)
    monkeypatch.setattr(model_cache, "_SBERT_FACTORY", sbert)
    # Track constructor restoration after install_model_cache's assignments.
    monkeypatch.setattr(transcribe, "_create_model", transcribe._create_model)
    monkeypatch.setattr(ner, "_create_nlp", ner._create_nlp)
    monkeypatch.setattr(embeddings, "_create_model", embeddings._create_model)
    yield calls
    for cached in (model_cache._cached_whisper, model_cache._cached_ner, model_cache._cached_sbert):
        cached.clear()


def test_installation_is_lazy_and_idempotent(factories: _Factories) -> None:
    """Opening and rerunning the upload page does not construct any trained models."""
    model_cache.install_model_cache()
    installed = (transcribe._create_model, ner._create_nlp, embeddings._create_model)
    model_cache.install_model_cache()
    assert installed == (transcribe._create_model, ner._create_nlp, embeddings._create_model)
    assert factories == _Factories()
    assert transcribe._MODEL is ner._NLP is embeddings._MODEL is None


def test_whisper_uses_only_its_model_device_and_compute_identity(factories: _Factories) -> None:
    """Unrelated job settings reuse Whisper weights while its three identity fields key loading."""
    model_cache.install_model_cache()
    first = transcribe._create_model(
        Settings(whisper_model="small", whisper_device="cpu", whisper_compute="int8")
    )
    again = transcribe._create_model(
        Settings(
            whisper_model="small",
            whisper_device="cpu",
            whisper_compute="int8",
            target_minutes=30,
            llm_model="another-llm",
            sbert_model="another-embedding",
        )
    )
    assert again is first
    assert factories.whisper == [("small", "cpu", "int8")]


@pytest.mark.parametrize(
    "changed",
    [
        {"whisper_model": "tiny"},
        {"whisper_device": "cpu"},
        {"whisper_compute": "int8"},
    ],
)
def test_whisper_replaces_resources_for_each_identity_change(
    factories: _Factories, changed: dict[str, str]
) -> None:
    """Each model, device, or compute change creates a distinct lazy Whisper resource."""
    model_cache.install_model_cache()
    first = transcribe._create_model(Settings())
    second = transcribe._create_model(Settings(**changed))
    assert second is not first
    assert len(factories.whisper) == 2


def test_spacy_reuses_one_lazy_resource(factories: _Factories) -> None:
    """Both direct construction and the existing NER singleton share one cached instance."""
    model_cache.install_model_cache()
    first = ner._create_nlp()
    assert ner._create_nlp() is first
    assert ner._get_nlp() is first
    assert factories.ner == [first]


def test_sbert_reuses_a_name_and_evicts_old_configurations(factories: _Factories) -> None:
    """Switching embedding names keeps only the most recent resource in Streamlit's cache."""
    model_cache.install_model_cache()
    first = embeddings._create_model("model-one")
    assert embeddings._create_model("model-one") is first
    second = embeddings._create_model("model-two")
    assert second is not first
    reloaded = embeddings._create_model("model-one")
    assert reloaded is not first
    assert factories.sbert == ["model-one", "model-two", "model-one"]


def test_whisper_cache_is_bounded_when_switching_back(factories: _Factories) -> None:
    """Model changes evict old cached entries instead of accumulating weights."""
    model_cache.install_model_cache()
    first = transcribe._create_model(Settings(whisper_model="small"))
    transcribe._create_model(Settings(whisper_model="tiny"))
    assert transcribe._create_model(Settings(whisper_model="small")) is not first
    assert factories.whisper == [
        ("small", "auto", "auto"),
        ("tiny", "auto", "auto"),
        ("small", "auto", "auto"),
    ]


def test_global_resources_work_in_a_background_worker_without_session_context(
    factories: _Factories,
) -> None:
    """The web runner can acquire shared weights outside Streamlit's rendering thread."""
    model_cache.install_model_cache()
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(transcribe._create_model, Settings()).result(timeout=5)
        again = executor.submit(transcribe._create_model, Settings()).result(timeout=5)
    assert again is first
    assert factories.whisper == [("small", "auto", "auto")]


def test_failed_lazy_construction_is_retried_on_next_request(
    factories: _Factories, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient model-loading failure leaves no successful cache entry to reuse."""
    attempts: list[str] = []

    def create(model_name: str) -> object:
        attempts.append(model_name)
        if len(attempts) == 1:
            raise RuntimeError("model loading failed")
        return object()

    monkeypatch.setattr(model_cache, "_SBERT_FACTORY", create)
    model_cache.install_model_cache()
    with pytest.raises(RuntimeError, match="model loading failed"):
        embeddings._create_model("model-one")
    resource = embeddings._create_model("model-one")
    assert embeddings._create_model("model-one") is resource
    assert attempts == ["model-one", "model-one"]
