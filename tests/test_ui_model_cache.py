"""Lazy web-only model resources use primitive keys and bounded caches."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from sts.app import model_cache
from sts.config import Settings
from sts.nlp import embeddings, ner
from sts.pipeline import transcribe
from sts.schemas import EpisodeInfo, JobParams, JobState, read_json, write_json


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


def test_whisper_identity_change_frees_the_previous_resource_before_allocating_another(
    factories: _Factories, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching a retained ASR model releases both caches before loading its replacement."""
    previous: weakref.ReferenceType[Whisper] | None = None

    class Whisper:
        """Supply a weak-referenceable resource without loading model weights."""

    def create(cfg: Settings) -> Whisper:
        """Verify the previous resource is gone at the next model allocation boundary."""
        if previous is not None:
            gc.collect()
            assert previous() is None
        factories.whisper.append((cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute))
        return Whisper()

    monkeypatch.setattr(model_cache, "_WHISPER_FACTORY", create)
    model_cache.install_model_cache()
    transcribe._get_model(
        Settings(whisper_model="small", whisper_device="cpu", whisper_compute="int8")
    )
    assert transcribe._MODEL is not None
    previous = weakref.ref(transcribe._MODEL[1])
    transcribe._get_model(
        Settings(whisper_model="tiny", whisper_device="cpu", whisper_compute="int8")
    )
    assert previous() is None
    assert transcribe._MODEL is not None
    assert transcribe._MODEL[0] == ("tiny", "cpu", "int8")
    assert factories.whisper == [("small", "cpu", "int8"), ("tiny", "cpu", "int8")]


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


def test_release_whisper_clears_core_and_web_cache_but_keeps_other_model_resources(
    factories: _Factories,
) -> None:
    """Ending ASR frees its cache independently of character and embedding model resources."""
    model_cache.install_model_cache()
    cfg = Settings(whisper_device="cpu", whisper_compute="int8")
    first = transcribe._get_model(cfg)
    assert transcribe._create_model(cfg) is first
    entities = ner._create_nlp()
    vectors = embeddings._create_model("model-one")
    transcribe.release_model()
    assert transcribe._MODEL is None
    assert ner._create_nlp() is entities
    assert embeddings._create_model("model-one") is vectors
    second = transcribe._get_model(cfg)
    assert second is not first
    assert factories.whisper == [("small", "cpu", "int8"), ("small", "cpu", "int8")]
    assert factories.ner == [entities] and factories.sbert == ["model-one"]


def test_releasing_an_empty_whisper_cache_is_idempotent_and_never_loads_models(
    factories: _Factories,
) -> None:
    """Repeated cache release leaves the next job free to load a new model lazily."""
    model_cache.install_model_cache()
    transcribe.release_model()
    transcribe.release_model()
    assert transcribe._MODEL is None
    assert factories == _Factories()
    resource = transcribe._get_model(Settings())
    assert transcribe._get_model(Settings()) is resource
    assert factories.whisper == [("small", "auto", "auto")]


def test_failed_asr_stage_retains_whisper_until_complete_then_next_job_rebuilds_it(
    factories: _Factories, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed episode keeps reusable weights; only the complete stage clears the web resource."""
    decoded: list[object] = []

    class Whisper:
        """Return empty, valid transcripts while recording which resource decoded an episode."""

        def transcribe(
            self, wav: object, **kwargs: object
        ) -> tuple[Iterator[SimpleNamespace], SimpleNamespace]:
            """Supply a deterministic empty segment generator without constructing any model."""
            decoded.append(self)
            return iter([]), SimpleNamespace(language="en")

    def factory(cfg: Settings) -> Whisper:
        """Construct only a recording fake Whisper resource."""
        factories.whisper.append((cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute))
        return Whisper()

    monkeypatch.setattr(model_cache, "_WHISPER_FACTORY", factory)
    model_cache.install_model_cache()
    fail_second_episode = True

    def extract(video_path: Path, audio_path: Path) -> Path:
        """Create tiny PCM audio, with a controllable failure before the second episode."""
        if video_path.stem == "E02" and fail_second_episode:
            raise RuntimeError("Second episode extraction interrupted")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
        return audio_path

    monkeypatch.setattr(transcribe, "extract_audio", extract)

    def make_job(name: str, episodes: int) -> Path:
        """Save the smallest ingested job contract needed to execute the real ASR stage."""
        directory = tmp_path / name
        state = JobState(
            job_id=name, params=JobParams(whisper_device="cpu", whisper_compute="int8")
        )
        state.stages.ingest.status = "done"
        state.stages.ingest.episode_count = episodes
        state.stages.ingest.episodes = [
            EpisodeInfo(
                episode=f"E{number:02d}",
                episode_number=number,
                source_path=directory / "episodes" / f"E{number:02d}.mp4",
                original_name=f"episode{number}.mp4",
                duration=90,
            )
            for number in range(1, episodes + 1)
        ]
        write_json(directory / "state.json", state)
        return directory

    directory = make_job("first-cache-job", 2)
    current_directory = directory
    cleared_states: list[str] = []
    original_clear = model_cache._load_whisper.clear

    def clear() -> None:
        """Record the durable stage status before delegating to Streamlit cache eviction."""
        cleared_states.append(
            read_json(current_directory / "state.json", JobState).stages.transcribe.status
        )
        original_clear()

    monkeypatch.setattr(model_cache._load_whisper, "clear", clear)
    with pytest.raises(RuntimeError, match="extraction interrupted"):
        transcribe.transcribe_job(directory)
    failed = read_json(directory / "state.json", JobState)
    assert failed.stages.transcribe.completed == ["E01"]
    assert failed.stages.transcribe.pending == ["E02"]
    assert failed.stages.transcribe.status == "failed"
    assert cleared_states == []
    assert transcribe._MODEL is not None
    first = transcribe._MODEL[1]
    assert transcribe._create_model(Settings(whisper_device="cpu", whisper_compute="int8")) is first
    assert factories.whisper == [("small", "cpu", "int8")]

    fail_second_episode = False
    assert transcribe.transcribe_job(directory).stages.transcribe.status == "done"
    assert decoded == [first, first]
    assert cleared_states == ["done"]
    assert transcribe._MODEL is None
    current_directory = make_job("next-cache-job", 1)
    assert transcribe.transcribe_job(current_directory).stages.transcribe.status == "done"
    assert len(decoded) == 3 and decoded[-1] is not first
    assert factories.whisper == [("small", "cpu", "int8"), ("small", "cpu", "int8")]
    assert cleared_states == ["done", "done"]
    assert transcribe._MODEL is None
