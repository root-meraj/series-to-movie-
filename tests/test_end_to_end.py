"""ZIP-to-movie orchestration, stage boundaries, and recovery with deterministic models."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from sts.config import Settings
from sts.nlp import embeddings, ner, scorer_llm
from sts.pipeline import controller, render, scenes, transcribe
from sts.schemas import (
    CharacterTable,
    JobState,
    ScenesDoc,
    ScoresDoc,
    SelectedClip,
    SelectionDoc,
    read_json,
)

_STAGES = ("ingest", "transcribe", "scenes", "characters", "baseline", "llm", "select", "render")


@pytest.fixture(autouse=True)
def legacy_pipeline_settings(monkeypatch: pytest.MonkeyPatch, isolate_settings: None) -> None:
    """Exercise the original visual/single-scene contract with deterministic media doubles."""
    for key, value in {
        "SCENE_METHOD": "visual",
        "EXCLUDE_CREDITS": "false",
        "SELECTION_STRATEGY": "density",
        "LLM_BATCH_SIZE": "1",
        "RENDER_ENCODER": "libx264",
    }.items():
        monkeypatch.setenv(key, value)


@dataclass
class _Calls:
    units: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    fail_at: dict[str, int] = field(default_factory=dict)

    def record(self, stage: str, unit: str) -> None:
        """Record one expensive unit and optionally inject its single failure."""
        self.units[stage].append(unit)
        if self.fail_at.get(stage) == len(self.units[stage]):
            raise RuntimeError(f"simulated {stage} interruption")


class _Whisper:
    def __init__(self, calls: _Calls) -> None:
        self.calls = calls

    def transcribe(
        self, wav: NDArray[np.float32], **options: bool
    ) -> tuple[Iterator[SimpleNamespace], SimpleNamespace]:
        """Return lazy dialogue for the episode encoded by its deterministic waveform."""
        episode_number = round(float(wav[0]) * 10)
        episode = f"E{episode_number:02d}"

        def segments() -> Iterator[SimpleNamespace]:
            self.calls.record("transcribe", episode)
            word = ("one", "two", "three")[episode_number - 1]
            yield SimpleNamespace(
                start=0.25,
                end=2.75,
                text=f"Alice discovers clue {word}.",
                no_speech_prob=0.0,
                avg_logprob=-0.1,
            )

        return segments(), SimpleNamespace(language="en")


class _NER:
    def __init__(self, calls: _Calls) -> None:
        self.calls = calls

    def pipe(self, texts: Iterable[str], **options: int) -> Iterator[SimpleNamespace]:
        """Identify Alice with a lightweight PERSON entity double."""
        texts = list(texts)
        self.calls.record("characters", " ".join(texts))
        for text in texts:
            entities = [SimpleNamespace(text="Alice", label_="PERSON")] if "Alice" in text else []
            yield SimpleNamespace(ents=entities)


class _Embedding:
    def __init__(self, calls: _Calls) -> None:
        self.calls = calls

    def get_embedding_dimension(self) -> int:
        """Report the three orthogonal clue dimensions."""
        return 3

    def encode(self, texts: list[str], **options: object) -> NDArray[np.float32]:
        """Keep different episode dialogue distinct without loading a trained model."""
        self.calls.record("baseline", " ".join(texts))
        return np.asarray(
            [[float(word in text) for word in ("one", "two", "three")] for text in texts],
            dtype=np.float32,
        )


class _MockProvider(scorer_llm.MockProvider):
    def __init__(self, calls: _Calls) -> None:
        super().__init__()
        self.ledger = calls

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        """Use the deterministic production MockProvider and inject transport interruptions."""
        self.ledger.record("llm", user)
        return super().complete(system, user, json_mode)


@pytest.fixture
def pipeline_calls(monkeypatch: pytest.MonkeyPatch, probe_calls: list[bytes]) -> _Calls:
    """Run real pipeline stages with tiny WAVs, deterministic models, and fake encoded media."""
    calls = _Calls()

    def extract(video_path: Path, audio_path: Path) -> Path:
        calls.record("extract", video_path.stem)
        value = int(video_path.stem[1:]) / 10
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, np.full(48000, value, dtype=np.float32), 16000, subtype="FLOAT")
        return audio_path

    def detect(video_path: Path, cfg: Settings) -> list[tuple[float, float]]:
        calls.record("scenes", video_path.stem)
        return [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]

    def encode(clip: SelectedClip, path: Path, *, loudnorm: bool = False) -> float:
        calls.record("render", clip.scene_id)
        duration = clip.end - clip.start
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"duration": duration}), encoding="utf-8")
        return duration

    def probe(path: Path, expected_duration: float | None = None) -> float:
        return float(json.loads(path.read_text(encoding="utf-8"))["duration"])

    def concat(paths: list[Path], manifest: Path, path: Path) -> float:
        calls.record("concat", path.name)
        duration = sum(probe(clip) for clip in paths)
        manifest.write_text("\n".join(str(clip) for clip in paths), encoding="utf-8")
        path.write_text(json.dumps({"duration": duration}), encoding="utf-8")
        return duration

    monkeypatch.setattr(transcribe, "extract_audio", extract)
    monkeypatch.setattr(transcribe, "_create_model", lambda cfg: _Whisper(calls))
    monkeypatch.setattr(scenes, "detect_scenes", detect)
    monkeypatch.setattr(ner, "_create_nlp", lambda: _NER(calls))
    monkeypatch.setattr(embeddings, "_create_model", lambda name: _Embedding(calls))
    monkeypatch.setattr(scorer_llm, "create_provider", lambda cfg: _MockProvider(calls))
    monkeypatch.setattr(render, "_tool", lambda name: name)
    monkeypatch.setattr(render, "render_clip", encode)
    monkeypatch.setattr(render, "_probe_encoded", probe)
    monkeypatch.setattr(render, "concat_clips", concat)
    return calls


def _archive(make_zip: Callable[[list[tuple[str, bytes]]], Path]) -> Path:
    return make_zip([("show_1x03.mp4", b"three"), ("ep1.mp4", b"one"), ("ep2.mp4", b"two")])


def _status(state: JobState, stage: str) -> str:
    if stage in {"characters", "baseline", "llm"}:
        return getattr(state.stages.nlp, stage).status
    return getattr(state.stages, stage).status


def _assert_complete(state: JobState, tmp_path: Path) -> Path:
    assert all(_status(state, stage) == "done" for stage in _STAGES)
    assert state.stages.nlp.status == "done"
    directory = tmp_path / "data/jobs" / state.job_id
    scene_records = read_json(directory / "scenes.json", ScenesDoc).root
    score_records = read_json(directory / "scores.json", ScoresDoc).root
    selection = read_json(directory / "selected.json", SelectionDoc)
    characters = read_json(directory / "characters.json", CharacterTable)
    assert len(scene_records) == len(score_records) == 9
    assert {score.scene_id for score in score_records} == {
        scene.scene_id for scene in scene_records
    }
    assert all(score.llm_score == 5 for score in score_records)
    assert selection.selected_total_seconds <= selection.target_seconds == 120
    assert selection.clips[0].scene_id == "E01_s0001"
    assert selection.clips[-1].scene_id == "E03_s0003"
    assert characters.characters[0].name == "Alice"
    assert (directory / "output/condensed_movie.mp4").is_file()
    captions = (directory / "output/condensed.srt").read_text(encoding="utf-8")
    assert "00:00:00,250 --> 00:00:02,750" in captions
    assert "Alice discovers clue one." in captions
    assert "Alice discovers clue two." in captions
    assert "Alice discovers clue three." in captions
    assert read_json(directory / "state.json", JobState) == state
    return directory


@pytest.mark.parametrize("stop", _STAGES)
def test_explicit_stage_stops_at_its_requested_boundary(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    pipeline_calls: _Calls,
    stop: str,
) -> None:
    """Every supported stop commits its work while downstream stages remain pending."""
    state = controller.run_job(_archive(make_zip), 2, cfg=Settings(llm_provider="mock"), stage=stop)
    boundary = _STAGES.index(stop)
    assert [_status(state, stage) for stage in _STAGES] == [
        "done" if index <= boundary else "pending" for index in range(len(_STAGES))
    ]
    directory = tmp_path / "data/jobs" / state.job_id
    assert (directory / "selected.json").exists() == (boundary >= _STAGES.index("select"))
    assert (directory / "output/condensed_movie.mp4").exists() == (stop == "render")
    assert state.errors == []


def test_default_run_connects_real_stage_contracts_from_zip_to_movie(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    pipeline_calls: _Calls,
    probe_calls: list[bytes],
) -> None:
    """The default run executes all stages and publishes a consistent movie/subtitle timeline."""
    state = controller.run_job(_archive(make_zip), 2, cfg=Settings(llm_provider="mock"))
    _assert_complete(state, tmp_path)
    assert probe_calls == [b"one", b"two", b"three"]
    assert pipeline_calls.units["transcribe"] == ["E01", "E02", "E03"]
    assert pipeline_calls.units["scenes"] == ["E01", "E02", "E03"]
    assert len(pipeline_calls.units["baseline"]) == 3  # Selection reuses embedding contracts.
    assert len(pipeline_calls.units["llm"]) == 12  # One summary plus three scenes per episode.
    assert len(pipeline_calls.units["render"]) == 4
    assert len(pipeline_calls.units["concat"]) == 1


def test_default_resume_finishes_an_earlier_stage_using_saved_settings(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    pipeline_calls: _Calls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage-limited job can resume through rendering despite changed process defaults."""
    state = controller.run_job(
        _archive(make_zip),
        2,
        cfg=Settings(llm_provider="mock", min_clip_seconds=15),
        stage="scenes",
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MIN_CLIP_SECONDS", "1000")
    finished = controller.resume_job(state.job_id)
    _assert_complete(finished, tmp_path)
    assert finished.params.llm_provider == "mock"
    assert finished.params.min_clip_seconds == 15
    assert pipeline_calls.units["transcribe"] == ["E01", "E02", "E03"]
    assert pipeline_calls.units["scenes"] == ["E01", "E02", "E03"]


@pytest.mark.parametrize(
    "failed_stage", ["transcribe", "scenes", "characters", "baseline", "llm", "render"]
)
def test_default_resume_recovers_failed_units_without_repeating_prior_work(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    pipeline_calls: _Calls,
    failed_stage: str,
) -> None:
    """Resume retries failed units and finishes the job while preserving upstream work."""
    pipeline_calls.fail_at[failed_stage] = 2
    with pytest.raises(RuntimeError, match=f"simulated {failed_stage} interruption"):
        controller.run_job(_archive(make_zip), 2, cfg=Settings(llm_provider="mock"))
    failed = controller.list_jobs()[0]
    boundary = _STAGES.index(failed_stage)
    assert _status(failed, failed_stage) == "failed"
    assert all(_status(failed, stage) == "done" for stage in _STAGES[:boundary])
    assert all(_status(failed, stage) == "pending" for stage in _STAGES[boundary + 1 :])
    assert len(failed.errors) == 1
    expected_error_stage = (
        "nlp" if failed_stage in {"characters", "baseline", "llm"} else failed_stage
    )
    assert failed.errors[0].stage == expected_error_stage
    first_completed = pipeline_calls.units[failed_stage][0]
    before = {stage: units.copy() for stage, units in pipeline_calls.units.items()}
    pipeline_calls.fail_at.clear()
    finished = controller.resume_job(failed.job_id)
    _assert_complete(finished, tmp_path)
    assert finished.errors == failed.errors
    assert pipeline_calls.units[failed_stage].count(first_completed) == 1
    for stage in _STAGES[:boundary]:
        if stage in before:
            assert pipeline_calls.units[stage] == before[stage]


def test_controller_records_failure_outside_the_stage_handler_and_resumes(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    pipeline_calls: _Calls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each early runner failure leaves one error; recovery preserves both failed attempts."""
    runner = controller.build_scenes

    def fail_before_handler(job_dir: Path) -> JobState:
        raise RuntimeError("scene runner failed before its handler")

    monkeypatch.setattr(controller, "build_scenes", fail_before_handler)
    with pytest.raises(RuntimeError, match="before its handler"):
        controller.run_job(_archive(make_zip), 2, cfg=Settings(llm_provider="mock"))
    failed = controller.list_jobs()[0]
    assert failed.stages.transcribe.status == "done"
    assert failed.stages.scenes.status == "failed"
    assert failed.stages.nlp.status == "pending"
    assert len(failed.errors) == 1
    assert failed.errors[0].stage == "scenes"
    with pytest.raises(RuntimeError, match="before its handler"):
        controller.resume_job(failed.job_id)
    repeated = controller.get_status(failed.job_id)
    assert repeated.stages.scenes.status == "failed"
    assert len(repeated.errors) == 2
    assert repeated.errors[0] == failed.errors[0]
    assert repeated.errors[1].stage == "scenes"
    monkeypatch.setattr(controller, "build_scenes", runner)
    finished = controller.resume_job(failed.job_id)
    _assert_complete(finished, tmp_path)
    assert finished.errors == repeated.errors
    assert pipeline_calls.units["transcribe"] == ["E01", "E02", "E03"]


def test_completed_job_resume_requires_no_source_archive_media_or_models(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    pipeline_calls: _Calls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All completed stages skip and preserve every artifact when the original archive is gone."""
    archive = _archive(make_zip)
    state = controller.run_job(archive, 2, cfg=Settings(llm_provider="mock"))
    directory = _assert_complete(state, tmp_path)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }
    counters = {stage: units.copy() for stage, units in pipeline_calls.units.items()}
    archive.unlink()

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("Completed stages must not acquire media tools or models")

    for module, function in (
        (transcribe, "_create_model"),
        (ner, "_create_nlp"),
        (embeddings, "_create_model"),
        (scorer_llm, "create_provider"),
        (render, "_tool"),
    ):
        monkeypatch.setattr(module, function, unexpected)
    assert controller.resume_job(state.job_id) == state
    assert dict(pipeline_calls.units) == counters
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
