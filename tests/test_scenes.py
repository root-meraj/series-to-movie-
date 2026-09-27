"""Scene boundary conversion, short-scene merging, and resumable scene production."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from sts.app.cli import main
from sts.config import Settings
from sts.pipeline import scenes
from sts.pipeline.controller import get_status, resume_job, run_job
from sts.schemas import AudioFeaturesDoc, JobState, ScenesDoc, TranscriptDoc, read_json, write_json


@dataclass
class _Timecode:
    seconds: float

    def get_seconds(self) -> float:
        """Expose only the supported FrameTimecode conversion used by scene detection."""
        return self.seconds


@pytest.mark.parametrize(
    ("bounds", "minimum", "expected"),
    [
        ([(0, 10)], 5.0, [(0.0, 10.0)]),
        ([(0, 2), (2, 10), (10, 20)], 5.0, [(0.0, 10.0), (10.0, 20.0)]),
        ([(0, 10), (10, 12), (12, 20)], 5.0, [(0.0, 10.0), (10.0, 20.0)]),
        ([(0, 10), (10, 12)], 5.0, [(0.0, 12.0)]),
        ([(0, 1), (1, 2), (2, 3)], 5.0, [(0.0, 3.0)]),
        ([(0, 2), (2, 4), (4, 6), (6, 8)], 5.0, [(0.0, 8.0)]),
        ([(0, 5), (5, 10)], 5.0, [(0.0, 5.0), (5.0, 10.0)]),
    ],
)
def test_timecode_conversion_merging_and_full_coverage(
    monkeypatch: pytest.MonkeyPatch,
    bounds: list[tuple[float, float]],
    minimum: float,
    expected: list[tuple[float, float]],
) -> None:
    """Merging retains every frame, including short initial/final scenes and uncut videos."""
    import scenedetect

    seen: dict[str, object] = {}

    def content_detector(**kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    def detect(
        path: Path, detector: object, *, start_in_scene: bool
    ) -> list[tuple[_Timecode, _Timecode]]:
        assert start_in_scene  # Uncut video must still produce a scene.
        return [(_Timecode(start), _Timecode(end)) for start, end in bounds]

    monkeypatch.setattr(scenedetect, "ContentDetector", content_detector)
    monkeypatch.setattr(scenedetect, "detect", detect)
    result = scenes.detect_scenes(
        Path("episode.mp4"), Settings(detector_threshold=31.5, min_scene_seconds=minimum)
    )
    assert seen == {"threshold": 31.5, "min_scene_len": 1}
    assert result == expected
    assert sum(end - start for start, end in result) == bounds[-1][1]
    assert all(isinstance(value, float) for pair in result for value in pair)


@pytest.mark.parametrize(
    "bounds",
    [[], [(1, 10)], [(0, 0)], [(0, float("inf"))], [(0, 10), (11, 20)], [(0, 10), (9, 20)]],
)
def test_invalid_detector_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch, bounds: list[tuple[float, float]]
) -> None:
    """No frames, gaps, overlaps, and non-finite boundaries cannot create scene artifacts."""
    import scenedetect

    monkeypatch.setattr(
        scenedetect,
        "detect",
        lambda *args, **kwargs: [(_Timecode(start), _Timecode(end)) for start, end in bounds],
    )
    with pytest.raises(ValueError):
        scenes.detect_scenes(Path("episode.mp4"), Settings())


@pytest.fixture
def transcribed_job(
    make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> JobState:
    """Prepare saved transcript/RMS contracts without loading Whisper."""
    state = run_job(
        make_zip([("ep1.mp4", b"one"), ("ep2.mp4", b"two"), ("ep3.mp4", b"three")]),
        stage="ingest",
        cfg=Settings(detector_threshold=31.0, min_scene_seconds=6.0),
    )
    directory = Path("data/jobs") / state.job_id
    for episode in state.stages.ingest.completed:
        write_json(
            directory / "transcripts" / f"{episode}.transcript.json",
            TranscriptDoc(episode=episode, language="en", segments=[]),
        )
        write_json(
            directory / "audio" / f"{episode}.rms.json",
            AudioFeaturesDoc(episode=episode, duration=90, rms_per_second=[0.25] * 90),
        )
    state.stages.transcribe.status = "done"
    state.stages.transcribe.completed = list(state.stages.ingest.completed)
    state.stages.transcribe.pending = []
    write_json(directory / "state.json", state)
    return state


@pytest.fixture
def detector_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float, float]]:
    """Record detection units and saved parameters without decoding actual videos."""
    calls: list[tuple[str, float, float]] = []

    def detect(path: Path, cfg: Settings) -> list[tuple[float, float]]:
        calls.append((path.stem, cfg.detector_threshold, cfg.min_scene_seconds))
        return [(0.0, 30.0), (30.0, 90.0)]

    monkeypatch.setattr(scenes, "detect_scenes", detect)
    return calls


def test_scene_stage_writes_all_contract_fields_and_season_bookends(
    transcribed_job: JobState, detector_calls: list[tuple[str, float, float]]
) -> None:
    """Every scene, including silence, is retained with only season-wide first/last flags."""
    completed = resume_job(transcribed_job.job_id, stage="scenes")
    assert completed.stages.scenes.status == "done"
    assert completed.stages.scenes.completed == ["E01", "E02", "E03"]
    assert completed.stages.scenes.pending == []
    records = read_json(Path("data/jobs") / completed.job_id / "scenes.json", ScenesDoc).root
    assert [record.scene_id for record in records] == [
        "E01_s0001",
        "E01_s0002",
        "E02_s0001",
        "E02_s0002",
        "E03_s0001",
        "E03_s0002",
    ]
    assert [record.is_first for record in records] == [True, False, False, False, False, False]
    assert [record.is_last for record in records] == [False, False, False, False, False, True]
    assert all(record.transcript == "" and record.dialogue_ratio == 0 for record in records)
    assert all(record.audio_rms == 0.25 for record in records)
    assert all(record.characters == [] for record in records)
    assert detector_calls == [(episode, 31.0, 6.0) for episode in ("E01", "E02", "E03")]
    for stage in ("nlp", "select", "render"):
        assert getattr(completed.stages, stage).status == "pending"


def test_scene_failure_resumes_only_unfinished_episodes_with_saved_settings(
    transcribed_job: JobState,
    detector_calls: list[tuple[str, float, float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E02 detection failure keeps E01's aligned records and does not repeat its detection."""
    original = scenes.detect_scenes

    def fail_second(path: Path, cfg: Settings) -> list[tuple[float, float]]:
        if path.stem == "E02":
            raise RuntimeError("simulated detector failure")
        return original(path, cfg)

    monkeypatch.setattr(scenes, "detect_scenes", fail_second)
    with pytest.raises(RuntimeError, match="detector failure"):
        resume_job(transcribed_job.job_id, stage="scenes")
    failed = get_status(transcribed_job.job_id)
    assert failed.stages.scenes.completed == ["E01"]
    assert failed.stages.scenes.pending == ["E02", "E03"]
    assert failed.errors[-1].stage == "scenes" and failed.errors[-1].episode == "E02"
    scene_path = Path("data/jobs") / failed.job_id / "scenes.json"
    previous = read_json(scene_path, ScenesDoc).root
    monkeypatch.setattr(scenes, "detect_scenes", original)
    monkeypatch.setenv("DETECTOR_THRESHOLD", "99")
    monkeypatch.setenv("MIN_SCENE_SECONDS", "20")
    completed = resume_job(failed.job_id, stage="scenes")
    assert completed.stages.scenes.status == "done"
    assert len(completed.errors) == 1
    assert read_json(scene_path, ScenesDoc).root[:2] == previous
    assert detector_calls == [(episode, 31.0, 6.0) for episode in ("E01", "E02", "E03")]


def test_interrupt_between_scene_write_and_checkpoint_does_not_duplicate_records(
    transcribed_job: JobState,
    detector_calls: list[tuple[str, float, float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scene records written before a crash are rebuilt if state has not committed the episode."""
    original = scenes.write_json

    def interrupt(path: Path, document: BaseModel) -> None:
        if isinstance(document, JobState) and document.stages.scenes.completed == ["E01"]:
            raise KeyboardInterrupt("simulated process interruption")
        original(path, document)

    monkeypatch.setattr(scenes, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        resume_job(transcribed_job.job_id, stage="scenes")
    directory = Path("data/jobs") / transcribed_job.job_id
    assert get_status(transcribed_job.job_id).stages.scenes.completed == []
    assert len(read_json(directory / "scenes.json", ScenesDoc).root) == 2
    monkeypatch.setattr(scenes, "write_json", original)
    resume_job(transcribed_job.job_id, stage="scenes")
    records = read_json(directory / "scenes.json", ScenesDoc).root
    assert len(records) == len({record.scene_id for record in records}) == 6
    assert [call[0] for call in detector_calls] == ["E01", "E01", "E02", "E03"]


@pytest.mark.parametrize("damage", ["missing", "invalid_json", "missing_episode"])
def test_unfinished_stage_repairs_scene_artifacts(
    transcribed_job: JobState, detector_calls: list[tuple[str, float, float]], damage: str
) -> None:
    """Missing or invalid aggregate records do not cause an unfinished stage to skip needed work."""
    completed = resume_job(transcribed_job.job_id, stage="scenes")
    directory = Path("data/jobs") / completed.job_id
    path = directory / "scenes.json"
    completed.stages.scenes.status = "in_progress"
    write_json(directory / "state.json", completed)
    if damage == "missing":
        path.unlink()
    elif damage == "invalid_json":
        path.write_text("broken JSON", encoding="utf-8")
    else:
        write_json(path, ScenesDoc(read_json(path, ScenesDoc).root[2:]))
    detector_calls.clear()
    assert resume_job(completed.job_id, stage="scenes").stages.scenes.status == "done"
    assert [call[0] for call in detector_calls] == (
        ["E01"] if damage == "missing_episode" else ["E01", "E02", "E03"]
    )
    assert len(read_json(path, ScenesDoc).root) == 6


def test_wrong_transcript_episode_records_failure(
    transcribed_job: JobState, detector_calls: list[tuple[str, float, float]]
) -> None:
    """Misidentified inputs cannot contaminate another episode's scene dialogue."""
    directory = Path("data/jobs") / transcribed_job.job_id
    write_json(
        directory / "transcripts/E01.transcript.json", TranscriptDoc(episode="E02", language="en")
    )
    with pytest.raises(ValueError, match="identifiers must match E01"):
        resume_job(transcribed_job.job_id, stage="scenes")
    failed = get_status(transcribed_job.job_id)
    assert failed.stages.scenes.status == "failed"
    assert failed.stages.scenes.completed == []
    assert failed.errors[-1].episode == "E01"
    assert detector_calls == []


def test_cli_scene_stop_and_completed_stage_are_idempotent(
    transcribed_job: JobState,
    detector_calls: list[tuple[str, float, float]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit scene stop preserves later stages, and repeated resume leaves state unchanged."""
    assert main(["resume", "--job", transcribed_job.job_id, "--stage", "scenes"]) == 0
    assert "scenes done" in capsys.readouterr().out
    directory = Path("data/jobs") / transcribed_job.job_id
    paths = [directory / "state.json", directory / "scenes.json"]
    previous = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    detector_calls.clear()
    assert main(["resume", "--job", transcribed_job.job_id, "--stage", "scenes"]) == 0
    assert detector_calls == []
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == previous
