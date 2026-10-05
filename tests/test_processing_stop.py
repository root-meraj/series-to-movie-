"""Safe stops interrupt active work and retain reusable completed checkpoints."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sts.app import runner as runner_module
from sts.app.runner import JobRunner
from sts.config import Settings
from sts.nlp import scorer_llm
from sts.pipeline import control, controller, scenes, transcribe
from sts.schemas import (
    AudioFeaturesDoc,
    JobState,
    ProcessingControl,
    ScenesDoc,
    SummaryResponse,
    TranscriptDoc,
    read_json,
    write_json,
)


def test_control_does_not_write_checkpoints_and_missing_control_keeps_running(
    tmp_path: Path,
) -> None:
    """Control requests cannot overwrite worker progress, and old jobs need no migration."""
    state = JobState(job_id="saved")
    write_json(tmp_path / "state.json", state)
    original = (tmp_path / "state.json").read_bytes()
    assert not control.stop_requested(tmp_path)
    control.check_stop(tmp_path)
    control.set_stop_requested(tmp_path, True)
    with pytest.raises(control.ProcessingStopped):
        control.check_stop(tmp_path)
    assert (tmp_path / "state.json").read_bytes() == original
    control.set_stop_requested(tmp_path, False)
    control.check_stop(tmp_path)
    assert (tmp_path / "state.json").read_bytes() == original


@pytest.mark.parametrize("invalid", ["true", 1, None])
def test_control_requires_an_explicit_boolean(invalid: object) -> None:
    """Malformed requests cannot silently coerce into stopping a job."""
    with pytest.raises(ValidationError):
        ProcessingControl(stop_requested=invalid)


def test_media_stop_terminates_and_reaps_only_the_owned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real long-running child exits promptly on stop instead of continuing in the background."""
    control.set_stop_requested(tmp_path, False)
    created = Event()
    children: list[subprocess.Popen[str]] = []
    original = subprocess.Popen

    def launch(*args: object, **kwargs: object) -> subprocess.Popen[str]:
        process = original(*args, **kwargs)
        children.append(process)
        created.set()
        return process

    monkeypatch.setattr(control.subprocess, "Popen", launch)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            control.run_media,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
        )
        assert created.wait(5)
        control.set_stop_requested(tmp_path, True)
        with pytest.raises(control.ProcessingStopped):
            future.result(timeout=5)
    assert len(children) == 1
    assert children[0].poll() is not None


def test_controlled_media_success_and_failure_preserve_normal_subprocess_behavior(
    tmp_path: Path,
) -> None:
    """Polling retains output and nonzero exit errors needed by existing stage handlers."""
    control.set_stop_requested(tmp_path, False)
    result = control.run_media([sys.executable, "-c", "print('ready')"], tmp_path)
    assert result.stdout.strip() == "ready"
    with pytest.raises(subprocess.CalledProcessError) as raised:
        control.run_media([sys.executable, "-c", "raise SystemExit(4)"], tmp_path)
    assert raised.value.returncode == 4


def test_runner_stop_releases_worker_without_failure_and_resume_clears_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping blocks overlapping launches until acknowledged, then permits a fresh resume."""
    runner = JobRunner(tmp_path / "jobs")
    entered, acknowledge = Event(), Event()
    attempts = 0

    def resume(job_id: str, *, jobs_dir: Path) -> JobState:
        nonlocal attempts
        attempts += 1
        state = controller.get_status(job_id, jobs_dir=jobs_dir)
        control.check_stop(jobs_dir / job_id)
        if attempts == 1:
            state.stages.ingest.status = "done"
            state.stages.transcribe.status = "in_progress"
            write_json(jobs_dir / job_id / "state.json", state)
            entered.set()
            assert acknowledge.wait(5)
            control.check_stop(jobs_dir / job_id)
        state.stages.render.status = "done"
        write_json(jobs_dir / job_id / "state.json", state)
        return state

    monkeypatch.setattr(runner_module, "resume_job", resume)
    state = controller.create_job(tmp_path / "unused.zip", jobs_dir=runner.jobs_dir)
    other = controller.create_job(tmp_path / "unused.zip", jobs_dir=runner.jobs_dir)
    try:
        assert runner.start(state.job_id)
        assert entered.wait(5)
        assert not runner.request_stop(other.job_id)
        assert runner.request_stop(state.job_id)
        assert runner.request_stop(state.job_id)
        assert runner.is_stopping(state.job_id)
        assert runner.busy
        with pytest.raises(RuntimeError, match="already running"):
            runner.start(other.job_id)
        acknowledge.set()
        paused = runner._futures[state.job_id].result(timeout=5)
        assert not runner.busy
        assert not runner.is_stopping(state.job_id)
        assert runner.error(state.job_id) is None
        assert paused.stages.ingest.status == "done"
        assert paused.stages.transcribe.status == "in_progress"
        assert paused.errors == []
        assert not runner.request_stop(state.job_id)
        assert runner.start(state.job_id)
        resumed = runner._futures[state.job_id].result(timeout=5)
        assert resumed.stages.render.status == "done"
        assert not control.stop_requested(runner.jobs_dir / state.job_id)
    finally:
        acknowledge.set()
        runner.shutdown()


def test_stop_during_scene_detection_preserves_prior_episode_and_resume_reuses_it(
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted episode never publishes partial scenes or records a processing error."""
    state = controller.run_job(
        make_zip([("ep1.mp4", b"one"), ("ep2.mp4", b"two")]), stage="ingest"
    )
    directory = Path("data/jobs") / state.job_id
    for episode in state.stages.ingest.completed:
        write_json(
            directory / "transcripts" / f"{episode}.transcript.json",
            TranscriptDoc(episode=episode, language="en", segments=[]),
        )
        write_json(
            directory / "audio" / f"{episode}.rms.json",
            AudioFeaturesDoc(episode=episode, duration=90, rms_per_second=[0.1] * 90),
        )
    state.stages.transcribe.status = "done"
    write_json(directory / "state.json", state)
    calls: list[str] = []

    def detect(path: Path, cfg: Settings) -> list[tuple[float, float]]:
        calls.append(path.stem)
        if calls == ["E01", "E02"]:
            control.set_stop_requested(directory, True)
            control.check_stop(directory)
        return [(0.0, 90.0)]

    monkeypatch.setattr(scenes, "detect_scenes", detect)
    with pytest.raises(control.ProcessingStopped):
        controller.resume_job(state.job_id, stage="scenes")
    paused = controller.get_status(state.job_id)
    assert paused.stages.scenes.completed == ["E01"]
    assert paused.stages.scenes.pending == ["E02"]
    assert paused.stages.scenes.status == "in_progress"
    assert paused.errors == []
    records = read_json(directory / "scenes.json", ScenesDoc).root
    assert [record.episode for record in records] == ["E01"]
    control.set_stop_requested(directory, False)
    finished = controller.resume_job(state.job_id, stage="scenes")
    assert finished.stages.scenes.status == "done"
    assert calls == ["E01", "E02", "E02"]


def test_scene_stop_is_checked_on_frames_even_without_detected_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uncut footage cannot defer the stop request until the episode ends."""
    import scenedetect

    directory = tmp_path / "job"
    control.set_stop_requested(directory, False)
    monkeypatch.setattr(control, "monotonic", iter([0.0, 1.0]).__next__)
    monkeypatch.setattr(
        scenedetect,
        "ContentDetector",
        lambda **kwargs: SimpleNamespace(process_frame=lambda *a: []),
    )

    def detect(path: Path, detector: object, **kwargs: object) -> list[object]:
        detector.process_frame(None, None)
        control.set_stop_requested(directory, True)
        detector.process_frame(None, None)
        raise AssertionError("Detection must have stopped on the second frame")

    monkeypatch.setattr(scenedetect, "detect", detect)
    with pytest.raises(control.ProcessingStopped):
        scenes.detect_scenes(directory / "episodes/E01.mp4", Settings())


def test_stop_prevents_another_llm_retry_after_the_active_request_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop cannot turn a malformed active response into several more model calls."""
    mock = scorer_llm.MockProvider()
    calls: list[str] = []

    def complete(system: str, user: str, json_mode: bool = True) -> str:
        calls.append(user)
        control.set_stop_requested(tmp_path, True)
        return "invalid JSON"

    monkeypatch.setattr(mock, "complete", complete)
    guarded = scorer_llm._StoppingProvider(mock, tmp_path)
    with pytest.raises(control.ProcessingStopped):
        scorer_llm._complete_json(guarded, "system", "summary", SummaryResponse)
    assert calls == ["summary"]


def test_stopped_whisper_generator_never_publishes_a_partial_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decoded first segment remains uncommitted if its episode is stopped before completion."""
    import numpy as np

    control.set_stop_requested(tmp_path, False)
    audio = tmp_path / "audio/E01.wav"
    transcript = tmp_path / "transcripts/E01.transcript.json"
    monkeypatch.setattr(transcribe.sf, "read", lambda *a, **k: (np.ones(16000), 16000))

    def decoded() -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(start=0.0, end=0.3, text="Hello", no_speech_prob=0.0, avg_logprob=0.0)
        control.set_stop_requested(tmp_path, True)
        yield SimpleNamespace(start=0.3, end=0.6, text="Again", no_speech_prob=0.0, avg_logprob=0.0)

    model = SimpleNamespace(transcribe=lambda *a, **k: (decoded(), SimpleNamespace(language="en")))
    monkeypatch.setattr(transcribe, "_get_model", lambda cfg: model)
    with pytest.raises(control.ProcessingStopped):
        transcribe.transcribe_episode(audio, transcript, Settings())
    assert not transcript.exists()
    assert not audio.with_suffix(".rms.json").exists()


def test_active_provider_timeout_during_stop_is_acknowledged_without_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that times out while stopping cannot turn the user's pause into an error."""
    mock = scorer_llm.MockProvider()

    def complete(system: str, user: str, json_mode: bool = True) -> str:
        control.set_stop_requested(tmp_path, True)
        raise scorer_llm.ProviderError("The active request timed out")

    monkeypatch.setattr(mock, "complete", complete)
    guarded = scorer_llm._StoppingProvider(mock, tmp_path)
    with pytest.raises(control.ProcessingStopped):
        guarded.complete("system", "summary")
