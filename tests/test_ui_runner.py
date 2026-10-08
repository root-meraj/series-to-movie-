"""Durable uploads, pending job creation, and serialized background worker recovery."""

from __future__ import annotations

import io
import subprocess
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, current_thread
from zipfile import BadZipFile, ZipFile

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from sts.app import runner as runner_module
from sts.app import ui
from sts.app.runner import JobRunner
from sts.config import Settings
from sts.pipeline import controller, ingest
from sts.schemas import (
    AudioFeaturesDoc,
    EpisodeInfo,
    JobState,
    RequiredRange,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    ScoresDoc,
    SelectedClip,
    SelectionDoc,
    SelectionStats,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)


def _upload(payload: bytes = b"episode") -> io.BytesIO:
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("ep1.mp4", payload)
    stream.seek(0)
    return stream


def _complete(job_id: str, *, jobs_dir: Path) -> JobState:
    state = controller.get_status(job_id, jobs_dir=jobs_dir)
    for stage in ("ingest", "transcribe", "scenes", "nlp", "select", "render"):
        getattr(state.stages, stage).status = "done"
    for stage in ("characters", "baseline", "llm"):
        getattr(state.stages.nlp, stage).status = "done"
    write_json(jobs_dir / job_id / "state.json", state)
    return state


def _wait_idle(runner: JobRunner) -> None:
    deadline = time.monotonic() + 5
    while runner.busy and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not runner.busy, "Background worker did not finish within five seconds"


@pytest.fixture
def job_runner(tmp_path: Path) -> Iterator[JobRunner]:
    """Close each executor so tests cannot leave background work behind."""
    runner = JobRunner(tmp_path / "jobs")
    try:
        yield runner
    finally:
        runner.shutdown()


def test_create_job_persists_pending_state_without_starting_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The UI can inspect a durable job before any media tool or model is acquired."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Creation must not start stages")

    monkeypatch.setattr(controller, "_run_stages", forbidden)
    state = controller.create_job(
        tmp_path / "not-uploaded-yet.zip",
        3,
        name="My Show",
        jobs_dir=tmp_path / "jobs",
        cfg=Settings(llm_provider="mock", whisper_model="tiny"),
    )
    assert controller.get_status(state.job_id, jobs_dir=tmp_path / "jobs") == state
    assert state.params.target_seconds == 180
    assert state.params.llm_provider == "mock"
    assert state.params.whisper_model == "tiny"
    assert state.params.name == "My Show"
    assert state.errors == []
    assert all(
        getattr(state.stages, name).status == "pending"
        for name in ("ingest", "transcribe", "scenes", "nlp", "select", "render")
    )


def test_run_job_reuses_creation_and_preserves_its_stage_boundary(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI's existing run path still creates once and respects explicit stage stops."""
    create = controller.create_job
    observed: list[JobState] = []

    def observe(*args: object, **kwargs: object) -> JobState:
        state = create(*args, **kwargs)
        observed.append(state)
        return state

    monkeypatch.setattr(controller, "create_job", observe)
    state = controller.run_job(
        make_zip([("ep1.mp4", b"video")]), stage="ingest", jobs_dir=tmp_path / "jobs"
    )
    assert len(observed) == 1
    assert observed[0].job_id == state.job_id
    assert observed[0].stages.ingest.status == "pending"
    assert state.stages.ingest.status == "done"
    assert state.stages.transcribe.status == "pending"
    assert probe_calls == [b"video"]


def test_uploaded_zip_is_durable_uses_fixed_path_and_waits_for_explicit_start(
    job_runner: JobRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published jobs retain the uploaded bytes after browser-owned file objects disappear."""
    calls: list[str] = []

    def resume(job_id: str, *, jobs_dir: Path) -> JobState:
        calls.append(job_id)
        return _complete(job_id, jobs_dir=jobs_dir)

    monkeypatch.setattr(runner_module, "resume_job", resume)
    upload = _upload(b"distinctive episode bytes")
    upload.name = "../../outside.zip"
    original = upload.getvalue()
    upload.seek(0, io.SEEK_END)
    state = job_runner.create(upload, 2, name="Saved Show", cfg=Settings(llm_provider="mock"))
    upload.close()
    assert state.params.zip_path == job_runner.jobs_dir / state.job_id / "input/source.zip"
    assert state.params.zip_path.read_bytes() == original
    assert job_runner.get_state(state.job_id) == state
    assert controller.list_jobs(jobs_dir=job_runner.jobs_dir) == [state]
    assert calls == []
    assert not job_runner.busy
    assert job_runner.active_job_id is None
    assert job_runner.result(state.job_id) is None
    assert job_runner.error(state.job_id) is None
    assert job_runner.start(state.job_id)
    _wait_idle(job_runner)
    assert calls == [state.job_id]
    assert job_runner.result(state.job_id).stages.render.status == "done"


def test_upload_reads_bounded_chunks_and_staging_is_invisible_to_job_discovery(
    job_runner: JobRunner,
) -> None:
    """Large ZIPs stream without whole-file copies or partially published state files."""
    payload = _upload(b"large episode" * 200000).getvalue()

    class TrackingUpload(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            """Require bounded reads and inspect job discovery while publication is incomplete."""
            assert 0 < size <= 1024 * 1024
            assert controller.list_jobs(jobs_dir=job_runner.jobs_dir) == []
            return super().read(size)

        def getvalue(self) -> bytes:
            """Forbid a second whole-file allocation of the incoming upload."""
            raise AssertionError("Uploads must stream through read()")

    state = job_runner.create(TrackingUpload(payload), 2, cfg=Settings(llm_provider="mock"))
    assert state.params.zip_path.read_bytes() == payload
    assert not list(job_runner.jobs_dir.parent.glob(".sts-upload-*"))


@pytest.mark.parametrize("payload", [b"", b"not a ZIP", b"PK\x03\x04truncated"])
def test_corrupt_upload_does_not_publish_partial_job(job_runner: JobRunner, payload: bytes) -> None:
    """A malformed archive fails before any job becomes visible to sidebar discovery."""
    with pytest.raises(BadZipFile):
        job_runner.create(io.BytesIO(payload), 2)
    assert controller.list_jobs(jobs_dir=job_runner.jobs_dir) == []
    assert list(job_runner.jobs_dir.iterdir()) == []
    assert not list(job_runner.jobs_dir.parent.glob(".sts-upload-*"))


def test_interrupted_upload_cleans_staging_and_preserves_existing_job(
    job_runner: JobRunner,
) -> None:
    """An input stream failure leaves existing jobs intact and no visible pending upload."""
    existing = job_runner.create(_upload(), 2)

    class BrokenUpload(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            """Fail after one chunk to exercise interrupted file persistence."""
            if self.tell():
                raise OSError("browser upload interrupted")
            return super().read(1)

    with pytest.raises(OSError, match="upload interrupted"):
        job_runner.create(BrokenUpload(b"partial upload"), 2)
    assert controller.list_jobs(jobs_dir=job_runner.jobs_dir) == [existing]
    assert not list(job_runner.jobs_dir.parent.glob(".sts-upload-*"))


@pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf")])
def test_invalid_target_never_publishes_uploaded_job(job_runner: JobRunner, target: float) -> None:
    """Settings validation happens before job publication and before background submission."""
    with pytest.raises(ValidationError):
        job_runner.create(_upload(), target)
    assert controller.list_jobs(jobs_dir=job_runner.jobs_dir) == []
    assert not job_runner.busy


def test_concurrent_duplicate_starts_launch_one_job_and_other_jobs_are_blocked(
    job_runner: JobRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sessions racing to start the same job cannot submit duplicate model work."""
    entered, release = Event(), Event()
    calls: list[str] = []

    def resume(job_id: str, *, jobs_dir: Path) -> JobState:
        calls.append(job_id)
        entered.set()
        assert release.wait(5)
        return _complete(job_id, jobs_dir=jobs_dir)

    monkeypatch.setattr(runner_module, "resume_job", resume)
    first = job_runner.create(_upload(), 2)
    second = job_runner.create(_upload(), 2)
    local_archive = job_runner.jobs_dir.parent / "local.zip"
    local_archive.write_bytes(_upload().getvalue())
    try:
        with ThreadPoolExecutor(max_workers=8) as clients:
            starts = list(clients.map(lambda _: job_runner.start(first.job_id), range(8)))
        assert starts.count(True) == 1
        assert entered.wait(5)
        assert job_runner.busy
        assert job_runner.active_job_id == first.job_id
        assert job_runner.is_running(first.job_id)
        assert not job_runner.is_running(second.job_id)
        assert job_runner.get_state(first.job_id).stages.ingest.status == "pending"
        assert job_runner.error(first.job_id) is None
        assert job_runner.result(first.job_id) is None
        with pytest.raises(RuntimeError, match="already running"):
            job_runner.start(second.job_id)
        with pytest.raises(RuntimeError, match="already running"):
            job_runner.create(_upload(), 2)
        with pytest.raises(RuntimeError, match="already running"):
            job_runner.create_from_path(local_archive, 2)
    finally:
        release.set()
    _wait_idle(job_runner)
    assert calls == [first.job_id]
    assert not job_runner.start(first.job_id)
    assert job_runner.active_job_id is None
    assert job_runner.start(second.job_id)
    _wait_idle(job_runner)
    assert calls == [first.job_id, second.job_id]


def test_pipeline_failure_is_inspectable_and_successful_retry_clears_worker_error(
    job_runner: JobRunner,
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker exceptions surface alongside durable stage errors and resume reuses the same job."""
    probe = ingest.subprocess.run
    attempts = 0

    def failing_probe(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary FFprobe failure")
        return probe(arguments, **options)

    def ingest_only(job_id: str, *, jobs_dir: Path) -> JobState:
        return controller.resume_job(job_id, jobs_dir=jobs_dir, stage="ingest")

    monkeypatch.setattr(ingest.subprocess, "run", failing_probe)
    monkeypatch.setattr(runner_module, "resume_job", ingest_only)
    state = job_runner.create(_upload(), 2, cfg=Settings(llm_provider="mock"))
    assert job_runner.start(state.job_id)
    _wait_idle(job_runner)
    failed = job_runner.get_state(state.job_id)
    assert failed.stages.ingest.status == "failed"
    assert len(failed.errors) == 1
    assert failed.errors[0].episode == "E01"
    assert job_runner.error(state.job_id) == "temporary FFprobe failure"
    assert job_runner.result(state.job_id) is None
    assert job_runner.start(state.job_id)
    _wait_idle(job_runner)
    assert job_runner.error(state.job_id) is None
    finished = job_runner.result(state.job_id)
    assert finished.stages.ingest.status == "done"
    assert finished.errors == failed.errors
    assert probe_calls == [b"episode"]


def test_early_worker_failure_surfaces_without_losing_pending_state_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures outside stage handlers surface while pending state remains available for retry."""
    events: list[str] = []

    def resume(job_id: str, *, jobs_dir: Path) -> JobState:
        assert current_thread().name.startswith("sts-job")
        state = controller.get_status(job_id, jobs_dir=jobs_dir)
        assert state.stages.ingest.status == "pending"
        events.append("resume")
        if len(events) == 1:
            raise RuntimeError("model cache unavailable")
        return _complete(job_id, jobs_dir=jobs_dir)

    monkeypatch.setattr(runner_module, "resume_job", resume)
    runner = JobRunner(tmp_path / "jobs")
    try:
        state = runner.create(_upload(), 2)
        assert runner.start(state.job_id)
        _wait_idle(runner)
        assert events == ["resume"]
        assert runner.error(state.job_id) == "model cache unavailable"
        assert runner.get_state(state.job_id) == state
        assert runner.start(state.job_id)
        _wait_idle(runner)
        assert events == ["resume", "resume"]
        assert runner.error(state.job_id) is None
        assert runner.result(state.job_id).stages.render.status == "done"
    finally:
        runner.shutdown()


def test_saved_completed_jobs_skip_worker_and_results_are_independent_copies(
    job_runner: JobRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed checkpoints suppress resubmission; caller mutations cannot alter final results."""
    monkeypatch.setattr(runner_module, "resume_job", _complete)
    state = job_runner.create(_upload(), 2)
    assert job_runner.start(state.job_id)
    _wait_idle(job_runner)
    result = job_runner.result(state.job_id)
    result.params.name = "Caller-only change"
    assert job_runner.result(state.job_id).params.name is None
    assert not job_runner.start(state.job_id)
    assert job_runner.get_state(state.job_id).stages.render.status == "done"


def test_invalid_job_ids_and_shutdown_cannot_schedule_work(job_runner: JobRunner) -> None:
    """Lookup validation and a closed runner reject unsafe or unavailable launch requests."""
    for job_id in ("../outside", "..\\outside", "", "missing"):
        with pytest.raises((ValueError, FileNotFoundError)):
            job_runner.start(job_id)
    assert not job_runner.busy
    job_runner.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        job_runner.start("anything")
    with pytest.raises(RuntimeError, match="shut down"):
        job_runner.create(_upload(), 2)
    with pytest.raises(RuntimeError, match="shut down"):
        job_runner.create_from_path(Path("anything.zip"), 2)


def test_local_archive_creation_resolves_and_persists_source_without_reading_payloads(
    job_runner: JobRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local ZIPs stay on disk; pending jobs retain absolute source paths without payload copies."""
    archive = tmp_path / "season.ZIP"
    archive.write_bytes(_upload(b"local episode bytes").getvalue())
    before = archive.stat()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Local archives must not copy or decompress video payloads")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(ZipFile, "read", forbidden)
    monkeypatch.setattr(ZipFile, "open", forbidden)
    state = job_runner.create_from_path(
        archive.relative_to(tmp_path),
        7,
        name="Local Show",
        cfg=Settings(llm_provider="mock", whisper_model="tiny"),
    )
    assert state.params.zip_path == archive.resolve()
    assert state.params.target_seconds == 420
    assert state.params.name == "Local Show"
    assert state.params.llm_provider == "mock"
    assert state.params.whisper_model == "tiny"
    assert archive.stat().st_size == before.st_size
    assert archive.stat().st_mtime_ns == before.st_mtime_ns
    assert not (job_runner.jobs_dir / state.job_id / "input").exists()
    assert job_runner.get_state(state.job_id) == state
    assert controller.list_jobs(jobs_dir=job_runner.jobs_dir) == [state]
    assert all(
        getattr(state.stages, name).status == "pending"
        for name in ("ingest", "transcribe", "scenes", "nlp", "select", "render")
    )
    assert not job_runner.busy


@pytest.mark.parametrize("invalid", ["missing", "directory", "extension", "corrupt", "truncated"])
def test_invalid_local_archive_never_publishes_job(
    job_runner: JobRunner,
    tmp_path: Path,
    invalid: str,
) -> None:
    """File paths, suffixes, and ZIP metadata are validated before creating any job state."""
    archive = tmp_path / "source.zip"
    if invalid == "directory":
        archive.mkdir()
    elif invalid == "extension":
        archive = tmp_path / "source.mp4"
        archive.write_bytes(_upload().getvalue())
    elif invalid == "corrupt":
        archive.write_bytes(b"not a ZIP archive")
    elif invalid == "truncated":
        archive.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises((ValueError, FileNotFoundError, BadZipFile)):
        job_runner.create_from_path(archive, 2)
    assert controller.list_jobs(jobs_dir=job_runner.jobs_dir) == []
    assert not job_runner.busy


@pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf")])
def test_invalid_local_archive_target_never_creates_job(
    job_runner: JobRunner,
    tmp_path: Path,
    target: float,
) -> None:
    """A valid local ZIP still needs a finite positive runtime before job publication."""
    archive = tmp_path / "source.zip"
    archive.write_bytes(_upload().getvalue())
    with pytest.raises(ValidationError):
        job_runner.create_from_path(archive, target)
    assert controller.list_jobs(jobs_dir=job_runner.jobs_dir) == []


def test_uploaded_job_retries_a_transient_windows_directory_lock(
    job_runner: JobRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient Windows reader cannot discard a valid uploaded job during publication."""
    original = Path.replace
    attempts: list[Path] = []

    def replace(source: Path, destination: Path) -> Path:
        if source.parent.name.startswith(".sts-upload-"):
            attempts.append(source)
            if len(attempts) <= 2:
                exception = PermissionError("Temporary directory lock")
                exception.winerror = 5
                raise exception
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(runner_module.time, "sleep", lambda seconds: None)
    state = job_runner.create(_upload(b"preserved episode"), 2)
    assert len(attempts) == 3
    assert job_runner.get_state(state.job_id) == state
    assert state.params.zip_path.is_file()
    assert not job_runner.busy


def _finished_source(runner: JobRunner) -> JobState:
    state = runner.create(
        _upload(),
        2,
        name="Earlier film",
        cfg=Settings(
            llm_provider="mock",
            scene_method="visual",
            selection_strategy="density",
            render_encoder="libx264",
            must_keep_ranges=[RequiredRange(episode="E01", start=3, end=8)],
        ),
    )
    directory = runner.jobs_dir / state.job_id
    scenes: list[SceneRecord] = []
    for episode in (1, 2):
        identifier = f"E{episode:02d}"
        source = directory / f"episodes/{identifier}.mp4"
        source.write_bytes(f"immutable episode {episode}".encode())
        (directory / f"audio/{identifier}.wav").write_bytes(f"saved audio {episode}".encode())
        write_json(
            directory / f"audio/{identifier}.rms.json",
            AudioFeaturesDoc(
                episode=identifier,
                sample_rate=16000,
                duration=90,
                rms_per_second=[0.0] * 90,
            ),
        )
        write_json(
            directory / f"transcripts/{identifier}.transcript.json",
            TranscriptDoc(
                episode=identifier,
                language="en",
                segments=[TranscriptSegment(start=3, end=8, text="An important clue.")],
            ),
        )
        state.stages.ingest.episodes.append(
            EpisodeInfo(
                episode=identifier,
                episode_number=episode,
                source_path=source,
                original_name=f"ep{episode}.mp4",
                duration=90,
            )
        )
        scenes.append(
            SceneRecord(
                scene_id=f"{identifier}_s0001",
                episode=identifier,
                episode_number=episode,
                source_path=source,
                start=0,
                end=90,
                duration=90,
            )
        )
    state.stages.ingest.episode_count = 2
    state.stages.ingest.completed = ["E01", "E02"]
    state.stages.transcribe.completed = ["E01", "E02"]
    for name in ("ingest", "transcribe", "scenes", "nlp", "select", "render"):
        getattr(state.stages, name).status = "done"
    for name in ("characters", "baseline", "llm"):
        getattr(state.stages.nlp, name).status = "done"
    state.stages.render.video_done = True
    state.stages.render.subtitles_done = True
    write_json(directory / "state.json", state)
    write_json(directory / "scenes.json", ScenesDoc(scenes))
    write_json(
        directory / "scores.json",
        ScoresDoc([ScoreRecord(scene_id=scene.scene_id, llm_score=5) for scene in scenes]),
    )
    write_json(
        directory / "selected.json",
        SelectionDoc(
            target_seconds=120,
            selected_total_seconds=30,
            clips=[
                SelectedClip(
                    scene_id=scenes[0].scene_id,
                    episode="E01",
                    source_path=scenes[0].source_path,
                    start=0,
                    end=30,
                )
            ],
            stats=SelectionStats(scenes_kept=1, scenes_dropped=1, original_seconds=180),
        ),
    )
    (directory / "output/condensed_movie.mp4").write_bytes(b"earlier movie kept")
    (directory / "output/condensed.srt").write_text(
        "1\n00:00:03,000 --> 00:00:08,000\nAn important clue.\n\n", encoding="utf-8"
    )
    return state


def _snapshot(directory: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_refine_creates_pending_story_job_preserves_original_and_requires_explicit_start(
    job_runner: JobRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refinement reuses immutable media without rewriting the old export or starting models."""
    original = _finished_source(job_runner)
    source = job_runner.jobs_dir / original.job_id
    before = _snapshot(source)
    launches: list[str] = []

    def resume(job_id: str, *, jobs_dir: Path) -> JobState:
        launches.append(job_id)
        return _complete(job_id, jobs_dir=jobs_dir)

    monkeypatch.setattr(runner_module, "resume_job", resume)
    required = [RequiredRange(episode="E02", start=15, end=45)]
    refined = job_runner.refine(original.job_id, must_keep_ranges=required, target_minutes=2.5)
    directory = job_runner.jobs_dir / refined.job_id
    assert refined.job_id != original.job_id
    assert refined.params.parent_job_id == original.job_id
    assert refined.params.target_seconds == 150
    assert refined.params.scene_method == "subtitles"
    assert refined.params.selection_strategy == "story"
    assert refined.params.exclude_credits
    assert refined.params.llm_batch_size > 1
    assert refined.params.render_encoder == "auto"
    assert refined.params.must_keep_ranges == required
    assert refined.stages.ingest.status == refined.stages.transcribe.status == "done"
    assert all(
        getattr(refined.stages, name).status == "pending"
        for name in ("scenes", "nlp", "select", "render")
    )
    assert not (directory / "scenes.json").exists()
    assert not (directory / "scores.json").exists()
    assert not (directory / "output/condensed_movie.mp4").exists()
    for previous, current in zip(
        original.stages.ingest.episodes, refined.stages.ingest.episodes, strict=True
    ):
        assert current.source_path.parent == directory / "episodes"
        assert current.source_path.read_bytes() == previous.source_path.read_bytes()
        assert current.source_path != previous.source_path
        assert (directory / f"transcripts/{current.episode}.transcript.json").read_bytes() == (
            source / f"transcripts/{current.episode}.transcript.json"
        ).read_bytes()
    assert _snapshot(source) == before
    assert not job_runner.busy
    assert launches == []
    assert job_runner.result(refined.job_id) is None
    assert job_runner.start(refined.job_id)
    _wait_idle(job_runner)
    assert launches == [refined.job_id]
    assert _snapshot(source) == before


def test_refine_default_retains_previous_target_and_protected_moments(
    job_runner: JobRunner,
) -> None:
    """Omitting optional refinement choices keeps the user's existing runtime and inclusions."""
    original = _finished_source(job_runner)
    refined = job_runner.refine(original.job_id)
    assert refined.params.target_seconds == original.params.target_seconds
    assert refined.params.must_keep_ranges == original.params.must_keep_ranges
    assert not job_runner.busy


@pytest.mark.parametrize("damage", ["unknown_episode", "past_end", "missing_artifact", "target"])
def test_refine_invalid_inputs_never_publish_a_new_edit(job_runner: JobRunner, damage: str) -> None:
    """Episode bounds, required media, and positive target time are checked before publication."""
    original = _finished_source(job_runner)
    intervals = [RequiredRange(episode="E01", start=3, end=8)]
    target = 2.0
    if damage == "unknown_episode":
        intervals = [RequiredRange(episode="E03", start=3, end=8)]
    elif damage == "past_end":
        intervals = [RequiredRange(episode="E01", start=85, end=100)]
    elif damage == "missing_artifact":
        (job_runner.jobs_dir / original.job_id / "audio/E01.wav").unlink()
    else:
        target = 0
    with pytest.raises((ValueError, FileNotFoundError)):
        job_runner.refine(original.job_id, must_keep_ranges=intervals, target_minutes=target)
    assert [state.job_id for state in controller.list_jobs(jobs_dir=job_runner.jobs_dir)] == [
        original.job_id
    ]
    assert not job_runner.busy


def test_refine_is_blocked_while_worker_is_active_and_after_shutdown(
    job_runner: JobRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new refinement cannot race an existing writer or use a closed runner."""
    original = _finished_source(job_runner)
    pending = job_runner.create(_upload(), 2)
    entered, release = Event(), Event()

    def resume(job_id: str, *, jobs_dir: Path) -> JobState:
        entered.set()
        assert release.wait(5)
        return _complete(job_id, jobs_dir=jobs_dir)

    monkeypatch.setattr(runner_module, "resume_job", resume)
    try:
        assert job_runner.start(pending.job_id)
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="Stop the current edit"):
            job_runner.refine(original.job_id)
        assert len(controller.list_jobs(jobs_dir=job_runner.jobs_dir)) == 2
    finally:
        release.set()
    _wait_idle(job_runner)
    job_runner.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        job_runner.refine(original.job_id)


class _RefinementRunner(JobRunner):
    def __init__(self, jobs_dir: Path) -> None:
        """Use actual refinement publication while recording launches without starting models."""
        super().__init__(jobs_dir)
        self.running_id: str | None = None
        self.started: list[str] = []
        self.refined: list[JobState] = []

    @property
    def busy(self) -> bool:
        """Expose the manually controlled worker's occupancy for real UI disabled states."""
        return self.running_id is not None

    def is_running(self, job_id: str) -> bool:
        """Expose an active job without a real background media/model task."""
        return self.running_id == job_id

    def refine(
        self,
        job_id: str,
        *,
        must_keep_ranges: list[RequiredRange] | None = None,
        target_minutes: float | None = None,
    ) -> JobState:
        """Create the real fresh edit and retain its validated choices for UI assertions."""
        state = super().refine(
            job_id, must_keep_ranges=must_keep_ranges, target_minutes=target_minutes
        )
        self.refined.append(state)
        return state

    def start(self, job_id: str) -> bool:
        """Record a requested start without submitting expensive pipeline work."""
        if self.running_id == job_id:
            return False
        self.started.append(job_id)
        self.running_id = job_id
        return True


@pytest.fixture
def refinement_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[_RefinementRunner, JobState, AppTest]]:
    """Exercise production Streamlit controls with saved media and a controlled worker."""
    runner = _RefinementRunner(tmp_path / "jobs")
    state = _finished_source(runner)
    monkeypatch.setattr(ui, "_jobs_dir", lambda: runner.jobs_dir)
    monkeypatch.setattr(ui, "_get_runner", lambda jobs_dir: runner)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    app = AppTest.from_string("from sts.app.ui import main\nmain()", default_timeout=10)
    app.session_state["active_job"] = state.job_id
    app.run()
    try:
        yield runner, state, app
    finally:
        runner.shutdown()


def test_refinement_ui_converts_minutes_to_seconds_keeps_old_pin_and_starts_new_job(
    refinement_ui: tuple[_RefinementRunner, JobState, AppTest],
) -> None:
    """Choosing another episode and decimal minute bounds creates one protected new film."""
    runner, original, app = refinement_ui
    before = _snapshot(runner.jobs_dir / original.job_id)
    app.checkbox(key=f"protect_{original.job_id}").check().run()
    app.selectbox(key=f"required_episode_{original.job_id}").select("E02")
    app.number_input(key=f"required_start_{original.job_id}").set_value(0.25)
    app.number_input(key=f"required_end_{original.job_id}").set_value(0.75)
    app.button(key=f"refine_{original.job_id}").click().run()
    assert not app.exception
    assert len(runner.refined) == 1
    new = runner.refined[0]
    assert new.params.must_keep_ranges == [
        RequiredRange(episode="E01", start=3, end=8),
        RequiredRange(episode="E02", start=15, end=45),
    ]
    assert runner.started == [new.job_id]
    assert app.session_state["active_job"] == new.job_id
    assert _snapshot(runner.jobs_dir / original.job_id) == before
    app.run()
    assert runner.started == [new.job_id]


@pytest.mark.parametrize("bounds", [(0.5, 0.5), (1.0, 0.5), (1.0, 2.0)])
def test_refinement_ui_invalid_times_show_error_without_creating_or_starting_edit(
    refinement_ui: tuple[_RefinementRunner, JobState, AppTest], bounds: tuple[float, float]
) -> None:
    """Zero/reversed intervals and times past the episode end cannot publish another edit."""
    runner, original, app = refinement_ui
    app.checkbox(key=f"protect_{original.job_id}").check().run()
    app.number_input(key=f"required_start_{original.job_id}").set_value(bounds[0])
    app.number_input(key=f"required_end_{original.job_id}").set_value(bounds[1])
    app.button(key=f"refine_{original.job_id}").click().run()
    assert not app.exception
    assert any("Couldn't refine this edit" in error.value for error in app.error)
    assert runner.refined == runner.started == []
    assert app.session_state["active_job"] == original.job_id
    assert len(controller.list_jobs(jobs_dir=runner.jobs_dir)) == 1


def test_refinement_ui_disables_new_edit_while_another_worker_runs(
    refinement_ui: tuple[_RefinementRunner, JobState, AppTest],
) -> None:
    """A completed project's refinement button cannot queue competing processing work."""
    runner, original, app = refinement_ui
    runner.running_id = "another_job"
    app.run()
    assert app.button(key=f"refine_{original.job_id}").disabled
    assert runner.refined == runner.started == []


@pytest.mark.parametrize("seconds, warning", [(30.0, True), (117.6, False), (120.0, False)])
def test_project_ui_reports_underfilled_runtime_and_only_warns_below_98_percent(
    refinement_ui: tuple[_RefinementRunner, JobState, AppTest], seconds: float, warning: bool
) -> None:
    """The displayed film length and guidance distinguish underfilled edits from rounding gaps."""
    runner, original, app = refinement_ui
    path = runner.jobs_dir / original.job_id / "selected.json"
    clips = [
        SelectedClip(
            scene_id="E01_s0001",
            episode="E01",
            source_path=original.stages.ingest.episodes[0].source_path,
            start=0,
            end=min(seconds, 90),
        )
    ]
    if seconds > 90:
        clips.append(
            SelectedClip(
                scene_id="E02_s0001",
                episode="E02",
                source_path=original.stages.ingest.episodes[1].source_path,
                start=0,
                end=seconds - 90,
            )
        )
    selection = SelectionDoc(
        target_seconds=120,
        selected_total_seconds=seconds,
        clips=clips,
        stats=SelectionStats(
            scenes_kept=len(clips), scenes_dropped=2 - len(clips), original_seconds=180
        ),
    )
    write_json(path, selection)
    app.run()
    assert not app.exception
    warnings = [element.value for element in app.warning]
    assert any("shorter than your target" in value for value in warnings) is warning
    stats = " ".join(element.value for element in app.markdown)
    assert "Film runtime" in stats
    assert f"{seconds / 60:.1f} min" in stats


def test_nlp_progress_advances_within_episode_and_refreshes_before_episode_checkpoint(
    refinement_ui: tuple[_RefinementRunner, JobState, AppTest],
) -> None:
    """Saved story-block progress advances the UI while the same episode remains pending."""
    runner, state, app = refinement_ui
    state.stages.render.status = "pending"
    state.stages.render.video_done = state.stages.render.subtitles_done = False
    state.stages.select.status = "pending"
    state.stages.nlp.status = "in_progress"
    story = state.stages.nlp.llm
    story.status = "in_progress"
    story.completed = ["E01"]
    story.pending = ["E02"]
    story.active_episode = "E02"
    story.summary_done = True
    story.scene_count = 10
    story.completed_scenes = [f"E02_s{index:04d}" for index in range(1, 3)]
    path = runner.jobs_dir / state.job_id / "state.json"
    write_json(path, state)
    runner.running_id = state.job_id
    app.run()
    assert not app.exception
    expected = (2 + (1 + 0.1 + 0.9 * 0.2) / 2) / 3
    assert ui._stage_progress(state, "nlp") == pytest.approx(expected)
    assert app.get("progress")[4].proto.value == int(expected * 100)
    assert "E02 · 2 of 10 blocks" in " ".join(element.value for element in app.markdown)
    before = app.get("progress")[4].proto.value
    story.completed_scenes = [f"E02_s{index:04d}" for index in range(1, 7)]
    write_json(path, state)
    app.run()
    assert app.get("progress")[4].proto.value > before
    assert "E02 · 6 of 10 blocks" in " ".join(element.value for element in app.markdown)
    assert read_json(path, JobState).stages.nlp.llm.pending == ["E02"]
    assert runner.started == []


@pytest.mark.parametrize("summary_done,count", [(False, 0), (True, 0), (True, 10)])
def test_nlp_partial_progress_caps_uncommitted_episode_and_describes_summary_work(
    summary_done: bool, count: int
) -> None:
    """Summary work and a fully scored pending episode stay below completion until committed."""
    state = JobState(job_id="nlp_progress")
    state.stages.nlp.status = "in_progress"
    state.stages.nlp.characters.status = state.stages.nlp.baseline.status = "done"
    story = state.stages.nlp.llm
    story.status = "in_progress"
    story.pending = ["E01"]
    story.active_episode = "E01"
    story.summary_done = summary_done
    story.scene_count = 10
    story.completed_scenes = [f"E01_s{index:04d}" for index in range(1, count + 1)]
    active = min(0.99, 0.1 * summary_done + 0.9 * count / 10)
    assert ui._stage_progress(state, "nlp") == pytest.approx((2 + active) / 3)
    assert ui._stage_progress(state, "nlp") < 1
    if not summary_done:
        assert ui._stage_detail(state, "nlp") == "E01 · Reading the story"
    else:
        assert ui._stage_detail(state, "nlp") == f"E01 · {count} of 10 blocks"
    state.stages.nlp.status = "done"
    assert ui._stage_progress(state, "nlp") == 1


@pytest.mark.parametrize("processing", [False, True])
def test_project_shows_saved_protected_moments_outside_refinement_controls(
    refinement_ui: tuple[_RefinementRunner, JobState, AppTest], processing: bool
) -> None:
    """Saved source timestamps remain visible even while processing hides the refinement form."""
    runner, state, app = refinement_ui
    state.stages.ingest.episodes[0].duration = 500
    state.stages.ingest.episodes[1].duration = 4000
    state.params.must_keep_ranges = [
        RequiredRange(episode="E01", start=210, end=390),
        RequiredRange(episode="E02", start=3661.25, end=3720),
    ]
    if processing:
        state.stages.render.status = "in_progress"
        runner.running_id = state.job_id
        # No selection exists yet, so a saved requirement must not claim it has
        # already been included in a finished or selected movie.
        (runner.jobs_dir / state.job_id / "selected.json").unlink()
    else:
        selection_path = runner.jobs_dir / state.job_id / "selected.json"
        selection = read_json(selection_path, SelectionDoc)
        selection.stats.original_seconds = 4500
        write_json(selection_path, selection)
    write_json(runner.jobs_dir / state.job_id / "state.json", state)
    app.run()
    assert not app.exception
    captions = [caption.value for caption in app.caption if "Protected moment" in caption.value]
    assert captions == [
        "Protected moment · E01 · 3:30–6:30 · Saved request",
        "Protected moment · E02 · 1:01:01.3–1:02:00 · Saved request",
    ]
    assert all("selected" not in caption and "included" not in caption for caption in captions)
    assert runner.started == []
    if processing:
        assert not app.get("checkbox")
        assert not any(expander.label == "Refine the story" for expander in app.expander)


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0:00"),
        (59.95, "1:00"),
        (213.673, "3:33.7"),
        (3599.95, "1:00:00"),
        (3661.25, "1:01:01.3"),
        (36000, "10:00:00"),
    ],
)
def test_protected_moment_clock_format_handles_fractional_seconds_and_hour_rollover(
    seconds: float, expected: str
) -> None:
    """Readable source timestamps preserve tenths and carry rounding into minutes and hours."""
    assert ui._moment_time(seconds) == expected
