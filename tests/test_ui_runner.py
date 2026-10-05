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

from sts.app import runner as runner_module
from sts.app.runner import JobRunner
from sts.config import Settings
from sts.pipeline import controller, ingest
from sts.schemas import JobState, write_json


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
def test_corrupt_upload_does_not_publish_partial_job(
    job_runner: JobRunner, payload: bytes
) -> None:
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

    def failing_probe(
        arguments: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
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
