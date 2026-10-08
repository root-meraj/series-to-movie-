"""Durable uploaded jobs and a single background worker shared by Streamlit sessions."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import BinaryIO
from zipfile import ZipFile

from sts.config import Settings
from sts.pipeline.control import ProcessingStopped, set_stop_requested, stop_requested
from sts.pipeline.controller import create_improved_job, create_job, get_status, resume_job
from sts.pipeline.transcribe import release_model
from sts.schemas import JobState, RequiredRange, write_json

_UPLOAD_CHUNK_BYTES = 1024 * 1024


class JobRunner:
    """Serialize model work and keep uploaded archives independent of browser sessions."""

    def __init__(self, jobs_dir: Path = Path("data/jobs")) -> None:
        """Create a lazily started executor with one worker for this jobs directory."""
        self.jobs_dir = jobs_dir.resolve()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sts-job")
        self._lock = Lock()
        self._futures: dict[str, Future[JobState]] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This job runner has been shut down")

    def _active(self) -> str | None:
        return next((job_id for job_id, future in self._futures.items() if not future.done()), None)

    @property
    def busy(self) -> bool:
        """Return whether the worker is processing a job, across all browser sessions."""
        with self._lock:
            return self._active() is not None

    @property
    def active_job_id(self) -> str | None:
        """Return the active job's identifier or None when the worker is idle."""
        with self._lock:
            return self._active()

    def create(
        self,
        upload: BinaryIO,
        target_minutes: float,
        *,
        name: str | None = None,
        cfg: Settings | None = None,
    ) -> JobState:
        """Stream and atomically publish a pending job with a durable input/source.zip.

        Hidden staging prevents partial uploads from appearing in the job list.
        ZIP structure is checked before publication; ingestion performs media and
        per-entry validation later. A browser filename never determines disk paths.
        """
        with self._lock:
            self._ensure_open()
            if self._active() is not None:
                raise RuntimeError("Another job is already running; wait for it to finish")
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".sts-upload-", dir=self.jobs_dir.parent) as temporary:
                staging = Path(temporary)
                state = create_job(
                    staging / "source.zip",
                    target_minutes,
                    name=name,
                    jobs_dir=staging,
                    cfg=cfg,
                )
                directory = staging / state.job_id
                input_dir = directory / "input"
                input_dir.mkdir()
                pending = input_dir / ".source.zip.part"
                try:
                    upload.seek(0)
                except OSError:
                    pass  # Non-seekable binary streams are consumed from their current position.
                with pending.open("wb") as stream:
                    while block := upload.read(_UPLOAD_CHUNK_BYTES):
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                with ZipFile(pending) as archive:
                    archive.infolist()
                pending.replace(input_dir / "source.zip")
                destination = self.jobs_dir / state.job_id
                state.params.zip_path = destination / "input/source.zip"
                write_json(directory / "state.json", state)
                for attempt in range(5):
                    try:
                        directory.replace(destination)
                        break
                    except PermissionError as exc:
                        # Windows indexers can briefly hold a newly created directory.
                        # Keep atomic publication and never remove an existing destination.
                        if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                            raise
                        time.sleep(0.05 * 2**attempt)
                return state

    def create_from_path(
        self,
        zip_path: Path,
        target_minutes: float,
        *,
        name: str | None = None,
        cfg: Settings | None = None,
    ) -> JobState:
        """Create a pending job referencing a validated ZIP already on local disk.

        Resolve the archive path and read ZIP metadata without copying its video
        payloads. Keep the original archive available until ingestion completes.
        Invalid paths or ZIP structures never publish a new job.
        """
        with self._lock:
            self._ensure_open()
            if self._active() is not None:
                raise RuntimeError("Another job is already running; wait for it to finish")
            source = zip_path.expanduser().resolve(strict=True)
            if not source.is_file() or source.suffix.casefold() != ".zip":
                raise ValueError("Choose a local .zip file")
            with ZipFile(source) as archive:
                archive.infolist()
            return create_job(
                source,
                target_minutes,
                name=name,
                jobs_dir=self.jobs_dir,
                cfg=cfg,
            )

    def start(self, job_id: str) -> bool:
        """Start or resume one job, returning False for repeated active or completed launches.

        Another active job raises RuntimeError instead of queuing model work.
        A failed future may be retried; replacing it clears the old worker error.
        """
        with self._lock:
            self._ensure_open()
            previous = self._futures.get(job_id)
            if previous is not None and not previous.done():
                return False
            if self._active() is not None:
                raise RuntimeError("Another job is already running; wait for it to finish")
            state = get_status(job_id, jobs_dir=self.jobs_dir)
            if state.stages.render.status == "done":
                return False
            set_stop_requested(self.jobs_dir / job_id, False)
            self._futures[job_id] = self._executor.submit(self._run, job_id)
            return True

    def refine(
        self,
        job_id: str,
        *,
        must_keep_ranges: list[RequiredRange] | None = None,
        target_minutes: float | None = None,
    ) -> JobState:
        """Create a fresh story edit from saved episodes while preserving the earlier export."""
        with self._lock:
            self._ensure_open()
            if self._active() is not None:
                raise RuntimeError("Stop the current edit or wait for it to finish before refining")
            return create_improved_job(
                job_id,
                jobs_dir=self.jobs_dir,
                must_keep_ranges=must_keep_ranges,
                target_minutes=target_minutes,
            )

    def _run(self, job_id: str) -> JobState:
        try:
            return resume_job(job_id, jobs_dir=self.jobs_dir)
        except ProcessingStopped:
            logging.getLogger(__name__).info("processing stopped: %s; checkpoints saved", job_id)
            release_model()
            return get_status(job_id, jobs_dir=self.jobs_dir)

    def request_stop(self, job_id: str) -> bool:
        """Request a safe stop for this active job, returning False when it is already idle."""
        with self._lock:
            if self._active() != job_id:
                return False
            set_stop_requested(self.jobs_dir / job_id, True)
            return True

    def is_stopping(self, job_id: str) -> bool:
        """Return whether the active worker is acknowledging this job's stop request."""
        with self._lock:
            return self._active() == job_id and stop_requested(self.jobs_dir / job_id)

    def is_running(self, job_id: str) -> bool:
        """Return whether this job is currently executing in the background worker."""
        with self._lock:
            future = self._futures.get(job_id)
            return future is not None and not future.done()

    def get_state(self, job_id: str) -> JobState:
        """Read the newest atomic state checkpoint without waiting for model work."""
        return get_status(job_id, jobs_dir=self.jobs_dir)

    def error(self, job_id: str) -> str | None:
        """Return a finished worker's exception message; pending/successful jobs have none."""
        with self._lock:
            future = self._futures.get(job_id)
        if future is None or not future.done():
            return None
        exception = future.exception()
        return str(exception) if exception is not None else None

    def result(self, job_id: str) -> JobState | None:
        """Return a successful worker's final state without blocking on a running/failed job."""
        with self._lock:
            future = self._futures.get(job_id)
        if future is None or not future.done() or future.exception() is not None:
            return None
        return future.result().model_copy(deep=True)

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting work and let any existing job finish without cancelling its stages."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait)
