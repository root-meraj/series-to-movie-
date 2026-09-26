"""Job management through Phase 2: ingestion and resumable transcription."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from uuid import uuid4

from sts.config import Settings
from sts.pipeline.ingest import ingest_zip
from sts.pipeline.transcribe import transcribe_job
from sts.schemas import ErrorRecord, JobParams, JobState, read_json, write_json


def _job_dir(job_id: str, jobs_dir: Path) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", job_id) is None:
        raise ValueError("Invalid job ID")
    root = jobs_dir.resolve()
    directory = (root / job_id).resolve()
    if directory.parent != root:
        raise ValueError("Job directory must be within jobs_dir")
    return directory


def _run_ingest(job_dir: Path) -> JobState:
    state = read_json(job_dir / "state.json", JobState)
    if state.stages.ingest.status == "done":
        logging.getLogger(__name__).info(
            "ingest done: skipping completed stage for %s", state.job_id
        )
        return state
    try:
        if state.params.zip_path is None:
            raise ValueError("Job has no saved ZIP path; create a new job")
        ingest_zip(state.params.zip_path, job_dir)
    except Exception as exc:
        # Ingest records per-episode errors itself. The controller catches failures
        # occurring before that handler and must preserve its newest checkpoint.
        state = read_json(job_dir / "state.json", JobState)
        if state.stages.ingest.status != "failed":
            state.stages.ingest.status = "failed"
            state.errors.append(ErrorRecord(stage="ingest", message=str(exc)))
            write_json(job_dir / "state.json", state)
        logging.getLogger(__name__).error("Job %s failed during ingest", state.job_id)
        raise
    return read_json(job_dir / "state.json", JobState)


def _run_stages(job_dir: Path, stage: str) -> JobState:
    state = _run_ingest(job_dir)
    if stage == "ingest":
        return state
    return transcribe_job(job_dir)


def _validate_stage(stage: str) -> None:
    if stage not in {"ingest", "transcribe"}:
        raise ValueError("Only --stage ingest or transcribe is implemented in Phase 2")


def run_job(
    zip_path: Path,
    target_minutes: float | None = None,
    *,
    name: str | None = None,
    jobs_dir: Path = Path("data/jobs"),
    cfg: Settings | None = None,
    stage: str = "transcribe",
) -> JobState:
    """Create a new job and run through transcription, or stop after ingestion."""
    _validate_stage(stage)
    settings = cfg if cfg is not None else Settings()
    minutes = settings.target_minutes if target_minutes is None else target_minutes
    params = JobParams(
        target_seconds=minutes * 60.0,
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
        whisper_compute=settings.whisper_compute,
        detector_threshold=settings.detector_threshold,
        zip_path=zip_path.resolve(),
        name=name,
    )
    job_id = uuid4().hex[:12]
    directory = _job_dir(job_id, jobs_dir)
    directory.mkdir(parents=True, exist_ok=False)
    for child in ("episodes", "audio", "transcripts", "output"):
        (directory / child).mkdir()
    state = JobState(job_id=job_id, params=params)
    write_json(directory / "state.json", state)
    logging.getLogger(__name__).info("Created job %s", job_id)
    return _run_stages(directory, stage)


def resume_job(
    job_id: str, *, jobs_dir: Path = Path("data/jobs"), stage: str = "transcribe"
) -> JobState:
    """Resume through the requested stage, using checkpoints and persisted model settings."""
    _validate_stage(stage)
    return _run_stages(_job_dir(job_id, jobs_dir), stage)


def get_status(job_id: str, *, jobs_dir: Path = Path("data/jobs")) -> JobState:
    """Read a job's validated state without starting pipeline work."""
    return read_json(_job_dir(job_id, jobs_dir) / "state.json", JobState)


def list_jobs(*, jobs_dir: Path = Path("data/jobs")) -> list[JobState]:
    """List valid saved jobs by creation time, newest first."""
    if not jobs_dir.exists():
        return []
    states = [
        get_status(path.parent.name, jobs_dir=jobs_dir) for path in jobs_dir.glob("*/state.json")
    ]
    return sorted(states, key=lambda state: state.created_at, reverse=True)
