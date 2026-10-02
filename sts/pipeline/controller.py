"""Job management through Phase 7, ending with runtime-constrained scene selection."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from uuid import uuid4

from sts.config import Settings
from sts.nlp.ner import build_characters
from sts.nlp.scorer_llm import build_llm
from sts.nlp.scorer_textrank import build_baseline
from sts.pipeline.ingest import ingest_zip
from sts.pipeline.render import render_job
from sts.pipeline.scenes import build_scenes
from sts.pipeline.selector import build_selection
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
    state = transcribe_job(job_dir)
    if stage == "transcribe":
        return state
    state = build_scenes(job_dir)
    if stage == "scenes":
        return state
    state = build_characters(job_dir)
    if stage == "characters":
        return state
    state = build_baseline(job_dir)
    if stage == "baseline":
        return state
    state = build_llm(job_dir)
    if stage == "llm":
        return state
    return build_selection(job_dir)


def _validate_stage(stage: str) -> None:
    if stage not in {"ingest", "transcribe", "scenes", "characters", "baseline", "llm", "select"}:
        raise ValueError(
            "Only --stage ingest, transcribe, scenes, characters, baseline, llm, or select "
            "is available"
        )


def run_job(
    zip_path: Path,
    target_minutes: float | None = None,
    *,
    name: str | None = None,
    jobs_dir: Path = Path("data/jobs"),
    cfg: Settings | None = None,
    stage: str = "select",
) -> JobState:
    """Create a job through scene selection, or stop at an earlier stage."""
    _validate_stage(stage)
    settings = cfg if cfg is not None else Settings()
    minutes = settings.target_minutes if target_minutes is None else target_minutes
    params = JobParams(
        target_seconds=minutes * 60.0,
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
        whisper_compute=settings.whisper_compute,
        detector_threshold=settings.detector_threshold,
        min_scene_seconds=settings.min_scene_seconds,
        max_characters=settings.max_characters,
        sbert_model=settings.sbert_model,
        min_clip_seconds=settings.min_clip_seconds,
        score_weights=settings.score_weights,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        ollama_base_url=settings.ollama_base_url,
        season_number=settings.season_number,
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
    job_id: str, *, jobs_dir: Path = Path("data/jobs"), stage: str = "select"
) -> JobState:
    """Resume through the requested stage, using checkpoints and persisted model settings."""
    _validate_stage(stage)
    return _run_stages(_job_dir(job_id, jobs_dir), stage)


def get_status(job_id: str, *, jobs_dir: Path = Path("data/jobs")) -> JobState:
    """Read a job's validated state without starting pipeline work."""
    return read_json(_job_dir(job_id, jobs_dir) / "state.json", JobState)


def render_saved_job(
    job_id: str, *, jobs_dir: Path = Path("data/jobs"), loudnorm: bool | None = None
) -> JobState:
    """Render an existing selection, reusing clip checkpoints and accepting edited clips."""
    return render_job(_job_dir(job_id, jobs_dir), loudnorm=loudnorm, force=True)


def list_jobs(*, jobs_dir: Path = Path("data/jobs")) -> list[JobState]:
    """List valid saved jobs by creation time, newest first."""
    if not jobs_dir.exists():
        return []
    states = [
        get_status(path.parent.name, jobs_dir=jobs_dir) for path in jobs_dir.glob("*/state.json")
    ]
    return sorted(states, key=lambda state: state.created_at, reverse=True)
