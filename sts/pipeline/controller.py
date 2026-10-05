"""End-to-end job orchestration with stage boundaries, error recording, and resume."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
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
from sts.schemas import (
    ErrorRecord,
    JobParams,
    JobState,
    StageName,
    StageState,
    read_json,
    write_json,
)


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


def _stage_state(state: JobState, label: str) -> StageState:
    if label in {"characters", "baseline", "llm"}:
        return getattr(state.stages.nlp, label)
    return getattr(state.stages, label)


def _run_stage(
    job_dir: Path, label: str, stage_name: StageName, runner: Callable[[Path], JobState]
) -> JobState:
    state = read_json(job_dir / "state.json", JobState)
    if _stage_state(state, label).status == "done":
        logging.getLogger(__name__).info(
            "%s done: skipping completed stage for %s", label, state.job_id
        )
        return state
    previous_error_count = len(state.errors)
    try:
        return runner(job_dir)
    except Exception as exc:
        # Read the newest checkpoint: a runner may have completed units before
        # failing. Its own handler normally records the episode and exact error.
        state = read_json(job_dir / "state.json", JobState)
        failed = _stage_state(state, label)
        if failed.status != "failed" or len(state.errors) <= previous_error_count:
            failed.status = "failed"
            if stage_name == "nlp":
                state.stages.nlp.status = "failed"
            state.errors.append(ErrorRecord(stage=stage_name, message=str(exc)))
            write_json(job_dir / "state.json", state)
        logging.getLogger(__name__).error("Job %s failed during %s", state.job_id, label)
        raise


def _run_stages(job_dir: Path, stage: str) -> JobState:
    runners: tuple[tuple[str, StageName, Callable[[Path], JobState]], ...] = (
        ("ingest", "ingest", _run_ingest),
        ("transcribe", "transcribe", transcribe_job),
        ("scenes", "scenes", build_scenes),
        ("characters", "nlp", build_characters),
        ("baseline", "nlp", build_baseline),
        ("llm", "nlp", build_llm),
        ("select", "select", build_selection),
        ("render", "render", render_job),
    )
    for label, stage_name, runner in runners:
        state = _run_stage(job_dir, label, stage_name, runner)
        if stage == label:
            return state
    return state


def _validate_stage(stage: str) -> None:
    if stage not in {
        "ingest",
        "transcribe",
        "scenes",
        "characters",
        "baseline",
        "llm",
        "select",
        "render",
    }:
        raise ValueError(
            "Only --stage ingest, transcribe, scenes, characters, baseline, llm, select, "
            "or render is available"
        )


def create_job(
    zip_path: Path,
    target_minutes: float | None = None,
    *,
    name: str | None = None,
    jobs_dir: Path = Path("data/jobs"),
    cfg: Settings | None = None,
) -> JobState:
    """Persist a pending job and its parameters without loading models or starting stages."""
    settings = cfg if cfg is not None else Settings()
    minutes = settings.target_minutes if target_minutes is None else target_minutes
    params = JobParams(
        target_seconds=minutes * 60.0,
        whisper_model=settings.whisper_model,
        whisper_device=settings.whisper_device,
        whisper_compute=settings.whisper_compute,
        transcript_source=settings.transcript_source,
        subtitle_language=settings.subtitle_language,
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
    return state


def run_job(
    zip_path: Path,
    target_minutes: float | None = None,
    *,
    name: str | None = None,
    jobs_dir: Path = Path("data/jobs"),
    cfg: Settings | None = None,
    stage: str = "render",
) -> JobState:
    """Create and complete a ZIP-to-movie job, or stop at an explicit earlier stage."""
    _validate_stage(stage)
    state = create_job(zip_path, target_minutes, name=name, jobs_dir=jobs_dir, cfg=cfg)
    return _run_stages(_job_dir(state.job_id, jobs_dir), stage)


def resume_job(
    job_id: str, *, jobs_dir: Path = Path("data/jobs"), stage: str = "render"
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
