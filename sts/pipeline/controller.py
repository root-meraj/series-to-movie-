"""End-to-end job orchestration with stage boundaries, error recording, and resume."""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from sts.config import Settings
from sts.nlp.ner import build_characters
from sts.nlp.scorer_llm import build_llm
from sts.nlp.scorer_textrank import build_baseline
from sts.pipeline.control import check_stop
from sts.pipeline.ingest import ingest_zip
from sts.pipeline.render import render_job
from sts.pipeline.scenes import build_scenes
from sts.pipeline.selector import build_selection
from sts.pipeline.transcribe import transcribe_job
from sts.schemas import (
    EpisodeInfo,
    ErrorRecord,
    JobParams,
    JobState,
    RequiredRange,
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
    state = read_json(job_dir / "state.json", JobState)
    for label, stage_name, runner in runners:
        if _stage_state(state, label).status != "done":
            check_stop(job_dir)
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
        scene_method=settings.scene_method,
        scene_block_seconds=settings.scene_block_seconds,
        exclude_credits=settings.exclude_credits,
        selection_strategy=settings.selection_strategy,
        must_keep_ranges=settings.must_keep_ranges,
        llm_batch_size=settings.llm_batch_size,
        render_encoder=settings.render_encoder,
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


def create_improved_job(
    job_id: str,
    *,
    jobs_dir: Path = Path("data/jobs"),
    must_keep_ranges: list[RequiredRange] | None = None,
    target_minutes: float | None = None,
) -> JobState:
    """Create a fresh edit from saved media/subtitles, preserving the earlier movie.

    Local hard links reuse media without copying a season or depending on the old
    job directory. Filesystems without hard links fall back to ordinary copies.
    Scene windows and NLP are rebuilt with the current story settings; completed
    dialogue extraction and valid character-mention caches are reused.
    """
    source_dir = _job_dir(job_id, jobs_dir)
    previous = read_json(source_dir / "state.json", JobState)
    if (
        previous.stages.transcribe.status != "done"
        or previous.stages.ingest.status != "done"
        or not previous.stages.ingest.episodes
    ):
        raise ValueError("Complete episode import and dialogue capture before refining an edit")
    requested_ranges = (
        must_keep_ranges if must_keep_ranges is not None else previous.params.must_keep_ranges
    )
    durations = {item.episode: item.duration for item in previous.stages.ingest.episodes}
    for interval in requested_ranges:
        if interval.episode not in durations or interval.end > durations[interval.episode]:
            raise ValueError("A required moment must lie within a known episode")
    for item in previous.stages.ingest.episodes:
        for required in (
            item.source_path,
            source_dir / f"audio/{item.episode}.wav",
            source_dir / f"audio/{item.episode}.rms.json",
            source_dir / f"transcripts/{item.episode}.transcript.json",
        ):
            if not required.is_file():
                raise FileNotFoundError(f"Saved episode artifact missing: {required.name}")
    cfg = Settings(
        transcript_source=previous.params.transcript_source,
        subtitle_language=previous.params.subtitle_language,
        whisper_model=previous.params.whisper_model,
        whisper_device=previous.params.whisper_device,
        whisper_compute=previous.params.whisper_compute,
        llm_provider=previous.params.llm_provider or Settings().llm_provider,
        llm_model=previous.params.llm_model or Settings().llm_model,
        ollama_base_url=previous.params.ollama_base_url or Settings().ollama_base_url,
        season_number=previous.params.season_number,
        scene_method="subtitles",
        selection_strategy="story",
        exclude_credits=True,
        must_keep_ranges=requested_ranges,
        llm_batch_size=8,
        render_encoder="auto",
    )
    state = create_job(
        previous.params.zip_path or source_dir / "input/source.zip",
        target_minutes if target_minutes is not None else previous.params.target_seconds / 60,
        name=f"{previous.params.name or 'Untitled season'} · refined",
        jobs_dir=jobs_dir,
        cfg=cfg,
    )
    directory = _job_dir(state.job_id, jobs_dir)
    state.params.parent_job_id = job_id
    state.params.zip_sha256 = previous.params.zip_sha256
    manifest: list[EpisodeInfo] = []
    for item in previous.stages.ingest.episodes:
        destination = directory / "episodes" / item.source_path.name
        try:
            os.link(item.source_path, destination)
        except OSError:
            shutil.copy2(item.source_path, destination)
        manifest.append(item.model_copy(update={"source_path": destination.resolve()}))
        for relative in (
            f"audio/{item.episode}.wav",
            f"audio/{item.episode}.rms.json",
            f"transcripts/{item.episode}.transcript.json",
            f"transcripts/{item.episode}.characters.json",
            f"transcripts/{item.episode}.credits.json",
        ):
            origin, target = source_dir / relative, directory / relative
            if not origin.is_file():
                if relative.endswith((".wav", ".rms.json", ".transcript.json")):
                    raise FileNotFoundError(f"Saved dialogue artifact missing: {origin.name}")
                continue
            if origin.suffix == ".wav":
                try:
                    os.link(origin, target)
                    continue
                except OSError:
                    pass
            shutil.copy2(origin, target)
    state.stages.ingest = previous.stages.ingest.model_copy(deep=True)
    state.stages.ingest.episodes = manifest
    state.stages.transcribe = previous.stages.transcribe.model_copy(deep=True)
    write_json(directory / "state.json", state)
    logging.getLogger(__name__).info(
        "Refined edit %s reuses %s subtitle capture", state.job_id, job_id
    )
    return state


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
