"""Content-cut detection, minimum-duration merging, and resumable scene alignment."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from pydantic import ValidationError

from sts.config import Settings
from sts.pipeline.align import align
from sts.schemas import (
    AudioFeaturesDoc,
    EpisodeInfo,
    ErrorRecord,
    JobState,
    SceneRecord,
    ScenesDoc,
    TranscriptDoc,
    read_json,
    write_json,
)


def _validate_bounds(bounds: list[tuple[float, float]]) -> None:
    if not bounds:
        raise ValueError("Scene detection returned no decoded video frames")
    if not math.isclose(bounds[0][0], 0.0, abs_tol=1e-6):
        raise ValueError("Scene boundaries must cover the video from zero")
    for index, (start, end) in enumerate(bounds):
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("Scene boundaries must be finite positive-length intervals")
        if index and not math.isclose(start, bounds[index - 1][1], rel_tol=0, abs_tol=1e-6):
            raise ValueError("Scene boundaries must be contiguous and chronological")


def detect_scenes(video_path: Path, cfg: Settings) -> list[tuple[float, float]]:
    """Detect cuts as float seconds, retaining uncut video and merging short scenes.

    Short scenes are accumulated into their next neighbor. A short final remainder
    merges backward. If the whole video is below the minimum, keep it as one scene.
    """
    from scenedetect import ContentDetector, detect

    detected = detect(
        video_path,
        ContentDetector(threshold=cfg.detector_threshold, min_scene_len=1),
        start_in_scene=True,
    )
    bounds = [(float(start.get_seconds()), float(end.get_seconds())) for start, end in detected]
    _validate_bounds(bounds)
    merged: list[tuple[float, float]] = []
    pending_start = bounds[0][0]
    for _, end in bounds:
        if end - pending_start >= cfg.min_scene_seconds:
            merged.append((pending_start, end))
            pending_start = end
    final_end = bounds[-1][1]
    if pending_start < final_end:
        if merged:
            merged[-1] = (merged[-1][0], final_end)
        else:
            merged.append((bounds[0][0], final_end))
    logging.getLogger(__name__).info(
        "scenes unit-progress: %s detected %d, merged to %d scenes",
        video_path.name,
        len(bounds),
        len(merged),
    )
    return merged


def _cached_episode_valid(records: list[SceneRecord], episode: EpisodeInfo) -> bool:
    if not records:
        return False
    try:
        _validate_bounds([(record.start, record.end) for record in records])
    except ValueError:
        return False
    return all(
        record.scene_id == f"{episode.episode}_s{index:04d}"
        and record.source_path == episode.source_path
        and record.episode_number == episode.episode_number
        for index, record in enumerate(records, start=1)
    )


def build_scenes(job_dir: Path) -> JobState:
    """Detect and align one episode at a time, checkpointing scenes.json before state."""
    logger = logging.getLogger(__name__)
    state_path = job_dir / "state.json"
    scenes_path = job_dir / "scenes.json"
    state = read_json(state_path, JobState)
    stage = state.stages.scenes
    if stage.status == "done":
        logger.info("scenes done: skipping completed stage for %s", state.job_id)
        return state
    episode: str | None = None
    try:
        stage.status = "in_progress"
        write_json(state_path, state)
        logger.info("scenes started: %s", state.job_id)
        if state.stages.transcribe.status != "done" or not state.stages.ingest.episodes:
            raise ValueError(
                "Scene alignment requires completed transcription and an episode manifest"
            )
        cfg = Settings(
            detector_threshold=state.params.detector_threshold,
            min_scene_seconds=state.params.min_scene_seconds,
        )
        episodes = sorted(state.stages.ingest.episodes, key=lambda item: item.episode_number)
        planned = [item.episode for item in episodes]
        try:
            previous = read_json(scenes_path, ScenesDoc).root
        except (OSError, UnicodeError, ValidationError):
            previous = []
        records: list[SceneRecord] = []
        verified: list[str] = []
        for item in episodes:
            cached = [record for record in previous if record.episode == item.episode]
            if item.episode in stage.completed and _cached_episode_valid(cached, item):
                records.extend(cached)
                verified.append(item.episode)
        stage.completed = verified
        stage.pending = [identifier for identifier in planned if identifier not in verified]
        write_json(scenes_path, ScenesDoc(records))
        write_json(state_path, state)

        for item in episodes:
            episode = item.episode
            if episode in stage.completed:
                logger.info("scenes unit-progress: %s already complete", episode)
                continue
            transcript = read_json(
                job_dir / "transcripts" / f"{episode}.transcript.json", TranscriptDoc
            )
            features = read_json(job_dir / "audio" / f"{episode}.rms.json", AudioFeaturesDoc)
            if transcript.episode != episode or features.episode != episode:
                raise ValueError(f"Transcript and RMS episode identifiers must match {episode}")
            logger.info("scenes unit-progress: %s detecting cuts", episode)
            bounds = detect_scenes(item.source_path, cfg)
            scene_records = [
                SceneRecord(
                    scene_id=f"{episode}_s{index:04d}",
                    episode=episode,
                    source_path=item.source_path,
                    start=start,
                    end=end,
                    duration=end - start,
                    episode_number=item.episode_number,
                    is_first=episode == planned[0] and index == 1,
                    is_last=episode == planned[-1] and index == len(bounds),
                )
                for index, (start, end) in enumerate(bounds, start=1)
            ]
            records.extend(align(scene_records, transcript.segments, features.rms_per_second))
            records.sort(key=lambda record: (record.episode_number, record.start))
            write_json(scenes_path, ScenesDoc(records))
            completed = set(stage.completed) | {episode}
            stage.completed = [identifier for identifier in planned if identifier in completed]
            stage.pending = [identifier for identifier in planned if identifier not in completed]
            write_json(state_path, state)
            logger.info(
                "scenes unit-progress: %s done (%d/%d)", episode, len(stage.completed), len(planned)
            )
        stage.status = "done"
        write_json(state_path, state)
        logger.info("scenes done: %d scenes across %d episodes", len(records), len(episodes))
        return state
    except Exception as exc:
        stage.status = "failed"
        state.errors.append(ErrorRecord(stage="scenes", episode=episode, message=str(exc)))
        write_json(state_path, state)
        logger.exception("scenes failed: %s", state.job_id)
        raise
