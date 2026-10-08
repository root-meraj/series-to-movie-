"""Subtitle story windows, optional visual cuts, and resumable scene alignment."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from sts.config import Settings
from sts.pipeline.align import align
from sts.pipeline.control import StopCheck, check_stop
from sts.pipeline.credits import detect_credits
from sts.schemas import (
    AudioFeaturesDoc,
    EpisodeCreditsDoc,
    EpisodeInfo,
    ErrorRecord,
    JobState,
    SceneRecord,
    ScenesDoc,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from scenedetect import FrameTimecode


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

    detector = ContentDetector(threshold=cfg.detector_threshold, min_scene_len=1)
    job_dir = video_path.parent.parent
    if (job_dir / "processing_control.json").is_file():
        stop = StopCheck(job_dir)
        process_frame = detector.process_frame

        def checked_frame(timecode: FrameTimecode, frame: NDArray[np.uint8]) -> list[FrameTimecode]:
            stop()
            return process_frame(timecode, frame)

        detector.process_frame = checked_frame
    detected = detect(
        video_path,
        detector,
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


def subtitle_scene_bounds(
    segments: list[TranscriptSegment], duration: float, block_seconds: float = 60.0
) -> list[tuple[float, float]]:
    """Group timed dialogue into contiguous story windows without decoding video.

    Prefer gaps near the requested length, normally keeping windows between half
    and one-and-a-half times that length. Overlapping cues are indivisible, so a
    very long cue can exceed that soft maximum. Silence is retained and divided
    into ordinary windows: a missing subtitle never proves that action is absent.
    The result covers the whole episode, including its opening and final frame.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Subtitle scene duration must be finite and positive")
    if not math.isfinite(block_seconds) or block_seconds <= 0:
        raise ValueError("Subtitle scene block_seconds must be finite and positive")

    # Ignore blank cues; clamp the others to the source runtime. A boundary may
    # touch a cue endpoint, but it must never lie inside any overlapping cue.
    spoken: list[tuple[float, float]] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        start, end = min(segment.start, duration), min(segment.end, duration)
        if not segment.text.strip() or start >= end:
            continue
        if spoken and start < spoken[-1][1]:
            spoken[-1] = (spoken[-1][0], max(spoken[-1][1], end))
        else:
            spoken.append((start, end))
    gaps: list[tuple[float, float]] = []
    previous_end = 0.0
    for start, end in spoken:
        gaps.append((previous_end, start))
        previous_end = end
    gaps.append((previous_end, duration))

    bounds: list[tuple[float, float]] = []
    start = 0.0
    minimum = block_seconds * 0.5
    maximum = block_seconds * 1.5
    while duration - start > maximum:
        desired = start + block_seconds
        lower = start + minimum
        upper = min(start + maximum, duration - minimum)
        candidates: list[tuple[float, float]] = []
        for gap_start, gap_end in gaps:
            if gap_end < lower or gap_start > upper:
                continue
            allowed_start, allowed_end = max(gap_start, lower), min(gap_end, upper)
            if gap_end - gap_start >= minimum:
                # Long silent intervals should form their own action candidates,
                # rather than dragging a whole gap into one dialogue window.
                cut = min(max(desired, allowed_start), allowed_end)
            else:
                cut = min(max((gap_start + gap_end) / 2.0, allowed_start), allowed_end)
            gap_bonus = min(gap_end - gap_start, block_seconds * 0.25) * 0.5
            candidates.append((abs(cut - desired) - gap_bonus, cut))
        if candidates:
            end = min(candidates)[1]
        else:
            # Dense, overlapping subtitles can cross the entire preferred range.
            # Keep the cue intact and end at the first safe point after it.
            end = next(
                (max(gap_start, lower) for gap_start, gap_end in gaps if gap_end >= lower),
                duration,
            )
            if duration - end < minimum:
                end = duration
        bounds.append((start, end))
        start = end
    if start < duration:
        bounds.append((start, duration))
    _validate_bounds(bounds)
    return bounds


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


def _credit_bounds(
    bounds: list[tuple[float, float]], document: EpisodeCreditsDoc
) -> list[tuple[float, float, bool]]:
    points = {point for interval in document.ranges for point in (interval.start, interval.end)}
    split: list[tuple[float, float, bool]] = []
    for start, end in bounds:
        edges = [start, *sorted(point for point in points if start < point < end), end]
        for left, right in zip(edges, edges[1:], strict=False):
            midpoint = (left + right) / 2
            credit = any(interval.start <= midpoint < interval.end for interval in document.ranges)
            split.append((left, right, credit))
    return split


def build_scenes(job_dir: Path) -> JobState:
    """Build and align episode windows, checkpointing scenes.json before state."""
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
            scene_method=state.params.scene_method,
            scene_block_seconds=state.params.scene_block_seconds,
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
            check_stop(job_dir)
            transcript = read_json(
                job_dir / "transcripts" / f"{episode}.transcript.json", TranscriptDoc
            )
            features = read_json(job_dir / "audio" / f"{episode}.rms.json", AudioFeaturesDoc)
            if transcript.episode != episode or features.episode != episode:
                raise ValueError(f"Transcript and RMS episode identifiers must match {episode}")
            if cfg.scene_method == "subtitles":
                bounds = subtitle_scene_bounds(
                    transcript.segments, item.duration, cfg.scene_block_seconds
                )
                logger.info(
                    "scenes unit-progress: %s built %d subtitle story windows without video scan",
                    episode,
                    len(bounds),
                )
            else:
                logger.info("scenes unit-progress: %s detecting cuts", episode)
                bounds = detect_scenes(item.source_path, cfg)
            credits_path = job_dir / "transcripts" / f"{episode}.credits.json"
            credits = EpisodeCreditsDoc(episode=episode, duration=item.duration)
            if state.params.exclude_credits:
                try:
                    credits = read_json(credits_path, EpisodeCreditsDoc)
                    if credits.episode != episode or not math.isclose(
                        credits.duration, item.duration
                    ):
                        raise ValueError("Credit checkpoint must match the episode")
                except (OSError, UnicodeError, ValidationError, ValueError):
                    credits = detect_credits(
                        item.source_path, transcript, item.duration, job_dir=job_dir
                    )
                    write_json(credits_path, credits)
            ranged = _credit_bounds(bounds, credits)
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
                    is_last=episode == planned[-1] and index == len(ranged),
                    is_credit=is_credit,
                )
                for index, (start, end, is_credit) in enumerate(ranged, start=1)
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
