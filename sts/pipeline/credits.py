"""Conservative end-credit ranges from chapters or a few final-minute video frames."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sts.pipeline.control import check_stop, run_media
from sts.schemas import CreditRange, EpisodeCreditsDoc, TranscriptDoc

FrameKind = Literal["credits", "blank", "other"]


def _check(job_dir: Path | None) -> None:
    if job_dir is not None:
        check_stop(job_dir)


def _run(arguments: list[str], job_dir: Path | None) -> subprocess.CompletedProcess[str]:
    if job_dir is not None:
        return run_media(arguments, job_dir)
    return subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _chapters(video_path: Path, job_dir: Path | None) -> list[dict[str, object]]:
    executable = shutil.which("ffprobe")
    if executable is None:
        return []
    result = _run(
        [executable, "-v", "error", "-show_chapters", "-of", "json", str(video_path)], job_dir
    )
    metadata = json.loads(result.stdout)
    if not isinstance(metadata, dict):
        return []
    chapters = metadata.get("chapters", [])
    if not isinstance(chapters, list):
        return []
    return [chapter for chapter in chapters if isinstance(chapter, dict)]


def _spoken_segments(transcript: TranscriptDoc) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for segment in transcript.segments:
        # Musical lyrics and standalone sound descriptions do not prove a story scene.
        # Bracketed speaker labels followed by words still count as spoken dialogue.
        text = re.sub(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>", " ", segment.text).strip()
        if text and "♪" not in text and "♫" not in text and re.search(r"\w", text):
            intervals.append((segment.start, segment.end))
    return intervals


def _protect_dialogue(
    start: float, end: float, spoken: list[tuple[float, float]], evidence: str
) -> CreditRange | None:
    # A post-credit conversation may be followed by silent action. Preserve its entire
    # remaining chapter/tail, rather than keeping only subtitle-sized fragments.
    overlapping = [begin for begin, finish in spoken if begin < end and finish > start]
    if overlapping:
        end = min(end, max(0.0, min(overlapping) - 3.0))
    if end - start < 5.0:
        return None
    return CreditRange(start=float(start), end=float(end), evidence=evidence)


def _metadata_ranges(
    chapters: list[dict[str, object]], duration: float, spoken: list[tuple[float, float]]
) -> list[CreditRange]:
    ranges: list[CreditRange] = []
    for chapter in chapters:
        tags = chapter.get("tags", {})
        if not isinstance(tags, dict):
            continue
        title = str(tags.get("title", ""))
        if not re.search(r"\b(?:end\s+|closing\s+)?credits\b", title, re.IGNORECASE):
            continue
        try:
            start = float(str(chapter["start_time"]))
            end = min(duration, float(str(chapter["end_time"])))
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or start < duration * 0.7:
            continue
        if duration - end <= 2.0:
            end = duration
        result = _protect_dialogue(start, end, spoken, f"chapter: {title}")
        if result is not None:
            ranges.append(result)
    merged: list[CreditRange] = []
    for item in sorted(ranges, key=lambda item: item.start):
        if merged and item.start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, item.end)
        else:
            merged.append(item)
    return merged


def _sample_frame(
    video_path: Path, timestamp: float, destination: Path, job_dir: Path | None
) -> NDArray[np.uint8] | None:
    import cv2

    executable = shutil.which("ffmpeg")
    if executable is None:
        return None
    _run(
        [
            executable,
            "-v",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2",
            "-threads",
            "1",
            str(destination),
        ],
        job_dir,
    )
    return cv2.imread(str(destination), cv2.IMREAD_GRAYSCALE)


def _frame_kind(frame: NDArray[np.uint8] | None) -> FrameKind:
    import cv2

    if frame is None or frame.ndim != 2 or frame.size == 0:
        return "other"
    if float(np.mean(frame < 30)) < 0.88:
        return "other"
    mask = (frame > max(30.0, float(np.median(frame)) + 25.0)).astype(np.uint8)
    occupied = float(mask.mean())
    if occupied < 0.0008:
        return "blank"
    if occupied > 0.13:
        return "other"
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = frame.shape
    glyphs = [
        row
        for row in stats[1:count]
        if 2 <= row[cv2.CC_STAT_HEIGHT] <= height * 0.085
        and 1 <= row[cv2.CC_STAT_WIDTH] <= width * 0.15
        and 3 <= row[cv2.CC_STAT_AREA] <= height * width * 0.005
    ]
    # Serif kerning can connect two or more letters into a single component.
    if len(glyphs) < 3 or sum(int(row[cv2.CC_STAT_WIDTH]) for row in glyphs) < width * 0.06:
        return "other"
    glyph_pixels = sum(int(row[cv2.CC_STAT_AREA]) for row in glyphs)
    if glyph_pixels < int(mask.sum()) * 0.72:
        return "other"
    # Glyphs need horizontal text rows. This rejects scattered stars/lights on a
    # dark story frame without reading names or downloading an OCR model.
    rows: dict[int, int] = {}
    for row in glyphs:
        key = int((row[cv2.CC_STAT_TOP] + row[cv2.CC_STAT_HEIGHT] / 2) // 8)
        rows[key] = rows.get(key, 0) + 1
    return "credits" if max(rows.values(), default=0) >= 3 else "other"


def _visual_ranges(
    video_path: Path, duration: float, spoken: list[tuple[float, float]], job_dir: Path | None
) -> list[CreditRange]:
    tail_start = max(0.0, duration - min(600.0, max(180.0, duration * 0.2)))
    times = [float(value) for value in np.arange(tail_start, max(tail_start, duration - 1), 10)]
    times.append(max(0.0, duration - 0.5))
    runs: list[list[tuple[float, FrameKind, str]]] = []
    current: list[tuple[float, FrameKind, str]] = []
    with TemporaryDirectory(prefix="sts-credit-frames-") as temporary:
        destination = Path(temporary) / "frame.png"
        for timestamp in times:
            _check(job_dir)
            frame = _sample_frame(video_path, timestamp, destination, job_dir)
            kind = _frame_kind(frame)
            if kind == "other" or any(
                begin < timestamp + 5 and end > timestamp - 5 for begin, end in spoken
            ):
                if current:
                    runs.append(current)
                current = []
                continue
            signature = hashlib.sha256(frame.tobytes()).hexdigest() if frame is not None else ""
            current.append((timestamp, kind, signature))
        if current:
            runs.append(current)
        ranges: list[CreditRange] = []
        for run in runs:
            text_frames = [item for item in run if item[1] == "credits"]
            if len(text_frames) < 3 or len({item[2] for item in text_frames}) < 3:
                continue
            if text_frames[-1][0] - text_frames[0][0] < 20:
                continue
            first = text_frames[0][0]
            # Refine only an evidenced credit transition, never the last subtitle.
            lower = max(tail_start, first - 10.0)
            upper = first
            for _ in range(4):
                _check(job_dir)
                midpoint = (lower + upper) / 2
                frame = _sample_frame(video_path, midpoint, destination, job_dir)
                if _frame_kind(frame) == "credits":
                    upper = midpoint
                else:
                    lower = midpoint
            start = upper
            last = run[-1][0]
            # A following story frame is preserved with a full sample-interval margin.
            end = duration if math.isclose(last, times[-1], abs_tol=1e-6) else last
            result = _protect_dialogue(start, end, spoken, "consecutive dark text credit frames")
            if result is not None:
                ranges.append(result)
        return ranges


def detect_credits(
    video_path: Path, transcript: TranscriptDoc, duration: float, *, job_dir: Path | None = None
) -> EpisodeCreditsDoc:
    """Find evidenced final credits, preserving quiet endings and post-credit dialogue.

    Named final credit chapters are authoritative. Otherwise, only changing text-like
    frames on dark backgrounds, repeated for at least twenty seconds near the end,
    establish a credit range. Unknown/unsupported material stays in the story. This
    reads no more than one small frame per ten seconds of the last ten minutes.
    """
    result = EpisodeCreditsDoc(episode=transcript.episode, duration=duration)
    logger = logging.getLogger(__name__)
    _check(job_dir)
    logger.info("credits started: %s", transcript.episode)
    spoken = _spoken_segments(transcript)
    try:
        ranges = _metadata_ranges(_chapters(video_path, job_dir), duration, spoken)
        if not ranges:
            ranges = _visual_ranges(video_path, duration, spoken, job_dir)
        result.ranges = ranges
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        logger.warning(
            "credits unit-progress: %s unknown (%s)", transcript.episode, type(exc).__name__
        )
    logger.info("credits done: %s, %d evidenced ranges", transcript.episode, len(result.ranges))
    return result
