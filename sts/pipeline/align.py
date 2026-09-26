"""Align timestamped dialogue and per-second audio energy with one episode's scenes."""

from __future__ import annotations

import math
from bisect import bisect_right
from itertools import pairwise

from sts.schemas import SceneRecord, TranscriptSegment


def _speech_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _mean_rms(start: float, end: float, values: list[float]) -> float:
    weighted_sum = 0.0
    for second in range(math.floor(start), min(math.ceil(end), len(values))):
        overlap = min(end, second + 1.0) - max(start, float(second))
        weighted_sum += values[second] * overlap
    return weighted_sum / (end - start)


def align(
    scenes: list[SceneRecord],
    transcript_segments: list[TranscriptSegment],
    rms_per_second: list[float],
) -> list[SceneRecord]:
    """Return enriched scene records, preserving silent scenes and input metadata.

    A contained segment belongs to its midpoint's scene. A segment crossing a cut
    goes to the scene with greatest overlap. Ties split its words proportionally,
    rounding split points to the nearest word (half toward the earlier scene).
    Dialogue ratios measure the union of actual speech overlaps in each scene,
    independently of text assignment. RMS means are weighted by time overlap;
    missing audio bins contribute zero. Inputs must describe one ordered episode.
    """
    if any(not math.isfinite(value) or value < 0 for value in rms_per_second):
        raise ValueError("RMS values must be finite and nonnegative")
    if not scenes:
        return []
    if len({scene.episode for scene in scenes}) != 1:
        raise ValueError("Align one episode at a time")
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("Scene IDs must be unique")
    if any(right.start < left.end for left, right in pairwise(scenes)):
        raise ValueError("Scenes must be chronological and nonoverlapping")

    text_parts: list[list[str]] = [[] for _ in scenes]
    speech_intervals: list[list[tuple[float, float]]] = [[] for _ in scenes]
    ends = [scene.end for scene in scenes]
    for segment in sorted(transcript_segments, key=lambda item: (item.start, item.end)):
        words = segment.text.split()
        if not words:
            continue
        overlaps: list[tuple[int, float]] = []
        index = bisect_right(ends, segment.start)
        while index < len(scenes) and scenes[index].start < segment.end:
            scene = scenes[index]
            start, end = max(scene.start, segment.start), min(scene.end, segment.end)
            if end > start:
                overlaps.append((index, end - start))
                speech_intervals[index].append((start, end))
            index += 1
        if not overlaps:
            continue
        largest = max(overlap for _, overlap in overlaps)
        owners = [
            (index, overlap)
            for index, overlap in overlaps
            if math.isclose(overlap, largest, rel_tol=0.0, abs_tol=1e-9)
        ]
        total_overlap = sum(overlap for _, overlap in owners)
        elapsed = 0.0
        word_start = 0
        for owner_number, (index, overlap) in enumerate(owners):
            elapsed += overlap
            word_end = (
                len(words)
                if owner_number == len(owners) - 1
                else math.floor(len(words) * elapsed / total_overlap + 0.5)
            )
            if word_end > word_start:
                text_parts[index].append(" ".join(words[word_start:word_end]))
            word_start = word_end

    return [
        SceneRecord.model_validate(
            {
                **scene.model_dump(),
                "transcript": " ".join(text_parts[index]),
                "dialogue_ratio": min(
                    1.0, _speech_duration(speech_intervals[index]) / scene.duration
                ),
                "audio_rms": _mean_rms(scene.start, scene.end, rms_per_second),
            }
        )
        for index, scene in enumerate(scenes)
    ]
