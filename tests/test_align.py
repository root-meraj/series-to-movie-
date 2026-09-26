"""Midpoint ownership, crossing dialogue, silence, timing unions, and weighted RMS."""

from pathlib import Path

import pytest

from sts.pipeline.align import align
from sts.schemas import SceneRecord, TranscriptSegment


def _scenes(bounds: list[tuple[float, float]]) -> list[SceneRecord]:
    return [
        SceneRecord(
            scene_id=f"E01_s{index:04d}",
            episode="E01",
            source_path=Path("E01.mp4"),
            start=start,
            end=end,
            duration=end - start,
            episode_number=1,
            is_first=index == 1,
            is_last=index == len(bounds),
        )
        for index, (start, end) in enumerate(bounds, start=1)
    ]


def test_midpoint_assignment_and_silent_scenes() -> None:
    """A contained segment enriches its own scene while silent scenes and metadata survive."""
    scenes = _scenes([(0, 10), (10, 20), (20, 30)])
    segments = [TranscriptSegment(start=12, end=16, text="We need to talk.")]
    aligned = align(scenes, segments, [0.2] * 10 + [0.8] * 10 + [0.1] * 10)
    assert [scene.transcript for scene in aligned] == ["", "We need to talk.", ""]
    assert [scene.dialogue_ratio for scene in aligned] == [0.0, 0.4, 0.0]
    assert [scene.audio_rms for scene in aligned] == pytest.approx([0.2, 0.8, 0.1])
    assert aligned[0].is_first and aligned[-1].is_last
    assert [scene.scene_id for scene in aligned] == [scene.scene_id for scene in scenes]
    assert all(scene.transcript == "" for scene in scenes)  # No input mutation.


@pytest.mark.parametrize(
    ("start", "end", "owner", "ratios"),
    [
        (8.0, 13.0, 1, [0.2, 0.3]),
        (7.0, 12.0, 0, [0.3, 0.2]),
    ],
)
def test_crossing_cut_uses_larger_overlap(
    start: float, end: float, owner: int, ratios: list[float]
) -> None:
    """Text goes to the larger overlap, but speech time is measured in both scenes."""
    result = align(
        _scenes([(0, 10), (10, 20)]),
        [TranscriptSegment(start=start, end=end, text="Follow the money.")],
        [],
    )
    assert result[owner].transcript == "Follow the money."
    assert result[1 - owner].transcript == ""
    assert [scene.dialogue_ratio for scene in result] == ratios


def test_equal_cut_overlap_splits_text_without_loss_or_duplication() -> None:
    """A symmetric crossing splits its words proportionally and keeps chronological order."""
    result = align(
        _scenes([(0, 10), (10, 20)]),
        [TranscriptSegment(start=8, end=12, text="one two three four five six")],
        [],
    )
    assert [scene.transcript for scene in result] == ["one two three", "four five six"]
    assert [scene.dialogue_ratio for scene in result] == [0.2, 0.2]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("one two three", ["one two", "three"]),
        ("word", ["word", ""]),
    ],
)
def test_tie_rounding_preserves_whole_words(text: str, expected: list[str]) -> None:
    """Odd word counts deterministically round the split toward the earlier scene."""
    result = align(
        _scenes([(0, 10), (10, 20)]), [TranscriptSegment(start=9, end=11, text=text)], []
    )
    assert [scene.transcript for scene in result] == expected


def test_segment_spanning_three_scenes_prefers_largest_overlap_over_midpoint() -> None:
    """For long segments the crossing rule takes precedence over the midpoint shortcut."""
    result = align(
        _scenes([(0, 10), (10, 12), (12, 20)]),
        [TranscriptSegment(start=4, end=16, text="A long spoken thought.")],
        [],
    )
    assert [scene.transcript for scene in result] == ["A long spoken thought.", "", ""]
    assert [scene.dialogue_ratio for scene in result] == [0.6, 1.0, 0.5]


def test_multiway_tie_splits_proportionally() -> None:
    """Equal overlap with three scenes divides a segment into three word ranges."""
    result = align(
        _scenes([(0, 2), (2, 4), (4, 6)]),
        [TranscriptSegment(start=0, end=6, text="one two three four five six")],
        [],
    )
    assert [scene.transcript for scene in result] == ["one two", "three four", "five six"]


def test_dialogue_ratio_unions_overlapping_segments() -> None:
    """Overlapping speech does not double-count time or push ratios above one."""
    result = align(
        _scenes([(0, 10)]),
        [
            TranscriptSegment(start=4, end=9, text="Second"),
            TranscriptSegment(start=1, end=6, text="First"),
            TranscriptSegment(start=2, end=3, text="Nested"),
        ],
        [],
    )
    assert result[0].dialogue_ratio == 0.8
    assert result[0].transcript == "First Nested Second"


def test_exact_boundaries_do_not_leak_into_adjacent_scene() -> None:
    """A segment ending at a cut belongs only to the scene before that cut."""
    result = align(
        _scenes([(0, 10), (10, 20)]),
        [
            TranscriptSegment(start=8, end=10, text="Left"),
            TranscriptSegment(start=10, end=12, text="Right"),
            TranscriptSegment(start=20, end=22, text="Outside"),
        ],
        [],
    )
    assert [scene.transcript for scene in result] == ["Left", "Right"]
    assert [scene.dialogue_ratio for scene in result] == [0.2, 0.2]


def test_partial_second_rms_is_time_weighted() -> None:
    """Fractional scene boundaries weight RMS bins by their duration of overlap."""
    result = align(_scenes([(0.75, 2.25)]), [], [0.2, 0.8, 0.4])
    assert result[0].audio_rms == pytest.approx((0.25 * 0.2 + 0.8 + 0.25 * 0.4) / 1.5)


def test_missing_audio_bins_and_blank_segments_are_silent() -> None:
    """Unavailable audio contributes zero, and empty transcript text is not speech."""
    result = align(_scenes([(0, 2)]), [TranscriptSegment(start=0, end=2, text="  ")], [0.4])
    assert result[0].audio_rms == 0.2
    assert result[0].dialogue_ratio == 0.0
    assert align([], [], []) == []


@pytest.mark.parametrize("bounds", [[(0, 10), (9, 20)], [(10, 20), (0, 10)]])
def test_overlapping_or_unsorted_scenes_are_rejected(bounds: list[tuple[float, float]]) -> None:
    """Ambiguous scene geometry is rejected before assigning dialogue."""
    with pytest.raises(ValueError, match="chronological and nonoverlapping"):
        align(_scenes(bounds), [], [])


@pytest.mark.parametrize("rms", [[-1.0], [float("nan")], [float("inf")]])
def test_invalid_rms_is_rejected(rms: list[float]) -> None:
    """Feature corruption cannot silently enter a scene record."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        align(_scenes([(0, 1)]), [], rms)
