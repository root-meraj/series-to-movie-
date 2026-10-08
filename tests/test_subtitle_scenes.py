"""Fast subtitle windows preserve dialogue, silence, and resumable episode work."""

from __future__ import annotations

from pathlib import Path

import pytest

from sts.pipeline import control, scenes
from sts.schemas import (
    AudioFeaturesDoc,
    EpisodeInfo,
    JobParams,
    JobState,
    ScenesDoc,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)


def _cue(start: float, end: float, text: str = "Dialogue") -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _assert_full_coverage(bounds: list[tuple[float, float]], duration: float) -> None:
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == duration
    assert sum(end - start for start, end in bounds) == pytest.approx(duration)
    assert all(end > start for start, end in bounds)
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:], strict=False))


@pytest.mark.parametrize(
    ("duration", "block", "expected"),
    [
        (20.0, 60.0, [(0.0, 20.0)]),
        (90.0, 60.0, [(0.0, 90.0)]),
        (200.0, 60.0, [(0.0, 60.0), (60.0, 120.0), (120.0, 200.0)]),
        (120.0, 30.0, [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 120.0)]),
    ],
)
def test_empty_subtitles_keep_deterministic_silent_windows(
    duration: float, block: float, expected: list[tuple[float, float]]
) -> None:
    """A missing transcript still supplies full action coverage without a video scan."""
    result = scenes.subtitle_scene_bounds([], duration, block)
    assert result == expected
    _assert_full_coverage(result, duration)


def test_dialogue_gaps_group_conversations_near_the_requested_length() -> None:
    """Cuts favor useful gaps between cues, retaining one coherent dialogue window."""
    segments = [_cue(2, 20), _cue(25, 52), _cue(58, 70), _cue(75, 110), _cue(120, 140)]
    bounds = scenes.subtitle_scene_bounds(segments, 240)
    assert bounds[0] == (0.0, 55.0)
    assert bounds[1] == (55.0, 115.0)
    assert all(30.0 <= end - start <= 90.0 for start, end in bounds)
    _assert_full_coverage(bounds, 240.0)


@pytest.mark.parametrize(
    "segments",
    [
        [_cue(45, 75)],
        [_cue(45, 75), _cue(60, 85)],
        [_cue(index, index + 12) for index in range(0, 240, 12)],
        [_cue(10, 170)],
        [_cue(10, 230)],
        [_cue(0, 240)],
    ],
)
def test_no_boundary_splits_a_subtitle_even_when_cues_overlap(
    segments: list[TranscriptSegment],
) -> None:
    """Cue integrity overrides the preferred maximum for a very long subtitle."""
    bounds = scenes.subtitle_scene_bounds(segments, 240)
    _assert_full_coverage(bounds, 240.0)
    for _, cut in bounds[:-1]:
        assert all(not segment.start < cut < segment.end for segment in segments)


def test_long_silent_action_spans_remain_separate_candidates() -> None:
    """Subtitles at either end must not drop the intervening action or create one giant clip."""
    bounds = scenes.subtitle_scene_bounds([_cue(0, 20), _cue(220, 240)], 300)
    assert (60.0, 120.0) in bounds
    assert (120.0, 180.0) in bounds
    assert max(end - start for start, end in bounds) <= 90.0
    _assert_full_coverage(bounds, 300.0)


def test_long_final_cue_does_not_leave_an_unselectable_tiny_tail() -> None:
    """An unusually long final subtitle absorbs a small source remainder intact."""
    assert scenes.subtitle_scene_bounds([_cue(10, 230)], 240) == [(0.0, 240.0)]


def test_subtitle_order_blank_cues_and_source_clamping() -> None:
    """Input order and out-of-source/blank subtitles cannot corrupt chronological bounds."""
    segments = [_cue(170, 260), _cue(30, 80, "  "), _cue(0, 10), _cue(260, 300)]
    bounds = scenes.subtitle_scene_bounds(segments, 240)
    assert bounds == scenes.subtitle_scene_bounds(list(reversed(segments)), 240)
    assert bounds[0] == (0.0, 60.0)
    _assert_full_coverage(bounds, 240.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_subtitle_window_configuration_is_rejected(value: float) -> None:
    """Nonpositive/nonfinite bounds fail before any scene artifact is written."""
    with pytest.raises(ValueError, match="duration"):
        scenes.subtitle_scene_bounds([], value)
    with pytest.raises(ValueError, match="block_seconds"):
        scenes.subtitle_scene_bounds([], 240, value)


@pytest.fixture
def subtitle_job(tmp_path: Path) -> Path:
    """Create two subtitle/RMS episodes without videos, FFprobe, or trained models."""
    directory = tmp_path / "subtitle-job"
    state = JobState(
        job_id="subtitle-job",
        params=JobParams(scene_method="subtitles", scene_block_seconds=60.0),
    )
    manifest: list[EpisodeInfo] = []
    for index in range(1, 3):
        episode = f"E{index:02d}"
        manifest.append(
            EpisodeInfo(
                episode=episode,
                episode_number=index,
                source_path=directory / "episodes" / f"{episode}.mkv",
                original_name=f"show.S01{episode}.mkv",
                duration=180.0,
            )
        )
        write_json(
            directory / "transcripts" / f"{episode}.transcript.json",
            TranscriptDoc(
                episode=episode,
                language="en",
                source="embedded_subtitles",
                segments=[_cue(2, 20, "A plan."), _cue(125, 130, "The plan succeeds.")],
            ),
        )
        write_json(
            directory / "audio" / f"{episode}.rms.json",
            AudioFeaturesDoc(episode=episode, duration=180, rms_per_second=[0.25] * 180),
        )
    state.stages.ingest.status = "done"
    state.stages.ingest.episodes = manifest
    state.stages.ingest.episode_count = 2
    state.stages.ingest.completed = [item.episode for item in manifest]
    state.stages.transcribe.status = "done"
    state.stages.transcribe.completed = [item.episode for item in manifest]
    write_json(directory / "state.json", state)
    return directory


def test_subtitle_stage_skips_frame_detection_and_aligns_all_story_windows(
    subtitle_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fast path uses only saved timestamps and preserves silent scene features."""

    def fail_detection(*args: object, **kwargs: object) -> None:
        raise AssertionError("Subtitle scene mode must never decode video frames")

    monkeypatch.setattr(scenes, "detect_scenes", fail_detection)
    monkeypatch.setenv("SCENE_METHOD", "visual")
    monkeypatch.setenv("SCENE_BLOCK_SECONDS", "10")
    completed = scenes.build_scenes(subtitle_job)
    records = read_json(subtitle_job / "scenes.json", ScenesDoc).root
    assert completed.stages.scenes.status == "done"
    assert completed.stages.scenes.completed == ["E01", "E02"]
    assert len(records) == 6
    assert [record.scene_id for record in records] == [
        f"E{episode:02d}_s{index:04d}" for episode in range(1, 3) for index in range(1, 4)
    ]
    assert sum(record.is_first for record in records) == 1
    assert sum(record.is_last for record in records) == 1
    assert records[1].transcript == ""
    assert records[1].dialogue_ratio == 0.0
    assert all(record.audio_rms == 0.25 for record in records)
    assert " ".join(record.transcript for record in records if record.transcript) == (
        "A plan. The plan succeeds. A plan. The plan succeeds."
    )


def test_subtitle_resume_reuses_completed_episode_with_saved_window_length(
    subtitle_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later episode failure preserves the first window checkpoint and saved method."""
    original = scenes.subtitle_scene_bounds
    calls: list[float] = []

    def interrupted(
        segments: list[TranscriptSegment], duration: float, block_seconds: float = 60.0
    ) -> list[tuple[float, float]]:
        calls.append(block_seconds)
        if len(calls) == 2:
            raise RuntimeError("Interrupted subtitle construction")
        return original(segments, duration, block_seconds)

    monkeypatch.setattr(scenes, "subtitle_scene_bounds", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted subtitle construction"):
        scenes.build_scenes(subtitle_job)
    previous = read_json(subtitle_job / "scenes.json", ScenesDoc).root
    assert len(previous) == 3
    failed = read_json(subtitle_job / "state.json", JobState)
    assert failed.stages.scenes.completed == ["E01"]
    assert failed.errors[-1].episode == "E02"
    monkeypatch.setenv("SCENE_METHOD", "visual")
    monkeypatch.setenv("SCENE_BLOCK_SECONDS", "20")
    completed = scenes.build_scenes(subtitle_job)
    assert completed.stages.scenes.status == "done"
    assert read_json(subtitle_job / "scenes.json", ScenesDoc).root[:3] == previous
    assert calls == [60.0, 60.0, 60.0]


def test_legacy_job_without_saved_scene_method_keeps_visual_detection(
    subtitle_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing application defaults does not rebuild legacy episodes with another method."""
    import json

    state_path = subtitle_job / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["params"].pop("scene_method")
    payload["params"].pop("scene_block_seconds")
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[str] = []

    def visual(path: Path, cfg: object) -> list[tuple[float, float]]:
        calls.append(path.stem)
        return [(0.0, 90.0), (90.0, 180.0)]

    monkeypatch.setattr(scenes, "detect_scenes", visual)
    monkeypatch.setenv("SCENE_METHOD", "subtitles")
    completed = scenes.build_scenes(subtitle_job)
    assert completed.stages.scenes.status == "done"
    assert calls == ["E01", "E02"]
    assert len(read_json(subtitle_job / "scenes.json", ScenesDoc).root) == 4


def test_subtitle_stop_preserves_completed_episode_and_resumes_only_pending_windows(
    subtitle_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop during fast scene construction retains progress without creating a failure."""
    original = scenes.subtitle_scene_bounds
    calls: list[float] = []

    def stopped(
        segments: list[TranscriptSegment], duration: float, block_seconds: float = 60.0
    ) -> list[tuple[float, float]]:
        calls.append(block_seconds)
        if len(calls) == 2:
            control.set_stop_requested(subtitle_job, True)
            control.check_stop(subtitle_job)
        return original(segments, duration, block_seconds)

    monkeypatch.setattr(scenes, "subtitle_scene_bounds", stopped)
    with pytest.raises(control.ProcessingStopped):
        scenes.build_scenes(subtitle_job)
    previous = read_json(subtitle_job / "scenes.json", ScenesDoc).root
    paused = read_json(subtitle_job / "state.json", JobState)
    assert paused.stages.scenes.status == "in_progress"
    assert paused.stages.scenes.completed == ["E01"]
    assert paused.errors == []
    control.set_stop_requested(subtitle_job, False)
    completed = scenes.build_scenes(subtitle_job)
    assert completed.stages.scenes.status == "done"
    assert read_json(subtitle_job / "scenes.json", ScenesDoc).root[:3] == previous
    assert calls == [60.0, 60.0, 60.0]
