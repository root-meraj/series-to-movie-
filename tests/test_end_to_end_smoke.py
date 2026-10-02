"""Opt-in ZIP-to-movie smoke with real local models and strictly mocked LLM calls."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

from scripts.make_fixture import make_fixture
from sts.app.cli import main
from sts.schemas import (
    JobState,
    RenderClipDoc,
    ScenesDoc,
    ScoresDoc,
    SelectionDoc,
    TranscriptDoc,
    read_json,
)


def _srt_seconds(timecode: str) -> float:
    hours, minutes, seconds, milliseconds = map(int, re.split("[:,]", timecode))
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _assert_subtitles(directory: Path, selection: SelectionDoc, duration: float) -> None:
    content = (directory / "output/condensed.srt").read_text(encoding="utf-8")
    blocks = content.strip().split("\n\n") if content.strip() else []
    actual_texts: list[str] = []
    previous_start = 0.0
    for index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        assert lines[0] == str(index)
        match = re.fullmatch(
            r"(\d{2,}:\d{2}:\d{2},\d{3}) --> (\d{2,}:\d{2}:\d{2},\d{3})", lines[1]
        )
        assert match is not None, block
        start, end = (_srt_seconds(value) for value in match.groups())
        assert previous_start <= start < end <= duration + 0.001
        previous_start = start
        actual_texts.append(" ".join(lines[2:]))
    expected_texts: list[str] = []
    for index, clip in enumerate(selection.clips, 1):
        transcript = read_json(
            directory / "transcripts" / f"{clip.episode}.transcript.json", TranscriptDoc
        )
        checkpoint = read_json(directory / "output/clips" / f"clip_{index:04d}.json", RenderClipDoc)
        for segment in sorted(transcript.segments, key=lambda item: (item.start, item.end)):
            text = " ".join(segment.text.split())
            start = max(clip.start, segment.start)
            end = min(clip.end, segment.end, clip.start + checkpoint.duration)
            # An interval smaller than a millisecond may round to no SRT cue.
            if text and end - start >= 0.001:
                expected_texts.append(text)
    assert actual_texts == expected_texts


def _artifact_snapshot(directory: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(directory)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.integration
def test_real_zip_to_condensed_movie_and_completed_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Produce a fully decoded two-minute movie and matching SRT, then skip finished work.

    This invokes the CLI in-process so the shared external-LLM transport guards
    remain active. Only local Whisper/spaCy/SBERT models may be loaded. Synthetic
    tone often transcribes to no dialogue, in which case an empty SRT is valid.
    The generated job remains under project data/jobs for manual inspection.
    """
    project = Path(__file__).resolve().parents[1]
    fixture = make_fixture(project / "tests/fixtures/end-to-end", scene_cuts=True)
    jobs_dir = project / "data/jobs"
    existing = {path.name for path in jobs_dir.iterdir()} if jobs_dir.exists() else set()
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE", "int8")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    assert main(
        [
            "run",
            "--zip",
            str(fixture),
            "--target-minutes",
            "2",
            "--name",
            "Phase 9 end-to-end smoke",
            "--jobs-dir",
            str(jobs_dir),
        ]
    ) == 0
    created = [path for path in jobs_dir.iterdir() if path.name not in existing]
    assert len(created) == 1
    directory = created[0]
    state = read_json(directory / "state.json", JobState)
    assert state.params.llm_provider == "mock"
    assert state.params.target_seconds == 120.0
    assert state.errors == []
    for name in ("ingest", "transcribe", "scenes", "nlp", "select", "render"):
        assert getattr(state.stages, name).status == "done", name
    for name in ("characters", "baseline", "llm"):
        assert getattr(state.stages.nlp, name).status == "done", name
    assert state.stages.ingest.completed == ["E01", "E02", "E03"]
    assert state.stages.render.pending == []
    assert state.stages.render.video_done and state.stages.render.subtitles_done
    scenes = read_json(directory / "scenes.json", ScenesDoc).root
    episode_bounds: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for scene in scenes:
        episode_bounds[scene.episode].append((scene.start, scene.end))
    assert len(scenes) == 9
    for bounds in episode_bounds.values():
        for actual, expected in zip(
            bounds, [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)], strict=True
        ):
            assert actual == pytest.approx(expected, abs=1 / 30)
    scores = read_json(directory / "scores.json", ScoresDoc).root
    assert {score.scene_id for score in scores} == {scene.scene_id for scene in scenes}
    assert all(score.llm_score is not None for score in scores)
    selection = read_json(directory / "selected.json", SelectionDoc)
    assert 0 < selection.selected_total_seconds <= 120.0
    assert selection.clips[0].scene_id == scenes[0].scene_id
    assert selection.clips[-1].scene_id == scenes[-1].scene_id
    movie = directory / "output/condensed_movie.mp4"
    assert movie.is_file() and (directory / "output/condensed.srt").is_file()
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    assert ffprobe is not None and ffmpeg is not None
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(movie)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    media = json.loads(probe.stdout)
    video = next(stream for stream in media["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in media["streams"] if stream["codec_type"] == "audio")
    assert (video["codec_name"], video["width"], video["height"], video["r_frame_rate"]) == (
        "h264", 1280, 720, "30/1"
    )
    assert (audio["codec_name"], audio["sample_rate"], audio["channels"]) == ("aac", "48000", 2)
    duration = float(media["format"]["duration"])
    assert duration == pytest.approx(selection.selected_total_seconds, abs=0.15)
    subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-xerror", "-i", str(movie), "-f", "null", "-"],
        check=True,
        capture_output=True,
        timeout=180,
    )
    _assert_subtitles(directory, selection, duration)
    before = _artifact_snapshot(directory)
    assert main(["resume", "--job", state.job_id, "--jobs-dir", str(jobs_dir)]) == 0
    assert _artifact_snapshot(directory) == before
