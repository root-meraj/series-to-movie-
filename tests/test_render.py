"""Uniform media commands, subtitle boundaries, and clip/publication crash recovery."""

from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from sts.app.cli import main
from sts.pipeline import render
from sts.pipeline.controller import get_status, render_saved_job
from sts.schemas import (
    EpisodeInfo,
    JobState,
    RenderClipDoc,
    SelectedClip,
    SelectionDoc,
    SelectionStats,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)


def _clip(episode: int, start: float = 10, end: float = 12) -> SelectedClip:
    return SelectedClip(
        scene_id=f"E{episode:02d}_s0001",
        episode=f"E{episode:02d}",
        source_path=Path(f"E{episode:02d}.mp4").resolve(),
        start=start,
        end=end,
    )


def _selection(clips: list[SelectedClip]) -> SelectionDoc:
    total = math.fsum(clip.end - clip.start for clip in clips)
    return SelectionDoc(
        target_seconds=max(10, total),
        selected_total_seconds=total,
        clips=clips,
        stats=SelectionStats(scenes_kept=len(clips), scenes_dropped=0, original_seconds=180),
    )


def _transcript(path: Path, episode: str, segments: list[tuple[float, float, str]]) -> None:
    write_json(
        path / f"{episode}.transcript.json",
        TranscriptDoc(
            episode=episode,
            language="en",
            segments=[
                TranscriptSegment(start=start, end=end, text=text) for start, end, text in segments
            ],
        ),
    )


def _probe_document(duration: float) -> dict[str, object]:
    return {
        "format": {"duration": str(duration)},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "pix_fmt": "yuv420p",
                "sample_aspect_ratio": "1:1",
                "r_frame_rate": "30/1",
                "time_base": "1/90000",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "time_base": "1/48000",
            },
        ],
    }


@pytest.fixture
def media_commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Simulate encoded files and FFprobe metadata without launching video processes."""
    calls: list[list[str]] = []

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[0] == "ffprobe":
            document = json.loads(Path(arguments[-1]).read_text(encoding="utf-8"))
            return subprocess.CompletedProcess(arguments, 0, json.dumps(document), "")
        if "-f" in arguments:
            manifest = Path(arguments[arguments.index("-i") + 1])
            paths = [
                Path(shlex.split(line)[1])
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.startswith("file ")
            ]
            duration = math.fsum(
                float(json.loads(path.read_text(encoding="utf-8"))["format"]["duration"])
                for path in paths
            )
        else:
            duration = float(arguments[arguments.index("-t") + 1])
        Path(arguments[-1]).write_text(json.dumps(_probe_document(duration)), encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(render.shutil, "which", lambda name: name)
    monkeypatch.setattr(render, "_run", run)
    return calls


def test_clip_reencodes_with_fast_seek_and_uniform_stream_parameters(
    tmp_path: Path,
    media_commands: list[list[str]],
) -> None:
    """Cuts must decode/re-encode, seek before input, and use duration rather than end time."""
    clip = _clip(1, start=10.125, end=12.75)
    clip.source_path.write_bytes(b"video")
    output = tmp_path / "encoded clip.mp4"
    assert render.render_clip(clip, output) == 2.625
    arguments = media_commands[0]
    assert arguments.index("-ss") < arguments.index("-i")
    assert float(arguments[arguments.index("-ss") + 1]) == 10.125
    assert float(arguments[arguments.index("-t") + 1]) == 2.625
    assert "-to" not in arguments and "copy" not in arguments
    for flag, expected in (
        ("-c:v", "libx264"),
        ("-preset", "veryfast"),
        ("-crf", "20"),
        ("-c:a", "aac"),
        ("-b:a", "192k"),
        ("-ar", "48000"),
        ("-ac", "2"),
    ):
        assert arguments[arguments.index(flag) + 1] == expected
    assert "pad=1280:720" in arguments[arguments.index("-vf") + 1]
    assert "fps=30" in arguments[arguments.index("-vf") + 1]
    assert output.is_file()
    assert not list(tmp_path.glob(".encoded clip.*.mp4"))


def test_optional_loudnorm_keeps_48khz_stereo_output(
    tmp_path: Path,
    media_commands: list[list[str]],
) -> None:
    """Loudness normalization is opt-in and cannot change the common audio format."""
    clip = _clip(1)
    clip.source_path.write_bytes(b"video")
    render.render_clip(clip, tmp_path / "clip.mp4", loudnorm=True)
    arguments = media_commands[0]
    assert arguments[arguments.index("-af") + 1] == "loudnorm=I=-16:TP=-1.5:LRA=11"
    assert arguments[arguments.index("-ar") + 1] == "48000"


def test_failed_encoding_preserves_previous_clip_and_cleans_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed FFmpeg process must never replace a playable clip with partial bytes."""
    clip = _clip(1)
    clip.source_path.write_bytes(b"source video")
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"previous complete clip")
    monkeypatch.setattr(render.shutil, "which", lambda name: name)

    def fail(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        Path(arguments[-1]).write_bytes(b"partial encode")
        raise RuntimeError("simulated FFmpeg failure")

    monkeypatch.setattr(render, "_run", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        render.render_clip(clip, output)
    assert output.read_bytes() == b"previous complete clip"
    assert not list(tmp_path.glob(".clip.*.mp4"))


def test_concat_rejects_nonuniform_streams_before_replacing_movie(
    tmp_path: Path,
    media_commands: list[list[str]],
) -> None:
    """A mismatched frame size cannot reach the stream-copy join."""
    path = tmp_path / "clip.mp4"
    document = _probe_document(2)
    document["streams"][0]["width"] = 640
    path.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "movie.mp4"
    output.write_bytes(b"previous movie")
    with pytest.raises(ValueError, match="uniform"):
        render.concat_clips([path], tmp_path / "clips.txt", output)
    assert output.read_bytes() == b"previous movie"
    assert not any(command[0] == "ffmpeg" for command in media_commands)


def test_concat_uses_only_uniform_clips_and_quotes_spaces_and_apostrophes(
    tmp_path: Path,
    media_commands: list[list[str]],
) -> None:
    """Stream copy is reserved for concatenation and the manifest follows FFmpeg quoting."""
    paths = [tmp_path / "clip one's.mp4", tmp_path / "clip two.mp4"]
    for path in paths:
        path.write_text(json.dumps(_probe_document(2.0)), encoding="utf-8")
    output = tmp_path / "movie.mp4"
    manifest = tmp_path / "clips.txt"
    assert render.concat_clips(paths, manifest, output) == 4.0
    command = next(arguments for arguments in media_commands if arguments[0] == "ffmpeg")
    assert command[command.index("-c") + 1] == "copy"
    assert command[command.index("-f") + 1] == "concat"
    assert command[command.index("-safe") + 1] == "0"
    assert "-copyts" in command
    assert "-ss" not in command and "-t" not in command
    assert "'\\''" in manifest.read_text(encoding="utf-8")
    assert output.is_file()


def test_subtitle_intersections_offsets_unicode_and_empty_text(tmp_path: Path) -> None:
    """Only overlapping dialogue is emitted, trimmed to clips and shifted onto movie time."""
    clips = [_clip(1), _clip(2, 20, 22.5)]
    transcripts = tmp_path / "transcripts"
    _transcript(
        transcripts,
        "E01",
        [(8, 9, "excluded"), (9, 10.5, "  before\ncut  "), (11, 13, "after cut"), (11, 11.5, " ")],
    )
    _transcript(transcripts, "E02", [(20.2, 20.7, "नमस्ते café"), (22.5, 23, "excluded")])
    output = tmp_path / "condensed.srt"
    render.export_srt(_selection(clips), transcripts, output)
    assert output.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:00,500\nbefore cut\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nafter cut\n\n"
        "3\n00:00:02,200 --> 00:00:02,700\nनमस्ते café\n\n"
    )


def test_subtitles_use_actual_clip_durations_and_follow_explicit_clip_order(tmp_path: Path) -> None:
    """Probed duration offsets prevent drift, including deliberately reordered selections."""
    transcripts = tmp_path / "transcripts"
    _transcript(transcripts, "E02", [(20, 21, "second episode first")])
    _transcript(transcripts, "E01", [(10, 11, "first episode second")])
    output = tmp_path / "out.srt"
    render.export_srt(
        _selection([_clip(2, 20, 22), _clip(1)]), transcripts, output, clip_durations=[2.034, 2]
    )
    assert "00:00:02,034 --> 00:00:03,034" in output.read_text(encoding="utf-8")


def test_subtitle_millisecond_rounding_carries_and_discards_zero_length_cues(
    tmp_path: Path,
) -> None:
    """Half milliseconds round up across second/minute/hour boundaries without zero cues."""
    clip = _clip(1, 0, 4000)
    transcripts = tmp_path / "transcripts"
    _transcript(
        transcripts,
        "E01",
        [
            (0.0001, 0.0002, "too short"),
            (59.9995, 60.1005, "minute"),
            (3599.9995, 3600.1005, "hour"),
        ],
    )
    output = tmp_path / "out.srt"
    render.export_srt(_selection([clip]), transcripts, output)
    text = output.read_text(encoding="utf-8")
    assert "too short" not in text
    assert "00:01:00,000 --> 00:01:00,101" in text
    assert "01:00:00,000 --> 01:00:00,101" in text
    assert text.startswith("1\n")


def test_silent_selection_exports_empty_srt(tmp_path: Path) -> None:
    """An empty transcription is valid and does not prevent a silent/action scene rendering."""
    _transcript(tmp_path, "E01", [])
    output = tmp_path / "out.srt"
    render.export_srt(_selection([_clip(1)]), tmp_path, output)
    assert output.read_bytes() == b""


@pytest.mark.parametrize("durations", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_subtitle_durations_fail_before_replacing_output(
    tmp_path: Path,
    durations: list[float],
) -> None:
    """The SRT timeline needs exactly one finite positive duration per clip."""
    output = tmp_path / "out.srt"
    output.write_text("previous subtitles", encoding="utf-8")
    with pytest.raises(ValueError):
        render.export_srt(_selection([_clip(1)]), tmp_path, output, clip_durations=durations)
    assert output.read_text(encoding="utf-8") == "previous subtitles"


@pytest.fixture
def selected_job(tmp_path: Path) -> Path:
    """Create a saved selection and transcripts with no models or upstream pipeline work."""
    directory = tmp_path / "jobs/renderfixture"
    clips = [_clip(1), _clip(2, 20, 22.5)]
    state = JobState(job_id=directory.name)
    for clip in clips:
        clip.source_path.write_bytes(clip.episode.encode())
        state.stages.ingest.episodes.append(
            EpisodeInfo(
                episode=clip.episode,
                episode_number=int(clip.episode[1:]),
                source_path=clip.source_path,
                original_name=clip.source_path.name,
                duration=90,
            )
        )
    state.stages.ingest.episode_count = 2
    for stage in (
        state.stages.ingest,
        state.stages.transcribe,
        state.stages.scenes,
        state.stages.nlp,
        state.stages.select,
    ):
        stage.status = "done"
    write_json(directory / "state.json", state)
    write_json(directory / "selected.json", _selection(clips))
    _transcript(directory / "transcripts", "E01", [(10.25, 11.75, "First clip")])
    _transcript(directory / "transcripts", "E02", [(20.2, 20.7, "Second clip")])
    return directory


def test_render_job_publishes_complete_outputs_and_skips_completed_stage(
    selected_job: Path,
    media_commands: list[list[str]],
) -> None:
    """A successful render checkpoints each clip and both final outputs; a repeat is a no-op."""
    state = render.render_job(selected_job)
    assert state.stages.render.status == "done"
    assert state.stages.render.completed == ["E01_s0001", "E02_s0001"]
    assert state.stages.render.pending == []
    assert state.stages.render.video_done and state.stages.render.subtitles_done
    assert (selected_job / "output/condensed_movie.mp4").is_file()
    assert "00:00:02,200" in (selected_job / "output/condensed.srt").read_text(encoding="utf-8")
    before = (selected_job / "state.json").read_bytes()
    media_commands.clear()
    assert render.render_job(selected_job) == state
    assert media_commands == []
    assert (selected_job / "state.json").read_bytes() == before


def test_clip_failure_resumes_from_last_completed_clip(
    selected_job: Path,
    media_commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-season render failure retains the first clip and resumes only the unfinished cut."""
    original = render.render_clip

    def fail_second(clip: SelectedClip, path: Path, *, loudnorm: bool = False) -> float:
        if clip.episode == "E02":
            raise RuntimeError("simulated encoder failure")
        return original(clip, path, loudnorm=loudnorm)

    monkeypatch.setattr(render, "render_clip", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        render.render_job(selected_job, loudnorm=True)
    failed = read_json(selected_job / "state.json", JobState)
    assert failed.stages.render.completed == ["E01_s0001"]
    assert failed.stages.render.pending == ["E02_s0001"]
    assert failed.errors[-1].stage == "render" and failed.errors[-1].episode == "E02"
    media_commands.clear()
    monkeypatch.setattr(render, "render_clip", original)
    state = render.render_job(selected_job)
    cuts = [command for command in media_commands if command[0] == "ffmpeg" and "-ss" in command]
    assert len(cuts) == 1 and "E02.mp4" in cuts[0][cuts[0].index("-i") + 1]
    assert "-af" in cuts[0] and state.stages.render.loudnorm
    assert state.errors == failed.errors


@pytest.mark.parametrize("boundary", ["cache", "concat", "srt", "done"])
def test_publication_failures_resume_and_reuse_encoded_clips(
    selected_job: Path,
    media_commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    """Each durable boundary supports recovery without re-encoding successful clip units."""
    persist = render.write_json
    concat = render.concat_clips
    subtitles = render.export_srt

    def fail_json(path: Path, document: BaseModel) -> None:
        if (
            boundary == "cache"
            and isinstance(document, JobState)
            and document.stages.render.completed == ["E01_s0001"]
        ) or (
            boundary == "done"
            and isinstance(document, JobState)
            and document.stages.render.status == "done"
        ):
            raise KeyboardInterrupt("simulated interruption")
        persist(path, document)

    def fail_concat(*args: object, **kwargs: object) -> float:
        raise RuntimeError("simulated concat failure")

    def fail_srt(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated subtitle failure")

    monkeypatch.setattr(render, "write_json", fail_json)
    if boundary == "concat":
        monkeypatch.setattr(render, "concat_clips", fail_concat)
    if boundary == "srt":
        monkeypatch.setattr(render, "export_srt", fail_srt)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        render.render_job(selected_job)
    media_commands.clear()
    monkeypatch.setattr(render, "write_json", persist)
    monkeypatch.setattr(render, "concat_clips", concat)
    monkeypatch.setattr(render, "export_srt", subtitles)
    assert render.render_job(selected_job).stages.render.status == "done"
    cuts = [command for command in media_commands if command[0] == "ffmpeg" and "-ss" in command]
    assert len(cuts) == (1 if boundary == "cache" else 0)
    joins = [command for command in media_commands if command[0] == "ffmpeg" and "-f" in command]
    assert len(joins) == (0 if boundary in {"srt", "done"} else 1)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "checkpoint", "source_changed"])
def test_only_invalid_clip_cache_is_reencoded(
    selected_job: Path,
    media_commands: list[list[str]],
    damage: str,
) -> None:
    """Corrupt encoded bytes, checkpoints, or changed source metadata cannot be reused."""
    state = render.render_job(selected_job)
    path = selected_job / "output/clips/clip_0002.mp4"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"broken video")
    elif damage == "checkpoint":
        path.with_suffix(".json").write_text("{", encoding="utf-8")
    else:
        state.stages.ingest.episodes[1].source_path.write_bytes(b"changed source")
    state.stages.render.status = "in_progress"
    write_json(selected_job / "state.json", state)
    media_commands.clear()
    assert render.render_job(selected_job).stages.render.status == "done"
    cuts = [command for command in media_commands if command[0] == "ffmpeg" and "-ss" in command]
    assert len(cuts) == 1 and "E02.mp4" in cuts[0][cuts[0].index("-i") + 1]


def test_interrupted_cache_repair_invalidates_previously_completed_outputs(
    selected_job: Path,
    media_commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repaired clip committed before a crash must invalidate the older movie and subtitles."""
    state = render.render_job(selected_job)
    state.stages.render.status = "in_progress"
    write_json(selected_job / "state.json", state)
    (selected_job / "output/clips/clip_0002.json").write_text("{", encoding="utf-8")
    persist = render.write_json

    def interrupt(path: Path, document: BaseModel) -> None:
        persist(path, document)
        if isinstance(document, RenderClipDoc) and document.scene_id == "E02_s0001":
            raise KeyboardInterrupt("simulated repair interruption")

    monkeypatch.setattr(render, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        render.render_job(selected_job)
    interrupted = read_json(selected_job / "state.json", JobState)
    assert not interrupted.stages.render.video_done
    assert not interrupted.stages.render.subtitles_done
    monkeypatch.setattr(render, "write_json", persist)
    media_commands.clear()
    assert render.render_job(selected_job).stages.render.status == "done"
    assert not any("-ss" in command for command in media_commands)
    assert sum(command[0] == "ffmpeg" and "-f" in command for command in media_commands) == 1


@pytest.mark.parametrize("output", ["condensed_movie.mp4", "condensed.srt"])
def test_damaged_final_outputs_repair_without_reencoding(
    selected_job: Path,
    media_commands: list[list[str]],
    output: str,
) -> None:
    """Resume checks final output hashes and repairs missing/corrupt artifacts independently."""
    state = render.render_job(selected_job)
    state.stages.render.status = "in_progress"
    write_json(selected_job / "state.json", state)
    (selected_job / "output" / output).write_bytes(b"corrupt output")
    media_commands.clear()
    assert render.render_job(selected_job).stages.render.status == "done"
    assert not any("-ss" in command for command in media_commands)


def test_rerender_accepts_edited_selection_and_reuses_unaffected_clip(
    selected_job: Path,
    media_commands: list[list[str]],
) -> None:
    """The explicit render command picks up manual edits without invoking upstream selection."""
    render.render_job(selected_job)
    selection = read_json(selected_job / "selected.json", SelectionDoc)
    selection.clips[1].end = 21.5
    selection = selection.model_copy(update={"selected_total_seconds": 3.5})
    write_json(selected_job / "selected.json", selection)
    media_commands.clear()
    state = render_saved_job(selected_job.name, jobs_dir=selected_job.parent)
    assert state.stages.render.status == "done"
    cuts = [command for command in media_commands if command[0] == "ffmpeg" and "-ss" in command]
    assert len(cuts) == 1
    assert float(cuts[0][cuts[0].index("-t") + 1]) == 1.5


def test_transcript_edit_refreshes_srt_without_reencoding(
    selected_job: Path,
    media_commands: list[list[str]],
) -> None:
    """Subtitle cache invalidation follows transcript bytes rather than selection alone."""
    render.render_job(selected_job)
    _transcript(selected_job / "transcripts", "E01", [(10.25, 11.75, "Updated dialogue")])
    media_commands.clear()
    render.render_job(selected_job, force=True)
    assert "Updated dialogue" in (selected_job / "output/condensed.srt").read_text(encoding="utf-8")
    assert not any("-ss" in command for command in media_commands)


@pytest.mark.parametrize(
    "damage", ["empty", "duplicate", "wrong_path", "past_end", "wrong_episode"]
)
def test_invalid_selection_is_recorded_before_encoding(
    selected_job: Path,
    media_commands: list[list[str]],
    damage: str,
) -> None:
    """An edited selection must remain nonempty and identify valid source intervals."""
    clips = read_json(selected_job / "selected.json", SelectionDoc).clips
    if damage == "empty":
        clips = []
    elif damage == "duplicate":
        clips.append(clips[0])
    elif damage == "wrong_path":
        clips[0].source_path = Path("unexpected.mp4")
    elif damage == "past_end":
        clips[0].end = 100
    else:
        clips[0].episode = "E03"
    write_json(selected_job / "selected.json", _selection(clips))
    with pytest.raises(ValueError):
        render.render_job(selected_job)
    state = read_json(selected_job / "state.json", JobState)
    assert state.stages.render.status == "failed"
    assert state.errors[-1].stage == "render"
    assert media_commands == []


def test_missing_tools_record_failures(
    selected_job: Path,
    media_commands: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup errors fail with state preserved and an actionable message."""
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="install FFmpeg"):
        render.render_job(selected_job)
    assert read_json(selected_job / "state.json", JobState).stages.render.status == "failed"


def test_missing_transcript_and_incomplete_selection_fail_before_encoding(
    selected_job: Path,
    media_commands: list[list[str]],
) -> None:
    """Required upstream artifacts fail early without starting expensive encodes."""
    (selected_job / "transcripts/E02.transcript.json").unlink()
    with pytest.raises(FileNotFoundError):
        render.render_job(selected_job)
    state = read_json(selected_job / "state.json", JobState)
    assert state.stages.render.status == "failed"
    assert media_commands == []
    state.stages.select.status = "pending"
    write_json(selected_job / "state.json", state)
    with pytest.raises(ValueError, match="completed scene selection"):
        render.render_job(selected_job)


def test_wrong_transcript_episode_preserves_previous_srt(tmp_path: Path) -> None:
    """A misplaced transcript cannot produce captions for a different episode."""
    _transcript(tmp_path, "E02", [(10, 11, "wrong episode")])
    (tmp_path / "E02.transcript.json").replace(tmp_path / "E01.transcript.json")
    output = tmp_path / "out.srt"
    output.write_text("previous captions", encoding="utf-8")
    with pytest.raises(ValueError, match="Transcript episode"):
        render.export_srt(_selection([_clip(1)]), tmp_path, output)
    assert output.read_text(encoding="utf-8") == "previous captions"


def test_cli_render_reports_outputs(
    selected_job: Path,
    media_commands: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The explicit render command reports movie/SRT paths and persists loudnorm."""
    assert (
        main(
            [
                "render",
                "--job",
                selected_job.name,
                "--jobs-dir",
                str(selected_job.parent),
                "--loudnorm",
            ]
        )
        == 0
    )
    assert "render done" in capsys.readouterr().out
    assert get_status(selected_job.name, jobs_dir=selected_job.parent).stages.render.loudnorm


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are required for the media smoke test",
)
def test_real_ffmpeg_mixed_shapes_fractional_cuts_and_playable_join(tmp_path: Path) -> None:
    """Real FFmpeg normalizes different inputs and the entire joined movie decodes cleanly."""
    sources: list[Path] = []
    for index, size in enumerate(("160x90", "90x120"), 1):
        path = tmp_path / f"episode {index}'s.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=duration=3:size={size}:rate={24 if index == 1 else 25}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration=3:sample_rate={44100 if index == 1 else 48000}",
                "-shortest",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ac",
                "1" if index == 1 else "2",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        sources.append(path)
    clips = [_clip(1, 0.25, 1.45), _clip(2, 0.5, 1.75)]
    paths: list[Path] = []
    durations: list[float] = []
    for index, (clip, source) in enumerate(zip(clips, sources, strict=True), 1):
        clip.source_path = source
        path = tmp_path / f"clip {index}'s.mp4"
        durations.append(render.render_clip(clip, path, loudnorm=index == 2))
        paths.append(path)
    output = tmp_path / "movie.mp4"
    duration = render.concat_clips(paths, tmp_path / "clips.txt", output)
    assert abs(duration - sum(durations)) < 0.05
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert all(
        abs(float(stream["start_time"])) < 0.001 for stream in json.loads(probe.stdout)["streams"]
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    _transcript(tmp_path, "E01", [(0.25, 1, "First source")])
    _transcript(tmp_path, "E02", [(0.5, 1, "Second source")])
    render.export_srt(_selection(clips), tmp_path, tmp_path / "movie.srt", clip_durations=durations)
    assert "First source" in (tmp_path / "movie.srt").read_text(encoding="utf-8")
    assert "Second source" in (tmp_path / "movie.srt").read_text(encoding="utf-8")
