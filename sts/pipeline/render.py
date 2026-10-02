"""Uniform clip encoding, resumable concatenation, and aligned subtitle export."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import ValidationError

from sts.schemas import (
    ErrorRecord,
    JobState,
    RenderClipDoc,
    SelectedClip,
    SelectionDoc,
    TranscriptDoc,
    read_json,
    write_json,
)

_VIDEO_FILTER = (
    "scale=trunc(ih*dar/2)*2:ih,setsar=1,"
    "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,"
    "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p"
)
_RENDER_VERSION = "720p30-stereo-v1"


def _tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"{name} is required; install FFmpeg and add its bin directory to PATH")
    return executable


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{Path(arguments[0]).stem} failed: {(exc.stderr or '')[-4000:]}"
        ) from exc


@contextmanager
def _temporary_output(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _write_text(path: Path, content: str) -> None:
    with _temporary_output(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _probe_encoded(path: Path, expected_duration: float | None = None) -> float:
    result = _run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    document: dict[str, Any] = json.loads(result.stdout)
    streams = document["streams"]
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(streams) != 2 or len(video) != 1 or len(audio) != 1:
        raise ValueError("Rendered media requires exactly one video and one audio stream")
    v, a = video[0], audio[0]
    if (
        v.get("codec_name") != "h264"
        or v.get("width") != 1280
        or v.get("height") != 720
        or v.get("pix_fmt") != "yuv420p"
        or v.get("sample_aspect_ratio") != "1:1"
        or Fraction(v.get("r_frame_rate", "0")) != 30
        or v.get("time_base") != "1/90000"
        or a.get("codec_name") != "aac"
        or int(a.get("sample_rate", 0)) != 48000
        or a.get("channels") != 2
        or a.get("time_base") != "1/48000"
    ):
        raise ValueError("Rendered streams do not match the uniform encoding contract")
    duration = float(document["format"]["duration"])
    if (
        not math.isfinite(duration)
        or duration <= 0
        or (expected_duration is not None and abs(duration - expected_duration) > 0.15)
    ):
        raise ValueError("Rendered duration does not match the requested clip timeline")
    return duration


def render_clip(clip: SelectedClip, out_path: Path, *, loudnorm: bool = False) -> float:
    """Re-encode one interval to uniform H.264/AAC MP4 and return its probed duration.

    All clips use 1280x720 letterboxing, square pixels, 30 fps, yuv420p, a 90 kHz
    video time base, and 48 kHz stereo AAC. The destination changes only after
    encoding and metadata validation succeed. Both source streams are required.
    """
    if not clip.source_path.is_file():
        raise FileNotFoundError(clip.source_path)
    duration = clip.end - clip.start
    with _temporary_output(out_path) as temporary:
        arguments = [
            _tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{clip.start:.9f}",
            "-i",
            str(clip.source_path),
            "-t",
            f"{duration:.9f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            _VIDEO_FILTER,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-profile:v",
            "high",
            "-level:v",
            "3.1",
            "-video_track_timescale",
            "90000",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
        ]
        if loudnorm:
            arguments.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
        arguments.extend(
            [
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )
        _run(arguments)
        actual_duration = _probe_encoded(temporary, duration)
        temporary.replace(out_path)
    return actual_duration


def concat_clips(clip_paths: list[Path], manifest_path: Path, out_path: Path) -> float:
    """Join already uniform clips with the concat demuxer and stream copy, atomically.

    File paths are quoted using FFmpeg's manifest syntax. No trimming occurs at
    this step. Return the probed container duration for the assembled movie.
    """
    if not clip_paths:
        raise ValueError("Cannot render an empty selection")
    lines = ["ffconcat version 1.0"]
    durations: list[float] = []
    for path in clip_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        value = path.resolve().as_posix()
        if any(character in value for character in ("\n", "\r", "\0")):
            raise ValueError("Concat file paths cannot contain newline or NUL characters")
        quoted = value.replace("'", "'\\''")
        lines.append(f"file '{quoted}'")
        durations.append(_probe_encoded(path))
    _write_text(manifest_path, "\n".join(lines) + "\n")
    with _temporary_output(out_path) as temporary:
        _run(
            [
                _tool("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-copyts",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "disabled",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )
        duration = _probe_encoded(temporary, math.fsum(durations))
        temporary.replace(out_path)
    return duration


def _milliseconds(seconds: float) -> int:
    return int((Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _timecode(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def export_srt(
    selection: SelectionDoc,
    transcripts_dir: Path,
    out_path: Path,
    *,
    clip_durations: list[float] | None = None,
) -> None:
    """Write clipped transcript cues with cumulative output offsets, in pure Python.

    Supplying probed clip durations matches the concat demuxer's actual timeline,
    avoiding accumulated frame-rounding drift. Without them, source interval
    lengths define offsets. Empty transcripts produce a valid empty SRT file.
    """
    durations = (
        clip_durations
        if clip_durations is not None
        else [clip.end - clip.start for clip in selection.clips]
    )
    if len(durations) != len(selection.clips) or any(
        not math.isfinite(duration) or duration <= 0 for duration in durations
    ):
        raise ValueError("Provide one finite, positive output duration per clip")
    document: TranscriptDoc | None = None
    cues: list[str] = []
    offset = 0.0
    for clip, duration in zip(selection.clips, durations, strict=True):
        if document is None or document.episode != clip.episode:
            document = read_json(transcripts_dir / f"{clip.episode}.transcript.json", TranscriptDoc)
            if document.episode != clip.episode:
                raise ValueError("Transcript episode must match its selected clip")
        for segment in sorted(document.segments, key=lambda item: (item.start, item.end)):
            text = " ".join(segment.text.split())
            start = max(clip.start, segment.start)
            end = min(clip.end, segment.end, clip.start + duration)
            if not text or end <= start:
                continue
            begin_ms = _milliseconds(offset + start - clip.start)
            end_ms = _milliseconds(offset + end - clip.start)
            if end_ms > begin_ms:
                cues.append(
                    f"{len(cues) + 1}\n{_timecode(begin_ms)} --> {_timecode(end_ms)}\n{text}\n\n"
                )
        offset = math.fsum([offset, duration])
    _write_text(out_path, "".join(cues))


def _clip_hash(clip: SelectedClip, loudnorm: bool) -> str:
    stat = clip.source_path.stat()
    return _hash_payload(
        {
            "clip": clip.model_dump(mode="json"),
            "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "loudnorm": loudnorm,
            "renderer": _RENDER_VERSION,
        }
    )


def _cached_clip(path: Path, clip: SelectedClip, fingerprint: str) -> RenderClipDoc | None:
    try:
        document = read_json(path.with_suffix(".json"), RenderClipDoc)
        if (
            document.scene_id != clip.scene_id
            or document.input_sha256 != fingerprint
            or _sha256(path) != document.output_sha256
        ):
            return None
        _probe_encoded(path, document.duration)
        return document
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ValidationError):
        return None


def _output_valid(path: Path, checksum: str | None) -> bool:
    try:
        return checksum is not None and _sha256(path) == checksum
    except OSError:
        return False


def _validate_clips(selection: SelectionDoc, state: JobState) -> None:
    if not selection.clips:
        raise ValueError("Cannot render an empty selection")
    if len({clip.scene_id for clip in selection.clips}) != len(selection.clips):
        raise ValueError("Selected clips must have unique scene IDs")
    episodes = {episode.episode: episode for episode in state.stages.ingest.episodes}
    for clip in selection.clips:
        source = episodes.get(clip.episode)
        if source is None or not clip.scene_id.startswith(f"{clip.episode}_"):
            raise ValueError("Selected clip must belong to a known episode")
        if clip.source_path.resolve() != source.source_path.resolve():
            raise ValueError("Selected source_path must match the ingestion manifest")
        if clip.end > source.duration + 0.001:
            raise ValueError("Selected clip exceeds its source episode duration")


def render_job(job_dir: Path, *, loudnorm: bool | None = None, force: bool = False) -> JobState:
    """Resume clips, concatenate a movie, and export SRT from an existing selected.json.

    Completed stages skip unless force=True (the explicit re-render command).
    Re-rendering validates edited selection and reuses unaffected clip caches.
    A changed selection, source file metadata, or transcript invalidates output
    checkpoints. Failed units retain the last complete encoded clips and movie.
    """
    logger = logging.getLogger(__name__)
    state_path = job_dir / "state.json"
    state = read_json(state_path, JobState)
    stage = state.stages.render
    if stage.status == "done" and not force:
        logger.info("render done: skipping completed stage for %s", state.job_id)
        return state
    episode: str | None = None
    try:
        was_done = stage.status == "done"
        stage.status = "in_progress"
        if loudnorm is not None:
            stage.loudnorm = loudnorm
        write_json(state_path, state)
        logger.info("render started: %s", state.job_id)
        if state.stages.select.status != "done":
            raise ValueError("Rendering requires completed scene selection")
        _tool("ffmpeg")
        _tool("ffprobe")
        selection = read_json(job_dir / "selected.json", SelectionDoc)
        _validate_clips(selection, state)
        fingerprints = [_clip_hash(clip, stage.loudnorm) for clip in selection.clips]
        fingerprint = _hash_payload(
            {
                "concat_version": "copyts-v1",
                "selection": selection.model_dump(mode="json"),
                "clips": fingerprints,
                "transcripts": [
                    _sha256(job_dir / "transcripts" / f"{identifier}.transcript.json")
                    for identifier in dict.fromkeys(clip.episode for clip in selection.clips)
                ],
            }
        )
        if fingerprint != stage.input_sha256 or was_done:
            stage.video_done = False
            stage.subtitles_done = False
        stage.input_sha256 = fingerprint
        stage.completed = []
        stage.pending = [clip.scene_id for clip in selection.clips]
        write_json(state_path, state)
        output_dir = job_dir / "output"
        paths: list[Path] = []
        durations: list[float] = []
        repaired = False
        for index, (clip, clip_fingerprint) in enumerate(
            zip(selection.clips, fingerprints, strict=True), 1
        ):
            episode = clip.episode
            path = output_dir / "clips" / f"clip_{index:04d}.mp4"
            checkpoint = _cached_clip(path, clip, clip_fingerprint)
            if checkpoint is None:
                # Persist invalidation before replacing a clip. A crash after
                # its cache commit must not reuse outputs built from older clips.
                stage.video_done = False
                stage.subtitles_done = False
                write_json(state_path, state)
                logger.info(
                    "render unit-progress: encoding %s (%d/%d)",
                    clip.scene_id,
                    index,
                    len(selection.clips),
                )
                duration = render_clip(clip, path, loudnorm=stage.loudnorm)
                checkpoint = RenderClipDoc(
                    scene_id=clip.scene_id,
                    input_sha256=clip_fingerprint,
                    output_sha256=_sha256(path),
                    duration=duration,
                )
                write_json(path.with_suffix(".json"), checkpoint)
                repaired = True
            paths.append(path)
            durations.append(checkpoint.duration)
            stage.completed.append(clip.scene_id)
            stage.pending.remove(clip.scene_id)
            write_json(state_path, state)
            logger.info(
                "render unit-progress: %s clip ready (%d/%d)",
                clip.scene_id,
                index,
                len(selection.clips),
            )
        episode = None
        movie_path = output_dir / "condensed_movie.mp4"
        if repaired or not stage.video_done or not _output_valid(movie_path, stage.video_sha256):
            stage.video_done = False
            write_json(state_path, state)
            concat_clips(paths, output_dir / "clips.txt", movie_path)
            stage.video_done = True
            stage.video_sha256 = _sha256(movie_path)
            write_json(state_path, state)
            logger.info("render unit-progress: movie concatenation done")
        else:
            _probe_encoded(movie_path, math.fsum(durations))
        srt_path = output_dir / "condensed.srt"
        if (
            repaired
            or not stage.subtitles_done
            or not _output_valid(srt_path, stage.subtitles_sha256)
        ):
            export_srt(selection, job_dir / "transcripts", srt_path, clip_durations=durations)
            stage.subtitles_done = True
            stage.subtitles_sha256 = _sha256(srt_path)
            write_json(state_path, state)
            logger.info("render unit-progress: subtitle export done")
        stage.status = "done"
        write_json(state_path, state)
        logger.info("render done: %s and %s", movie_path, srt_path)
        return state
    except Exception as exc:
        stage.status = "failed"
        state.errors.append(ErrorRecord(stage="render", episode=episode, message=str(exc)))
        write_json(state_path, state)
        logger.exception("render failed: %s", state.job_id)
        raise
