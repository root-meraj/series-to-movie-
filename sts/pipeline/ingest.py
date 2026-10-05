"""Ordered, validated ZIP ingestion with atomic per-episode checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import subprocess
import zlib
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from zipfile import ZipFile, ZipInfo

from sts.pipeline.control import check_stop
from sts.schemas import EpisodeInfo, ErrorRecord, JobParams, JobState, read_json, write_json


def _natural_key(name: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", name)
    )


def _episode_coordinates(name: str) -> tuple[int, int] | None:
    for pattern in (r"s(\d{1,2})e(\d{1,2})(?!\d)", r"(\d{1,2})x(\d{1,2})(?!\d)"):
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match[1]), int(match[2])
    match = re.search(r"(?:episode|ep?)[\s._-]?(\d{1,3})(?!\d)", name, flags=re.IGNORECASE)
    if match:
        return 1, int(match[1])
    return None


def _episode_key(member: ZipInfo) -> tuple[int, int, tuple[tuple[int, int | str], ...], str]:
    name = PurePosixPath(member.filename.replace("\\", "/")).name
    coordinates = _episode_coordinates(name)
    if coordinates is not None:
        return *coordinates, _natural_key(name), member.filename
    # Seasonless filenames belong to season 1. The first number is the episode
    # hint; the full natural key breaks ties (including multipart filenames).
    numbers = re.findall(r"\d+", Path(name).stem)
    return 1, int(numbers[0]) if numbers else 2**63, _natural_key(name), member.filename


def _video_members(archive: ZipFile) -> list[ZipInfo]:
    videos: list[ZipInfo] = []
    names: set[str] = set()
    for member in archive.infolist():
        normalized = member.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or ":" in normalized:
            raise ValueError(f"Unsafe ZIP entry: {member.filename}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"ZIP symlinks are not supported: {member.filename}")
        if member.is_dir():
            continue
        if str(path).casefold() in names:
            raise ValueError(f"Duplicate ZIP entry: {member.filename}")
        names.add(str(path).casefold())
        if path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
            if member.flag_bits & 1:
                raise ValueError(f"Encrypted ZIP entry is not supported: {member.filename}")
            videos.append(member)
    if not videos:
        raise ValueError("The ZIP contains no supported video files")
    if not any(
        _episode_coordinates(PurePosixPath(item.filename.replace("\\", "/")).name)
        for item in videos
    ):
        return sorted(
            videos,
            key=lambda item: (
                _natural_key(PurePosixPath(item.filename.replace("\\", "/")).name),
                item.filename,
            ),
        )
    return sorted(videos, key=_episode_key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches_archive(path: Path, member: ZipInfo) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum == member.CRC


def _probe_duration(path: Path, ffprobe: str) -> float:
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"FFprobe rejected the video: {exc.stderr.strip()}") from exc
    try:
        metadata = json.loads(result.stdout)
        videos = [
            stream
            for stream in metadata.get("streams", [])
            if stream.get("codec_type") == "video"
            and not stream.get("disposition", {}).get("attached_pic", 0)
        ]
        if not videos:
            raise ValueError("Media contains no video stream")
        candidates = [metadata.get("format", {}).get("duration")]
        candidates.extend(stream.get("duration") for stream in videos)
        duration = None
        for candidate in candidates:
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                duration = value
                break
        if duration is None:
            raise ValueError("Media has no finite positive duration")
        if duration < 60.0:
            raise ValueError(f"Video duration {duration:.3f}s is below the 60s minimum")
        return duration
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise ValueError("FFprobe returned invalid media metadata") from exc


def _extract_video(archive: ZipFile, member: ZipInfo, destination: Path, ffprobe: str) -> float:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            with archive.open(member) as source:
                while block := source.read(1024 * 1024):
                    check_stop(destination.parent.parent)
                    stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        duration = _probe_duration(temporary_path, ffprobe)
        temporary_path.replace(destination)
        return duration
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ingest_zip(zip_path: Path, job_dir: Path) -> list[EpisodeInfo]:
    """Extract and probe episodes in natural order, saving progress after each one.

    Reject corrupt, short, or unsafe inputs without discarding earlier completed
    episodes. Retrying the same ZIP resumes the checkpoint. Completed stages are
    idempotent, and a changed source ZIP requires a new job.
    """
    logger = logging.getLogger(__name__)
    job_dir = job_dir.resolve()
    state_path = job_dir / "state.json"
    state = (
        read_json(state_path, JobState)
        if state_path.exists()
        else JobState(job_id=job_dir.name, params=JobParams(zip_path=zip_path.resolve()))
    )
    stage = state.stages.ingest
    if stage.status == "done":
        logger.info("ingest done: checkpoint already complete (%d episodes)", stage.episode_count)
        return list(stage.episodes)

    episode: str | None = None
    try:
        stage.status = "in_progress"
        write_json(state_path, state)
        logger.info("ingest started: %s", state.job_id)
        zip_path = zip_path.resolve(strict=True)
        fingerprint = _sha256(zip_path)
        if state.params.zip_sha256 is not None and state.params.zip_sha256 != fingerprint:
            raise ValueError("Source ZIP changed since ingestion started; create a new job")
        state.params.zip_path = zip_path
        state.params.zip_sha256 = fingerprint
        write_json(state_path, state)
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise RuntimeError(
                "FFprobe is required; install FFmpeg and add its bin directory to PATH"
            )
        episodes_dir = job_dir / "episodes"
        if episodes_dir.is_symlink():
            raise ValueError("The episodes directory must not be a symlink")
        episodes_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path) as archive:
            members = _video_members(archive)
            planned = [f"E{index:02d}" for index in range(1, len(members) + 1)]
            verified: list[EpisodeInfo] = []
            previous = {item.episode: item for item in stage.episodes}
            for index, member in enumerate(members, start=1):
                check_stop(job_dir)
                identifier = f"E{index:02d}"
                suffix = PurePosixPath(member.filename.replace("\\", "/")).suffix.lower()
                destination = episodes_dir / f"{identifier}{suffix}"
                cached = previous.get(identifier)
                if (
                    cached is not None
                    and identifier in stage.completed
                    and cached.original_name == member.filename
                    and cached.source_path == destination
                    and _matches_archive(destination, member)
                ):
                    verified.append(cached)
            stage.episodes = verified
            stage.completed = [item.episode for item in verified]
            stage.pending = [
                identifier for identifier in planned if identifier not in stage.completed
            ]
            stage.episode_count = len(verified)
            write_json(state_path, state)

            for index, member in enumerate(members, start=1):
                episode = f"E{index:02d}"
                check_stop(job_dir)
                if episode in stage.completed:
                    logger.info("ingest unit-progress: %s already complete", episode)
                    continue
                suffix = PurePosixPath(member.filename.replace("\\", "/")).suffix.lower()
                destination = episodes_dir / f"{episode}{suffix}"
                duration = _extract_video(archive, member, destination, ffprobe)
                stage.episodes.append(
                    EpisodeInfo(
                        episode=episode,
                        episode_number=index,
                        source_path=destination,
                        original_name=member.filename,
                        duration=duration,
                    )
                )
                stage.episodes.sort(key=lambda item: item.episode_number)
                stage.completed = [item.episode for item in stage.episodes]
                stage.pending = [item for item in planned if item not in stage.completed]
                stage.episode_count = len(stage.completed)
                write_json(state_path, state)
                logger.info("ingest unit-progress: %s done (%d/%d)", episode, index, len(members))

        stage.status = "done"
        state.stages.transcribe.pending = list(stage.completed)
        write_json(state_path, state)
        logger.info("ingest done: %d episodes", stage.episode_count)
        return list(stage.episodes)
    except Exception as exc:
        stage.status = "failed"
        state.errors.append(ErrorRecord(stage="ingest", episode=episode, message=str(exc)))
        write_json(state_path, state)
        logger.exception("ingest failed: %s", state.job_id)
        raise
