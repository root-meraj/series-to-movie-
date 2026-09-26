"""ZIP ordering, safe extraction, media validation, and ingestion failures."""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZipInfo

import pytest

from sts.pipeline.ingest import ingest_zip
from sts.schemas import JobState, read_json


def test_mixed_episode_names_sort_and_keep_real_extensions(
    tmp_path: Path, make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> None:
    """Season/episode patterns and seasonless ep10 sort numerically, not lexically."""
    archive = make_zip(
        [
            ("ep10.mp4", b"ten"),
            ("folder/show_1x02.mkv", b"two"),
            ("show.S01E01.MP4", b"one"),
            ("README.txt", b"ignored"),
        ]
    )
    episodes = ingest_zip(archive, tmp_path / "job")
    assert [item.episode for item in episodes] == ["E01", "E02", "E03"]
    assert [item.source_path.name for item in episodes] == ["E01.mp4", "E02.mkv", "E03.mp4"]
    assert probe_calls == [b"one", b"two", b"ten"]
    assert [item.source_path.read_bytes() for item in episodes] == probe_calls
    state = read_json(tmp_path / "job/state.json", JobState)
    assert state.stages.ingest.status == "done"
    assert state.stages.ingest.episode_count == 3
    assert state.stages.transcribe.pending == ["E01", "E02", "E03"]


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["s02e02.mp4", "s01e10.mp4", "s01e02.mp4"], [2, 1, 0]),
        (["show_2x01.mkv", "show_1x10.mkv", "show_1x02.mkv"], [2, 1, 0]),
        (["episode_10.avi", "episode 2.mov", "episode-1.webm"], [2, 1, 0]),
        (["clip10.mp4", "clip2.mp4", "clip1.mp4"], [2, 1, 0]),
        (["bravo2.mp4", "alpha10.mp4", "alpha2.mp4"], [2, 1, 0]),
        (["zeta.mp4", "Alpha.mp4", "beta.mp4"], [1, 2, 0]),
        (["s02e01_1x01.mp4", "s01e02.mp4"], [1, 0]),
    ],
)
def test_ordering_patterns_and_fallback(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    names: list[str],
    expected: list[int],
) -> None:
    """Regex priority, multi-season order, and natural fallback are deterministic."""
    archive = make_zip([(name, str(index).encode()) for index, name in enumerate(names)])
    ingest_zip(archive, tmp_path / "job")
    assert probe_calls == [str(index).encode() for index in expected]


@pytest.mark.parametrize("unsafe", ["../escape.mp4", "/absolute.mp4", "C:/bad.mp4", "..\\bad.mp4"])
def test_unsafe_archive_paths_are_rejected(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    unsafe: str,
) -> None:
    """A ZIP cannot write outside the job directory on any supported platform."""
    archive = make_zip([(unsafe, b"video")])
    with pytest.raises(ValueError, match="Unsafe ZIP entry"):
        ingest_zip(archive, tmp_path / "job")
    assert probe_calls == []
    assert not (tmp_path / "escape.mp4").exists()
    assert read_json(tmp_path / "job/state.json", JobState).stages.ingest.status == "failed"


def test_symlink_archive_entry_is_rejected(tmp_path: Path, probe_calls: list[bytes]) -> None:
    """Unix symlink metadata is rejected even when ingestion runs on Windows."""
    archive = tmp_path / "show.zip"
    member = ZipInfo("link.mp4")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as output:
        output.writestr(member, "../outside.mp4")
    with pytest.raises(ValueError, match="symlinks"):
        ingest_zip(archive, tmp_path / "job")
    assert probe_calls == []


def test_case_colliding_entries_are_rejected(
    tmp_path: Path, make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> None:
    """Ambiguous archive entries fail consistently on case-sensitive and Windows filesystems."""
    archive = make_zip([("ep1.mp4", b"a"), ("EP1.mp4", b"b")])
    with pytest.raises(ValueError, match="Duplicate"):
        ingest_zip(archive, tmp_path / "job")
    assert probe_calls == []


def test_empty_archive_is_rejected(
    tmp_path: Path, make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> None:
    """Archives without supported videos do not silently create successful jobs."""
    with pytest.raises(ValueError, match="no supported video"):
        ingest_zip(make_zip([("readme.txt", b"text")]), tmp_path / "job")
    assert probe_calls == []


def test_invalid_zip_records_failure(tmp_path: Path, probe_calls: list[bytes]) -> None:
    """A corrupt ZIP records an error and leaves ingestion resumable."""
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(BadZipFile):
        ingest_zip(archive, tmp_path / "job")
    state = read_json(tmp_path / "job/state.json", JobState)
    assert state.errors[-1].stage == "ingest"
    assert state.errors[-1].episode is None
    assert probe_calls == []


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"streams": [], "format": {"duration": "90"}}, "no video stream"),
        ({"streams": [{"codec_type": "audio"}], "format": {"duration": "90"}}, "no video stream"),
        ({"streams": [{"codec_type": "video"}], "format": {"duration": "59.99"}}, "60s minimum"),
        ({"streams": [{"codec_type": "video"}], "format": {"duration": "nan"}}, "finite positive"),
        ({"streams": [{"codec_type": "video"}], "format": {}}, "finite positive"),
        ([], "invalid media metadata"),
    ],
)
def test_invalid_media_is_not_committed(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
    metadata: object,
    message: str,
) -> None:
    """Invalid probe results do not leave a final episode or an extraction temporary file."""
    monkeypatch.setattr(
        "sts.pipeline.ingest.subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(metadata), ""),
    )
    with pytest.raises(ValueError, match=message):
        ingest_zip(make_zip([("ep1.mp4", b"bad")]), tmp_path / "job")
    assert list((tmp_path / "job/episodes").iterdir()) == []
    state = read_json(tmp_path / "job/state.json", JobState)
    assert state.stages.ingest.completed == []
    assert state.stages.ingest.pending == ["E01"]
    assert state.errors[-1].episode == "E01"


def test_video_stream_duration_fallback_accepts_sixty_seconds(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid stream duration is usable when container duration is unavailable."""
    metadata = {"streams": [{"codec_type": "video", "duration": "60"}], "format": {}}
    monkeypatch.setattr(
        "sts.pipeline.ingest.subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(metadata), ""),
    )
    episodes = ingest_zip(make_zip([("ep1.mp4", b"video")]), tmp_path / "job")
    assert episodes[0].duration == 60.0


def test_failed_ffprobe_cleans_temporary_file(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decoder rejection is recorded without marking the episode completed."""

    def fail(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, args, stderr="invalid data")

    monkeypatch.setattr("sts.pipeline.ingest.subprocess.run", fail)
    with pytest.raises(ValueError, match="FFprobe rejected"):
        ingest_zip(make_zip([("ep1.mp4", b"broken")]), tmp_path / "job")
    assert list((tmp_path / "job/episodes").iterdir()) == []


def test_missing_ffprobe_is_actionable_and_persisted(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing system dependency produces a clear error and a failed checkpoint."""
    monkeypatch.setattr("sts.pipeline.ingest.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="FFprobe is required"):
        ingest_zip(make_zip([("ep1.mp4", b"video")]), tmp_path / "job")
    assert read_json(tmp_path / "job/state.json", JobState).stages.ingest.status == "failed"
