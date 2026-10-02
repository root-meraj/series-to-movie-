"""Synthetic fixture modes preserve ordering, containers, and atomic publication."""

from __future__ import annotations

import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import make_fixture


@pytest.fixture
def fixture_commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stand in for FFmpeg while retaining the real ZIP assembly and file moves."""
    commands: list[list[str]] = []
    monkeypatch.setattr(make_fixture.shutil, "which", lambda name: name)

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        destination = Path(arguments[-1])
        destination.write_bytes(b"mkv" if destination.suffix == ".mkv" else b"mp4")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(make_fixture.subprocess, "run", run)
    return commands


def test_default_fixture_preserves_original_source_and_tricky_order(
    tmp_path: Path, fixture_commands: list[list[str]]
) -> None:
    """The backward-compatible default retains uncut video and valid archive containers."""
    archive_path = make_fixture.make_fixture(tmp_path / "fixture")
    assert "testsrc2=duration=90:size=640x360:rate=30" in fixture_commands[0]
    assert "-filter_complex" not in fixture_commands[0]
    assert "-ss" not in fixture_commands[1]
    assert fixture_commands[1][fixture_commands[1].index("-c") + 1] == "copy"
    with ZipFile(archive_path) as archive:
        assert archive.namelist() == ["ep10.mp4", "show_1x02.mkv", "show.S01E01.mp4"]
        assert archive.read("show_1x02.mkv") == b"mkv"
        assert archive.read("show.S01E01.mp4") == b"mp4"
    assert (archive_path.parent / "ep_test.mp4").read_bytes() == b"mp4"


def test_cut_fixture_encodes_three_shots_and_maps_the_fourth_inputs_audio(
    tmp_path: Path, fixture_commands: list[list[str]]
) -> None:
    """Cut mode uses exact 30-second hard cuts without changing total episode length."""
    archive_path = make_fixture.make_fixture(tmp_path / "cut fixture", scene_cuts=True)
    command = fixture_commands[0]
    assert all(
        f"color=c={color}:duration=30:size=640x360:rate=30" in command
        for color in ("red", "green", "blue")
    )
    assert "[0:v][1:v][2:v]concat=n=3:v=1:a=0[video]" in command
    assert "3:a:0" in command
    assert "sine=frequency=440:duration=90" in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert archive_path.is_file()


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_fixture_reports_missing_media_tool_before_creating_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Missing system tools produce an actionable failure before any files are written."""
    monkeypatch.setattr(
        make_fixture.shutil, "which", lambda name: None if name == missing else name
    )
    destination = tmp_path / "fixture"
    with pytest.raises(RuntimeError, match="FFmpeg|FFprobe"):
        make_fixture.make_fixture(destination, scene_cuts=True)
    assert not destination.exists()


def test_failed_generation_preserves_previously_published_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_commands: list[list[str]]
) -> None:
    """A failed new encoding cannot replace an existing fixture video or ZIP."""
    destination = tmp_path / "fixture"
    archive = make_fixture.make_fixture(destination)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}

    def fail(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(make_fixture.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        make_fixture.make_fixture(destination, scene_cuts=True)
    assert archive.exists()
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
