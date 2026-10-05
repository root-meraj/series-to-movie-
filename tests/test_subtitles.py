"""Embedded text-track selection, subtitle parsing, and a real FFmpeg mux smoke test."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sts.pipeline import subtitles
from sts.schemas import TranscriptDoc


def _stream(
    index: int,
    *,
    language: str | None = "eng",
    codec: str = "subrip",
    title: str = "English",
    forced: bool = False,
    default: bool = False,
) -> dict[str, object]:
    tags = {"title": title}
    if language is not None:
        tags["language"] = language
    return {
        "index": index,
        "codec_type": "subtitle",
        "codec_name": codec,
        "tags": tags,
        "disposition": {"forced": int(forced), "default": int(default)},
    }


def _srt(cues: list[tuple[str, str, str]]) -> str:
    return (
        "\n\n".join(
            f"{index}\n{start} --> {end}\n{text}"
            for index, (start, end, text) in enumerate(cues, 1)
        )
        + "\n"
    )


def _media(
    monkeypatch: pytest.MonkeyPatch,
    streams: list[dict[str, object]],
    outputs: dict[int, str | subprocess.CalledProcessError],
) -> list[list[str]]:
    commands: list[list[str]] = []

    def run(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        assert options["check"] is True
        assert options["encoding"] == "utf-8"
        if arguments[0] == "ffprobe":
            output = json.dumps({"streams": streams})
        else:
            assert arguments[-3:] == ["-f", "srt", "pipe:1"]
            index = int(arguments[arguments.index("-map") + 1].split(":")[1])
            output = outputs[index]
            if isinstance(output, subprocess.CalledProcessError):
                raise output
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(subtitles.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(subtitles.subprocess, "run", run)
    return commands


def test_multiline_html_ass_and_float_timestamps_are_cleaned_and_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text styles disappear while dialogue and millisecond timestamps survive extraction."""
    output = "\ufeff" + _srt(
        [
            (
                "00:00:01,025",
                "00:00:02,500",
                r"{\an8}<i>Hello &amp; welcome.</i>" + "\n" + r"Next\Nline.",
            ),
            ("00:00:09,250", "00:00:12,000", "<font color='red'>Caf&eacute;</font><br>now."),
        ]
    ).replace("\n", "\r\n")
    commands = _media(monkeypatch, [_stream(3)], {3: output})
    result = subtitles.extract_subtitles(Path("episode with spaces.mkv"), "E01", 10.0)
    assert result is not None
    document, index = result
    assert index == 3
    assert document.episode == "E01" and document.language == "en"
    assert document.source == "embedded_subtitles"
    assert [(item.start, item.end, item.text) for item in document.segments] == [
        (1.025, 2.5, "Hello & welcome. Next line."),
        (9.25, 10.0, "Café now."),
    ]
    assert commands[1][commands[1].index("-map") + 1] == "0:3"
    assert "episode with spaces.mkv" in commands[1]


def test_malformed_unsorted_duplicate_and_non_dialogue_cues_are_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid cues cannot publish bad intervals or turn music captions into dialogue."""
    output = (
        _srt(
            [
                ("00:00:05,000", "00:00:06,000", "Later."),
                ("00:00:01.001", "00:00:02.050", "First."),
                ("00:00:01.001", "00:00:02.050", "First."),
                ("00:00:04,000", "00:00:03,000", "Reversed."),
                ("00:60:00,000", "01:01:00,000", "Invalid minutes."),
                ("00:00:10,000", "00:00:11,000", "Past the episode."),
                ("00:00:02,100", "00:00:02,200", "[MUSIC]"),
                ("00:00:02,200", "00:00:02,300", "(APPLAUSE)"),
                ("00:00:02,300", "00:00:02,400", "♪ Song lyrics ♪"),
                ("00:00:02,400", "00:00:02,500", "..."),
            ]
        )
        + "\nThis is not a cue\n"
    )
    _media(monkeypatch, [_stream(2)], {2: output})
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E02", 10.0)
    assert result is not None
    assert [(item.start, item.end, item.text) for item in result[0].segments] == [
        (1.001, 2.05, "First."),
        (5.0, 6.0, "Later."),
    ]


@pytest.mark.parametrize("tag", ["en", "eng", "English", "en-US", "en_GB"])
def test_matching_language_precedes_more_populated_unknown_track(
    monkeypatch: pytest.MonkeyPatch, tag: str
) -> None:
    """An unknown-language track cannot outweigh a valid requested-language track."""
    _media(
        monkeypatch,
        [_stream(2, language=None), _stream(5, language=tag)],
        {
            2: _srt(
                [("00:00:00,000", "00:00:01,000", "One."), ("00:00:02,000", "00:00:03,000", "Two.")]
            ),
            5: _srt([("00:00:01,000", "00:00:02,000", "Yes.")]),
        },
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)
    assert result is not None and result[1] == 5


def test_requested_other_language_uses_its_aliases_and_rejects_english(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection honors a caller's preferred language beyond the English default."""
    commands = _media(
        monkeypatch,
        [_stream(1), _stream(4, language="spa")],
        {
            4: _srt([("00:00:01,000", "00:00:02,000", "Hola.")]),
        },
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0, language="Spanish")
    assert result is not None and result[1] == 4
    assert result[0].language == "es"
    assert len(commands) == 2


def test_sparse_forced_signs_songs_wrong_language_and_image_tracks_are_not_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial translations and image captions are excluded without OCR or wrong-language text."""
    streams = [
        _stream(1, forced=True),
        _stream(2, title="English [Forced]"),
        _stream(3, title="Signs & Songs"),
        _stream(4, title="English Signs"),
        _stream(5, language="jpn"),
        _stream(6, codec="hdmv_pgs_subtitle"),
        _stream(7, codec="dvd_subtitle"),
        _stream(8, language="und", codec="ass", title="Dialogue"),
    ]
    commands = _media(
        monkeypatch,
        streams,
        {
            1: _srt([("00:00:01,000", "00:00:02,000", "A sign.")]),
            8: _srt([("00:00:01,000", "00:00:02,000", "Hello.")]),
        },
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)
    assert result is not None and result[1] == 8
    assert len(commands) == 3


def _full_dialogue() -> str:
    return _srt(
        [
            (
                f"00:{start // 60:02d}:{start % 60:02d},000",
                f"00:{(start + 2) // 60:02d}:{(start + 2) % 60:02d},000",
                f"Evidence line {index}.",
            )
            for index, start in enumerate(range(0, 120, 4))
        ]
    )


def test_substantial_forced_disposition_is_usable_when_full_dialogue_was_mistagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incorrect forced metadata cannot hide a full requested-language dialogue transcript."""
    _media(monkeypatch, [_stream(3, forced=True)], {3: _full_dialogue()})
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 120.0)
    assert result is not None and result[1] == 3
    assert len(result[0].segments) == 30


def test_full_non_forced_dialogue_precedes_full_forced_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both full tracks are present, normal non-forced dialogue remains preferred."""
    _media(
        monkeypatch,
        [_stream(1, forced=True), _stream(4)],
        {1: _full_dialogue(), 4: _full_dialogue()},
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 120.0)
    assert result is not None and result[1] == 4


@pytest.mark.parametrize("duration", [240.0, 1000.0])
def test_forced_fallback_rejects_insufficient_coverage_or_episode_span(
    monkeypatch: pytest.MonkeyPatch,
    duration: float,
) -> None:
    """A populated but partial forced track cannot stand in for the complete episode."""
    _media(monkeypatch, [_stream(3, forced=True)], {3: _full_dialogue()})
    assert subtitles.extract_subtitles(Path("episode.mkv"), "E01", duration) is None


def test_untagged_dialogue_falls_back_when_matching_text_is_corrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad extracted text in a preferred track does not hide a usable unknown-language track."""
    _media(
        monkeypatch,
        [_stream(1), _stream(2, language=None)],
        {
            1: "corrupt subtitle text",
            2: _srt([("00:00:01,000", "00:00:01,500", "No.")]),
        },
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 3600.0)
    assert result is not None and result[1] == 2
    assert result[0].segments[0].text == "No."


def test_cue_count_precedes_coverage_and_default_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dialogue cues are the first quality comparison among matching-language tracks."""
    _media(
        monkeypatch,
        [_stream(1, default=True), _stream(3)],
        {
            1: _srt([("00:00:00,000", "00:00:09,000", "Long cue.")]),
            3: _srt(
                [("00:00:00,000", "00:00:01,000", "One."), ("00:00:02,000", "00:00:03,000", "Two.")]
            ),
        },
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)
    assert result is not None and result[1] == 3


def test_equal_cue_count_prefers_union_coverage_before_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping cues cannot inflate a track's actual dialogue coverage."""
    _media(
        monkeypatch,
        [_stream(1, default=True), _stream(3)],
        {
            1: _srt(
                [("00:00:00,000", "00:00:02,000", "One."), ("00:00:00,000", "00:00:02,000", "Two.")]
            ),
            3: _srt(
                [("00:00:00,000", "00:00:02,000", "One."), ("00:00:02,000", "00:00:03,000", "Two.")]
            ),
        },
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)
    assert result is not None and result[1] == 3


@pytest.mark.parametrize("default", [False, True])
def test_equal_quality_uses_default_then_lowest_absolute_index(
    monkeypatch: pytest.MonkeyPatch, default: bool
) -> None:
    """Track choice is stable after quality ties, honoring default metadata before index."""
    cue = _srt([("00:00:00,000", "00:00:01,000", "Hello.")])
    _media(monkeypatch, [_stream(4, default=default), _stream(1)], {1: cue, 4: cue})
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)
    assert result is not None and result[1] == (4 if default else 1)


@pytest.mark.parametrize("codec", ["subrip", "ass", "ssa", "mov_text", "webvtt", "text"])
def test_text_codec_variants_accept_a_legitimate_sparse_episode(
    monkeypatch: pytest.MonkeyPatch, codec: str
) -> None:
    """A single short line is sufficient; there is no arbitrary minimum episode coverage."""
    _media(
        monkeypatch,
        [_stream(1, codec=codec)],
        {
            1: _srt([("00:00:01,000", "00:00:01,500", "Help!")]),
        },
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 7200.0)
    assert result is not None and len(result[0].segments) == 1


@pytest.mark.parametrize("output", ["", "not srt", "1\n00:00:01,000 --> 00:00:02,000\n[MUSIC]"])
def test_empty_malformed_and_sound_only_tracks_allow_asr_fallback(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    """Unusable text yields no transcript so the caller can choose speech recognition."""
    _media(monkeypatch, [_stream(1)], {1: output})
    assert subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0) is None


def test_no_eligible_tracks_does_not_require_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """An image-only or subtitle-free file returns immediately after inspecting metadata."""
    commands = _media(monkeypatch, [_stream(1, codec="hdmv_pgs_subtitle")], {})
    monkeypatch.setattr(
        subtitles.shutil, "which", lambda name: "ffprobe" if name == "ffprobe" else None
    )
    assert subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0) is None
    assert len(commands) == 1


@pytest.mark.parametrize("tool", ["ffprobe", "ffmpeg"])
def test_missing_media_tool_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """A setup problem is explicit rather than silently pretending that subtitles are absent."""
    _media(monkeypatch, [_stream(1)], {1: ""})
    monkeypatch.setattr(subtitles.shutil, "which", lambda name: None if name == tool else name)
    with pytest.raises(RuntimeError, match="required.*PATH"):
        subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)


@pytest.mark.parametrize("metadata", ["not json", "[]", '{"streams": {}}'])
def test_invalid_probe_metadata_is_reported(monkeypatch: pytest.MonkeyPatch, metadata: str) -> None:
    """Corrupt probe output is distinguished from a valid file having no subtitles."""
    _media(monkeypatch, [], {})
    monkeypatch.setattr(
        subtitles.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, metadata, ""),
    )
    with pytest.raises(RuntimeError, match="invalid subtitle metadata"):
        subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)


def test_failed_probe_preserves_tool_error_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt video reports its path and the media tool's failure explanation."""
    _media(monkeypatch, [], {})

    def fail(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, arguments, stderr="broken media")

    monkeypatch.setattr(subtitles.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="episode.mkv: broken media"):
        subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)


def test_failed_candidate_can_recover_with_another_track(monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt subtitle stream does not prevent extraction from an independently usable one."""
    _media(
        monkeypatch,
        [_stream(1), _stream(2)],
        {
            1: subprocess.CalledProcessError(1, ["ffmpeg"], stderr="bad subtitle packet"),
            2: _srt([("00:00:01,000", "00:00:02,000", "Recovered.")]),
        },
    )
    result = subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)
    assert result is not None and result[1] == 2


def test_all_candidate_process_failures_raise_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction errors remain visible when no alternate stream can be used."""
    _media(
        monkeypatch,
        [_stream(7)],
        {
            7: subprocess.CalledProcessError(1, ["ffmpeg"], stderr="bad subtitle packet"),
        },
    )
    with pytest.raises(RuntimeError, match="stream 7.*episode.mkv: bad subtitle packet"):
        subtitles.extract_subtitles(Path("episode.mkv"), "E01", 10.0)


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_invalid_duration_fails_before_media_process(duration: float) -> None:
    """Only finite nonnegative bounds can become JSON transcript timestamps."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        subtitles.extract_subtitles(Path("episode.mkv"), "E01", duration)


def test_real_muxed_subtitles_extract_without_loading_a_model(tmp_path: Path) -> None:
    """An actual text track round-trips through FFprobe/FFmpeg with exact Unicode and timing."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and FFprobe are required for the local media smoke test")
    cue_path = tmp_path / "spoken captions.srt"
    cue_path.write_text(
        _srt(
            [
                ("00:00:00,500", "00:00:01,750", "Hello, café."),
                ("00:00:02,000", "00:00:03,500", "We should leave."),
            ]
        ),
        encoding="utf-8",
    )
    video_path = tmp_path / "synthetic episode.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=10:d=4",
            "-i",
            str(cue_path),
            "-map",
            "0:v:0",
            "-map",
            "1:0",
            "-c:v",
            "libx264",
            "-c:s",
            "srt",
            "-metadata:s:s:0",
            "language=eng",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = subtitles.extract_subtitles(video_path, "E01", 4.0)
    assert result is not None
    document, index = result
    assert index == 1
    assert isinstance(document, TranscriptDoc)
    assert document.source == "embedded_subtitles"
    assert [(item.start, item.end, item.text) for item in document.segments] == [
        (0.5, 1.75, "Hello, café."),
        (2.0, 3.5, "We should leave."),
    ]
