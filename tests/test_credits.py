"""Credit exclusions require metadata or repeated frame evidence, never silence alone."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from sts.pipeline import credits
from sts.pipeline.control import ProcessingStopped, set_stop_requested
from sts.schemas import TranscriptDoc, TranscriptSegment


def _transcript(*segments: tuple[float, float, str]) -> TranscriptDoc:
    return TranscriptDoc(
        episode="E01",
        language="en",
        segments=[
            TranscriptSegment(start=start, end=end, text=text) for start, end, text in segments
        ],
    )


def _card(index: int = 0) -> NDArray[np.uint8]:
    frame = np.zeros((360, 640), dtype=np.uint8)
    for row, text in enumerate(("EXECUTIVE PRODUCERS", "DAVID JOHNSON", f"MICHAEL WILSON {index}")):
        cv2.putText(frame, text, (160, 110 + row * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 220, 1)
    return frame


@pytest.fixture
def no_chapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep heuristic tests independent of FFprobe and real media."""
    monkeypatch.setattr(credits, "_chapters", lambda *args: [])


def test_named_credit_chapter_removes_exact_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit chapters provide precise starts without scanning any video frames."""
    monkeypatch.setattr(
        credits,
        "_chapters",
        lambda *args: [
            {"start_time": "800", "end_time": "999.5", "tags": {"title": "End Credits"}}
        ],
    )
    monkeypatch.setattr(
        credits, "_sample_frame", lambda *args: pytest.fail("metadata skips frames")
    )
    result = credits.detect_credits(Path("E01.mkv"), _transcript((797, 799, "Goodbye.")), 1000)
    assert [(item.start, item.end) for item in result.ranges] == [(800.0, 1000.0)]
    assert result.ranges[0].evidence == "chapter: End Credits"


def test_metadata_preserves_postcredit_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything from post-credit dialogue through the final silent action is retained."""
    monkeypatch.setattr(
        credits,
        "_chapters",
        lambda *args: [{"start_time": "800", "end_time": "1000", "tags": {"title": "Credits"}}],
    )
    result = credits.detect_credits(Path("E01.mkv"), _transcript((940, 943, "He is alive.")), 1000)
    assert [(item.start, item.end) for item in result.ranges] == [(800.0, 937.0)]


def test_duplicate_credit_chapters_remain_one_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overlapping metadata cannot produce an invalid exclusion contract."""
    monkeypatch.setattr(
        credits,
        "_chapters",
        lambda *args: [
            {"start_time": "800", "end_time": "990", "tags": {"title": "Credits"}},
            {"start_time": "850", "end_time": "1000", "tags": {"title": "End Credits"}},
        ],
    )
    result = credits.detect_credits(Path("E01.mkv"), _transcript(), 1000)
    assert [(item.start, item.end) for item in result.ranges] == [(800.0, 1000.0)]


def test_quiet_ending_without_text_is_not_credits(
    monkeypatch: pytest.MonkeyPatch, no_chapters: None
) -> None:
    """A long subtitle gap or black final frames alone never authorizes trimming."""
    monkeypatch.setattr(credits, "_sample_frame", lambda *args: np.zeros((360, 640), np.uint8))
    result = credits.detect_credits(
        Path("E01.mkv"), _transcript((700, 704, "I'll wait here.")), 1000
    )
    assert result.ranges == []


def test_changing_text_cards_establish_credit_range(
    monkeypatch: pytest.MonkeyPatch, no_chapters: None
) -> None:
    """Repeated changing text on black can identify credits without chapters."""
    monkeypatch.setattr(
        credits,
        "_sample_frame",
        lambda path, time, out, job: (
            _card(int(time)) if time >= 850 else np.full((360, 640), 120, np.uint8)
        ),
    )
    result = credits.detect_credits(
        Path("E01.mkv"), _transcript((790, 794, "It's finished.")), 1000
    )
    assert len(result.ranges) == 1
    assert 850 <= result.ranges[0].start <= 850.625
    assert result.ranges[0].end == 1000


def test_static_title_card_is_not_a_credit_roll(
    monkeypatch: pytest.MonkeyPatch, no_chapters: None
) -> None:
    """An identical title shown repeatedly is insufficient credit evidence."""
    monkeypatch.setattr(credits, "_sample_frame", lambda *args: _card())
    assert credits.detect_credits(Path("E01.mkv"), _transcript(), 1000).ranges == []


def test_final_seconds_with_quiet_story_are_preserved(
    monkeypatch: pytest.MonkeyPatch, no_chapters: None
) -> None:
    """A final quiet shot cannot be swallowed by extending the last credit sample."""
    monkeypatch.setattr(
        credits,
        "_sample_frame",
        lambda path, time, out, job: (
            _card(int(time)) if 850 <= time < 995 else np.full((360, 640), 120, np.uint8)
        ),
    )
    result = credits.detect_credits(Path("E01.mkv"), _transcript(), 1000)
    assert result.ranges[0].end == 990


def test_serif_letter_connections_are_valid_text() -> None:
    """A short card with joined serif letters still has plausible horizontal text."""
    frame = np.zeros((360, 640), np.uint8)
    for index in range(3):
        cv2.rectangle(frame, (276 + index * 30, 153), (302 + index * 30, 167), 220, 1)
        cv2.line(frame, (276 + index * 30, 160), (302 + index * 30, 160), 220, 1)
    assert credits._frame_kind(frame) == "credits"


def test_frame_evidence_preserves_quiet_postcredit_scene(
    monkeypatch: pytest.MonkeyPatch, no_chapters: None
) -> None:
    """A following story frame ends exclusion even when its scene has no subtitles."""
    monkeypatch.setattr(
        credits,
        "_sample_frame",
        lambda path, time, out, job: (
            _card(int(time)) if 850 <= time < 930 else np.full((360, 640), 120, np.uint8)
        ),
    )
    result = credits.detect_credits(Path("E01.mkv"), _transcript(), 1000)
    assert len(result.ranges) == 1
    assert result.ranges[0].start >= 850
    assert result.ranges[0].end == 920


def test_postcredit_spoken_subtitles_protect_story(
    monkeypatch: pytest.MonkeyPatch, no_chapters: None
) -> None:
    """Spoken dialogue interrupts a visually credit-like run and remains preserved."""
    monkeypatch.setattr(credits, "_sample_frame", lambda path, time, out, job: _card(int(time)))
    result = credits.detect_credits(
        Path("E01.mkv"), _transcript((945, 949, "The killer escaped.")), 1000
    )
    assert result.ranges
    assert result.ranges[0].end <= 942
    assert all(not (item.start < 949 and item.end > 945) for item in result.ranges)


def test_musical_and_sound_cues_do_not_block_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credit music/standalone sound descriptions do not count as post-credit speech."""
    monkeypatch.setattr(
        credits,
        "_chapters",
        lambda *args: [{"start_time": "800", "end_time": "1000", "tags": {"title": "Credits"}}],
    )
    result = credits.detect_credits(
        Path("E01.mkv"),
        _transcript((900, 905, "♪ I will be there ♪"), (950, 955, "[music ends]")),
        1000,
    )
    assert result.ranges[0].end == 1000


def test_dark_scattered_lights_are_not_text() -> None:
    """Sparse stars do not share credit-like horizontal glyph rows."""
    frame = np.zeros((360, 640), np.uint8)
    for index in range(12):
        cv2.rectangle(
            frame, (40 + index * 40, 15 + index * 25), (43 + index * 40, 18 + index * 25), 220, -1
        )
    assert credits._frame_kind(frame) == "other"


def test_credit_detection_keeps_owned_stop_request(tmp_path: Path) -> None:
    """A requested stop propagates instead of becoming an unknown-credit result."""
    set_stop_requested(tmp_path, True)
    with pytest.raises(ProcessingStopped):
        credits.detect_credits(Path("E01.mkv"), _transcript(), 1000, job_dir=tmp_path)


def test_stop_between_sparse_frame_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_chapters: None
) -> None:
    """Cancellation during sampling is checked before decoding another frame."""
    calls: list[float] = []

    def fake_frame(
        path: Path, timestamp: float, out: Path, job_dir: Path | None
    ) -> NDArray[np.uint8]:
        calls.append(timestamp)
        set_stop_requested(tmp_path, True)
        return _card()

    monkeypatch.setattr(credits, "_sample_frame", fake_frame)
    with pytest.raises(ProcessingStopped):
        credits.detect_credits(Path("E01.mkv"), _transcript(), 1000, job_dir=tmp_path)
    assert len(calls) == 1


def test_missing_tool_or_broken_media_fails_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown media cannot cause arbitrary final-minute trimming."""

    def fail(*args: object) -> list[dict[str, object]]:
        raise subprocess.CalledProcessError(1, "ffprobe", stderr="corrupt media")

    monkeypatch.setattr(credits, "_chapters", fail)
    assert credits.detect_credits(Path("E01.mkv"), _transcript(), 1000).ranges == []


def test_probe_uses_path_and_no_video_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chapter lookup issues a metadata-only FFprobe request with the exact source path."""
    seen: list[list[str]] = []

    def fake_run(args: list[str], job_dir: Path | None) -> subprocess.CompletedProcess[str]:
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps({"chapters": []}), "")

    monkeypatch.setattr(credits.shutil, "which", lambda name: name)
    monkeypatch.setattr(credits, "_run", fake_run)
    assert credits._chapters(Path("season space/E01.mkv"), None) == []
    assert seen[0][-1] == str(Path("season space/E01.mkv"))
    assert "-show_chapters" in seen[0]


def test_frame_samples_seek_before_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sparse samples seek first and decode only one small frame, keeping costs bounded."""
    seen: list[list[str]] = []

    def fake_run(args: list[str], job_dir: Path | None) -> subprocess.CompletedProcess[str]:
        seen.append(args)
        cv2.imwrite(args[-1], _card())
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(credits.shutil, "which", lambda name: name)
    monkeypatch.setattr(credits, "_run", fake_run)
    assert credits._sample_frame(Path("E01.mkv"), 850.25, tmp_path / "frame.png", None) is not None
    assert seen[0].index("-ss") < seen[0].index("-i")
    assert seen[0][seen[0].index("-frames:v") + 1] == "1"
