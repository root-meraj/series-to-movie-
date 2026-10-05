"""Extract usable embedded text subtitles without loading a speech model."""

from __future__ import annotations

import html
import json
import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sts.pipeline.control import run_media
from sts.schemas import TranscriptDoc, TranscriptSegment

_TEXT_CODECS = frozenset({"subrip", "ass", "ssa", "mov_text", "webvtt", "text"})
_LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "it": "it",
    "ita": "it",
    "italian": "it",
    "pt": "pt",
    "por": "pt",
    "portuguese": "pt",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kor": "ko",
    "korean": "ko",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "chinese": "zh",
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    "ar": "ar",
    "ara": "ar",
    "arabic": "ar",
    "nl": "nl",
    "nld": "nl",
    "dut": "nl",
    "dutch": "nl",
    "tr": "tr",
    "tur": "tr",
    "turkish": "tr",
}
_UNKNOWN_LANGUAGES = frozenset({"", "und", "unknown", "none", "null", "mul"})
_FORCED_TITLE = re.compile(r"\b(?:forced|signs?|songs?)\b", re.IGNORECASE)
_TIMING = re.compile(
    r"^\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})(?:\s+.*)?$"
)


@dataclass(frozen=True)
class _Candidate:
    index: int
    language_match: bool
    default: bool
    forced: bool


def _normalize_language(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-").split("-", 1)[0]
    return _LANGUAGE_ALIASES.get(normalized, normalized)


def _flag(value: object) -> bool:
    return value is True or value == 1 or value == "1"


def _candidates(streams: list[object], language: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "subtitle":
            continue
        if str(stream.get("codec_name", "")).casefold() not in _TEXT_CODECS:
            continue
        index = stream.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            continue
        tags = stream.get("tags", {})
        disposition = stream.get("disposition", {})
        if not isinstance(tags, dict) or not isinstance(disposition, dict):
            continue
        if _FORCED_TITLE.search(str(tags.get("title", ""))):
            continue
        tagged_language = _normalize_language(str(tags.get("language", "")))
        if tagged_language not in _UNKNOWN_LANGUAGES and tagged_language != language:
            continue
        forced = _flag(disposition.get("forced"))
        if forced and tagged_language != language:
            continue
        candidates.append(
            _Candidate(
                index=index,
                language_match=tagged_language == language,
                default=_flag(disposition.get("default")),
                forced=forced,
            )
        )
    return candidates


def _timestamp(value: str) -> float | None:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    if int(minutes) >= 60 or float(seconds) >= 60:
        return None
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\{\\[^}]*\}", "", value)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"</?[A-Za-z][^>]*>", "", value)
    value = value.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
    return " ".join(value.split())


def _has_speech(text: str) -> bool:
    if "♪" in text or "♫" in text:
        return False
    words = re.sub(r"\[[^]]*\]|\([^)]*\)", "", text).strip()
    if words.casefold().strip(".! ") in {
        "music",
        "silence",
        "instrumental",
        "applause",
        "inaudible",
        "no dialogue",
    }:
        return False
    return any(character.isalpha() for character in words)


def _parse_srt(value: str, episode: str, duration: float, language: str) -> TranscriptDoc:
    segments: list[TranscriptSegment] = []
    seen: set[tuple[float, float, str]] = set()
    normalized = value.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", normalized.strip()):
        lines = block.splitlines()
        for offset, line in enumerate(lines[:2]):
            timing = _TIMING.fullmatch(line)
            if timing is not None:
                body_start = offset + 1
                break
        else:
            continue
        start, end = (_timestamp(item) for item in timing.groups())
        if start is None or end is None:
            continue
        end = min(duration, end)
        text = _clean_text("\n".join(lines[body_start:]))
        if end <= start or not _has_speech(text):
            continue
        key = (start, end, text.casefold())
        if key not in seen:
            seen.add(key)
            segments.append(TranscriptSegment(start=start, end=end, text=text))
    segments.sort(key=lambda item: (item.start, item.end))
    return TranscriptDoc(
        episode=episode, language=language, segments=segments, source="embedded_subtitles"
    )


def _coverage(segments: list[TranscriptSegment]) -> float:
    total = 0.0
    previous_end = 0.0
    for segment in segments:
        total += max(0.0, segment.end - max(previous_end, segment.start))
        previous_end = max(previous_end, segment.end)
    return total


def _substantial_dialogue(segments: list[TranscriptSegment], duration: float) -> bool:
    return (
        duration > 0
        and len(segments) >= max(20, math.ceil(duration / 30.0))
        and _coverage(segments) >= duration * 0.25
        and max(item.end for item in segments) - segments[0].start >= duration * 0.65
    )


def _run(arguments: list[str], description: str, job_dir: Path) -> str:
    try:
        return run_media(arguments, job_dir).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            (exc.stderr or "").strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise RuntimeError(f"{description}: {detail or 'media tool failed'}") from exc


def extract_subtitles(
    video_path: Path,
    episode: str,
    duration: float,
    *,
    language: str = "en",
) -> tuple[TranscriptDoc, int] | None:
    """Return the best embedded dialogue transcript and its absolute subtitle stream index.

    Only text subtitle codecs are supported. Signs/songs/forced titles and other languages
    are ignored. A forced disposition can be overridden only for substantial dialogue in
    the requested language: enough cues, 25% speech coverage, and 65% episode span.
    Matching-language tracks take precedence over untagged tracks, with full non-forced
    dialogue preferred to this guarded fallback.
    The remaining tracks rank by speech cue count, coverage, default flag, then lowest index.
    Empty or malformed subtitle text returns ``None`` for speech-recognition fallback.
    Missing media tools, invalid probe metadata, and extraction failures raise ``RuntimeError``.
    """
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Subtitle duration must be finite and nonnegative")
    preferred_language = _normalize_language(language)
    if not preferred_language or preferred_language in _UNKNOWN_LANGUAGES:
        raise ValueError("A preferred subtitle language is required")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("FFprobe is required to inspect embedded subtitles; add it to PATH")
    logger = logging.getLogger(__name__)
    logger.info("subtitles started: %s", episode)
    metadata = _run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", str(video_path)],
        f"Unable to inspect subtitles in {video_path.name}",
        video_path.parent.parent,
    )
    try:
        document = json.loads(metadata)
        streams = document.get("streams", []) if isinstance(document, dict) else None
        if not isinstance(streams, list):
            raise ValueError("streams must be a list")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"FFprobe returned invalid subtitle metadata for {video_path.name}"
        ) from exc
    candidates = _candidates(streams, preferred_language)
    if not candidates:
        logger.info("subtitles done: %s has no eligible text tracks", episode)
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to extract embedded subtitles; add it to PATH")
    usable: list[tuple[tuple[bool, bool, bool, int, float, bool, int], TranscriptDoc, int]] = []
    failures: list[RuntimeError] = []
    for candidate in candidates:
        try:
            text = _run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-i",
                    str(video_path),
                    "-map",
                    f"0:{candidate.index}",
                    "-f",
                    "srt",
                    "pipe:1",
                ],
                f"Unable to extract subtitle stream {candidate.index} from {video_path.name}",
                video_path.parent.parent,
            )
        except RuntimeError as exc:
            failures.append(exc)
            logger.warning("subtitles unit-progress: %s stream %s failed", episode, candidate.index)
            continue
        transcript = _parse_srt(text, episode, duration, preferred_language)
        if not transcript.segments:
            continue
        substantial = _substantial_dialogue(transcript.segments, duration)
        if candidate.forced and not substantial:
            continue
        rank = (
            candidate.language_match,
            substantial,
            not candidate.forced,
            len(transcript.segments),
            _coverage(transcript.segments),
            candidate.default,
            -candidate.index,
        )
        usable.append((rank, transcript, candidate.index))
    if not usable:
        if failures:
            raise failures[0]
        logger.info("subtitles done: %s has no usable dialogue cues", episode)
        return None
    _, transcript, index = max(usable, key=lambda item: item[0])
    if next(candidate.forced for candidate in candidates if candidate.index == index):
        logger.warning(
            "subtitles unit-progress: %s stream %s has a forced flag but substantial full dialogue",
            episode,
            index,
        )
    logger.info("subtitles done: %s stream %s (%d cues)", episode, index, len(transcript.segments))
    return transcript, index
