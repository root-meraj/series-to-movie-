"""Subtitle import follows normal transcript/RMS contracts without invoking Whisper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from sts.pipeline import transcribe
from sts.schemas import (
    AudioFeaturesDoc,
    EpisodeInfo,
    JobParams,
    JobState,
    TranscriptDoc,
    read_json,
    write_json,
)


def _job(tmp_path: Path, source: str = "auto") -> Path:
    directory = tmp_path / "subtitle-job"
    state = JobState(
        job_id="subtitle-job",
        params=JobParams(transcript_source=source, whisper_device="cpu", whisper_compute="int8"),
    )
    state.stages.ingest.status = "done"
    state.stages.ingest.episode_count = 1
    state.stages.ingest.episodes = [
        EpisodeInfo(
            episode="E01",
            episode_number=1,
            source_path=directory / "episodes/E01.mkv",
            original_name="episode1.mkv",
            duration=90.0,
        )
    ]
    write_json(directory / "state.json", state)
    return directory


def _extract_audio(video: Path, audio: Path) -> Path:
    audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio, np.full(160000, 0.25, dtype=np.float32), 16000, subtype="FLOAT")
    return audio


def _subtitles() -> tuple[TranscriptDoc, int]:
    return TranscriptDoc(
        episode="E01",
        language="en",
        segments=[{"start": 1.0, "end": 3.0, "text": "We found the missing evidence."}],
    ), 3


def test_automatic_subtitle_import_never_loads_whisper_and_commits_normal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text subtitles skip ASR while RMS, episode checkpoints, and timestamps remain usable."""
    directory = _job(tmp_path)

    def extract(*args: object, **kwargs: object) -> tuple[TranscriptDoc, int]:
        current = read_json(directory / "state.json", JobState).stages.transcribe.current
        assert current.activity == "extracting_subtitles"
        return _subtitles()

    monkeypatch.setattr(transcribe, "extract_subtitles", extract)
    monkeypatch.setattr(transcribe, "extract_audio", _extract_audio)
    result = transcribe.transcribe_job(directory)
    assert result.stages.transcribe.completed == ["E01"]
    assert result.stages.transcribe.pending == []
    assert result.stages.transcribe.current is None
    assert result.stages.transcribe.status == "done"
    document = read_json(directory / "transcripts/E01.transcript.json", TranscriptDoc)
    assert document.source == "embedded_subtitles"
    assert document.language == "en" and document.segments == _subtitles()[0].segments
    assert (
        read_json(directory / "audio/E01.rms.json", AudioFeaturesDoc).rms_per_second == [0.25] * 10
    )
    assert transcribe._MODEL is None


def test_subtitles_only_missing_track_fails_before_audio_or_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit subtitles-only mode records a useful error rather than silently invoking ASR."""
    directory = _job(tmp_path, "subtitles")

    def unexpected_audio(*args: object) -> Path:
        raise AssertionError("Audio extraction must not run without the required subtitles")

    monkeypatch.setattr(transcribe, "extract_audio", unexpected_audio)
    with pytest.raises(ValueError, match="No usable en text subtitles"):
        transcribe.transcribe_job(directory)
    state = read_json(directory / "state.json", JobState)
    assert state.stages.transcribe.status == "failed"
    assert state.stages.transcribe.completed == []
    assert state.errors[-1].episode == "E01"


def test_automatic_mode_falls_back_to_whisper_using_saved_preferences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subtitle-free episode retains ASR fallback and ignores later environment changes."""
    directory = _job(tmp_path)
    state = read_json(directory / "state.json", JobState)
    state.params.subtitle_language = "fr"
    write_json(directory / "state.json", state)
    languages: list[str] = []

    def missing(video: Path, episode: str, duration: float, *, language: str) -> None:
        languages.append(language)

    class Model:
        def transcribe(self, wav: object, **kwargs: object) -> tuple[object, object]:
            """Return an empty, valid Whisper decode without constructing a trained model."""
            return iter([]), SimpleNamespace(language="fr")

    monkeypatch.setenv("TRANSCRIPT_SOURCE", "subtitles")
    monkeypatch.setenv("SUBTITLE_LANGUAGE", "en")
    monkeypatch.setattr(transcribe, "extract_subtitles", missing)
    monkeypatch.setattr(transcribe, "extract_audio", _extract_audio)
    monkeypatch.setattr(transcribe, "_create_model", lambda cfg: Model())
    assert transcribe.transcribe_job(directory).stages.transcribe.status == "done"
    document = read_json(directory / "transcripts/E01.transcript.json", TranscriptDoc)
    assert languages == ["fr"]
    assert document.source == "whisper" and document.segments == []


def test_failed_subtitle_publication_retries_whole_episode_without_false_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving RMS before a failed transcript write cannot mark the episode complete."""
    directory = _job(tmp_path, "subtitles")
    calls: list[int] = []

    def extract(*args: object, **kwargs: object) -> tuple[TranscriptDoc, int]:
        calls.append(1)
        return _subtitles()

    original_write = transcribe.write_json

    def fail_transcript(path: Path, document: object) -> None:
        if path.name == "E01.transcript.json":
            raise OSError("transcript disk error")
        original_write(path, document)

    monkeypatch.setattr(transcribe, "extract_subtitles", extract)
    monkeypatch.setattr(transcribe, "extract_audio", _extract_audio)
    monkeypatch.setattr(transcribe, "write_json", fail_transcript)
    with pytest.raises(OSError, match="transcript disk error"):
        transcribe.transcribe_job(directory)
    assert read_json(directory / "state.json", JobState).stages.transcribe.completed == []
    monkeypatch.setattr(transcribe, "write_json", original_write)
    assert transcribe.transcribe_job(directory).stages.transcribe.status == "done"
    assert len(calls) == 2


def test_subtitles_only_resume_replaces_a_legacy_asr_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing an unfinished job to subtitles-only reimports incompatible ASR artifacts."""
    directory = _job(tmp_path, "subtitles")
    _extract_audio(Path("E01.mkv"), directory / "audio/E01.wav")
    document = TranscriptDoc(episode="E01", language="hi", segments=[])
    write_json(directory / "transcripts/E01.transcript.json", document)
    write_json(
        directory / "audio/E01.rms.json",
        AudioFeaturesDoc(
            episode="E01",
            duration=10,
            rms_per_second=[0.25] * 10,
        ),
    )
    state = read_json(directory / "state.json", JobState)
    state.stages.transcribe.completed = ["E01"]
    write_json(directory / "state.json", state)
    monkeypatch.setattr(transcribe, "extract_subtitles", lambda *args, **kwargs: _subtitles())
    monkeypatch.setattr(transcribe, "extract_audio", _extract_audio)
    assert transcribe.transcribe_job(directory).stages.transcribe.status == "done"
    assert read_json(directory / "transcripts/E01.transcript.json", TranscriptDoc).source == (
        "embedded_subtitles"
    )
