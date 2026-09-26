"""Mocked Whisper filtering, RMS extraction, persistence, and mid-season resume."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray
from pydantic import ValidationError

from sts.app.cli import main
from sts.config import Settings
from sts.pipeline import transcribe
from sts.pipeline.controller import get_status, list_jobs, resume_job, run_job
from sts.schemas import AudioFeaturesDoc, JobState, TranscriptDoc, read_json, write_json


@dataclass
class _Segment:
    start: float = 0.0
    end: float = 1.0
    text: str = "Hello there."
    no_speech_prob: float = 0.0
    avg_logprob: float = -0.1


class _Info:
    def __init__(self, model: _Model) -> None:
        self.model = model

    @property
    def language(self) -> str:
        """Require the segment generator to be exhausted before metadata is used."""
        assert self.model.exhausted
        return "en"


class _Model:
    def __init__(self) -> None:
        self.segments = [_Segment()]
        self.calls: list[tuple[NDArray[np.float32], dict[str, bool]]] = []
        self.fail_on_call: int | None = None
        self.exhausted = False

    def transcribe(
        self, wav: NDArray[np.float32], **options: bool
    ) -> tuple[Iterator[_Segment], _Info]:
        """Return a lazy generator, including failures during consumption."""
        self.calls.append((wav, options))
        self.exhausted = False

        def generate() -> Iterator[_Segment]:
            if self.fail_on_call == len(self.calls):
                raise RuntimeError("simulated Whisper decoding failure")
            yield from self.segments
            self.exhausted = True

        return generate(), _Info(self)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> _Model:
    """Install a deterministic Whisper double without importing or downloading a model."""
    fake = _Model()
    monkeypatch.setattr(transcribe, "_create_model", lambda cfg: fake)
    return fake


def _write_wav(path: Path, samples: NDArray[np.float32], sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate, subtype="FLOAT")
    return path


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    """Make a short mono WAV for unit tests without invoking FFmpeg."""
    return _write_wav(tmp_path / "audio/E01.wav", np.full(160000, 0.25, dtype=np.float32))


@pytest.fixture
def extraction_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace audio extraction while keeping real WAV and JSON persistence."""
    calls: list[str] = []

    def extract(video_path: Path, audio_path: Path) -> Path:
        calls.append(video_path.stem)
        return _write_wav(audio_path, np.full(48000, 0.25, dtype=np.float32))

    monkeypatch.setattr(transcribe, "extract_audio", extract)
    return calls


@pytest.fixture
def ingested_job(
    make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> JobState:
    """Create an ingested three-episode season with frozen nondefault Whisper settings."""
    return run_job(
        make_zip([("ep3.mp4", b"three"), ("ep1.mp4", b"one"), ("ep2.mp4", b"two")]),
        cfg=Settings(whisper_model="tiny", whisper_device="cpu", whisper_compute="int8"),
        stage="ingest",
    )


def test_decodes_once_materializes_generator_and_passes_required_options(
    audio: Path, model: _Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whisper receives the decoded waveform, with VAD and anti-loop options enabled."""
    reads: list[Path] = []
    original = sf.read

    def read_once(path: Path, **kwargs: object) -> tuple[NDArray[np.float32], int]:
        reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(sf, "read", read_once)
    output = audio.parent.parent / "transcripts/E01.transcript.json"
    document = transcribe.transcribe_episode(audio, output, Settings())
    assert reads == [audio]
    assert model.exhausted
    assert model.calls[0][0].dtype == np.float32
    assert model.calls[0][1] == {
        "word_timestamps": False,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    assert read_json(output, TranscriptDoc) == document
    assert document.segments[0].text == "Hello there."
    assert read_json(audio.with_suffix(".rms.json"), AudioFeaturesDoc).rms_per_second == [0.25] * 10


def test_filters_hallucinations_and_consecutive_duplicates(audio: Path, model: _Model) -> None:
    """Low-confidence speech, no-speech artifacts, and repeated retained text are removed."""
    model.segments = [
        _Segment(0, 1, "  Hello   there.  "),
        _Segment(1, 2, "HELLO THERE."),
        _Segment(2, 3, "music artifact", no_speech_prob=0.61),
        _Segment(3, 4, "low confidence", avg_logprob=-1.01),
        _Segment(4, 5, "Exact thresholds stay.", no_speech_prob=0.6, avg_logprob=-1.0),
        _Segment(5, 6, "   "),
        _Segment(6, 7, "Hello there."),
    ]
    document = transcribe.transcribe_episode(audio, audio.parent / "result.json", Settings())
    assert [segment.text for segment in document.segments] == [
        "Hello there.",
        "Exact thresholds stay.",
        "Hello there.",
    ]


def test_empty_transcript_and_silent_rms_are_valid(audio: Path, model: _Model) -> None:
    """Silence preserves a valid empty transcript and zero-valued features."""
    _write_wav(audio, np.zeros(16000, dtype=np.float32))
    model.segments = []
    document = transcribe.transcribe_episode(audio, audio.parent / "result.json", Settings())
    assert document.language == "en"
    assert document.segments == []
    assert read_json(audio.with_suffix(".rms.json"), AudioFeaturesDoc).rms_per_second == [0.0]


def test_partial_second_rms_uses_actual_samples(audio: Path, model: _Model) -> None:
    """RMS uses sample amplitudes and does not dilute the final half-second with padding."""
    samples = np.concatenate(
        (
            np.zeros(16000, dtype=np.float32),
            np.tile(np.array([-0.5, 0.5], dtype=np.float32), 8000),
            np.ones(8000, dtype=np.float32),
        )
    )
    _write_wav(audio, samples)
    transcribe.transcribe_episode(audio, audio.parent / "result.json", Settings())
    features = read_json(audio.with_suffix(".rms.json"), AudioFeaturesDoc)
    assert features.duration == 2.5
    assert features.rms_per_second == [0.0, 0.5, 1.0]


def test_invalid_timestamps_are_dropped_or_clamped(audio: Path, model: _Model) -> None:
    """Whisper timestamps cannot create reversed or out-of-audio transcript intervals."""
    model.segments = [
        _Segment(-1, 1, "Beginning"),
        _Segment(3, 2, "Reversed"),
        _Segment(float("nan"), 4, "Invalid"),
        _Segment(20, 21, "Beyond audio"),
        _Segment(9, 11, "Ending"),
    ]
    document = transcribe.transcribe_episode(audio, audio.parent / "result.json", Settings())
    assert [(item.start, item.end) for item in document.segments] == [(0.0, 1.0), (9.0, 10.0)]


@pytest.mark.parametrize("kind", ["stereo", "sample_rate", "empty", "nonfinite"])
def test_invalid_wav_is_rejected_before_whisper(audio: Path, model: _Model, kind: str) -> None:
    """Audio validation prevents invalid model input and misleading feature files."""
    samples = np.zeros((16000, 2) if kind == "stereo" else 16000, dtype=np.float32)
    if kind == "empty":
        samples = np.zeros(0, dtype=np.float32)
    if kind == "nonfinite":
        samples[0] = np.nan
    _write_wav(audio, samples, sample_rate=8000 if kind == "sample_rate" else 16000)
    with pytest.raises(ValueError):
        transcribe.transcribe_episode(audio, audio.parent / "result.json", Settings())
    assert model.calls == []
    assert not audio.with_suffix(".rms.json").exists()


def test_generator_failure_does_not_publish_transcript(audio: Path, model: _Model) -> None:
    """Lazy model errors are raised before either result is committed."""
    model.fail_on_call = 1
    output = audio.parent / "result.json"
    with pytest.raises(RuntimeError, match="decoding failure"):
        transcribe.transcribe_episode(audio, output, Settings())
    assert not output.exists()
    assert not audio.with_suffix(".rms.json").exists()


def test_model_singleton_is_reused_and_replaced_for_new_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One model is loaded per matching configuration, without a growing cache of models."""
    loads: list[tuple[str, str, str]] = []

    def create(cfg: Settings) -> _Model:
        loads.append((cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute))
        return _Model()

    monkeypatch.setattr(transcribe, "_create_model", create)
    first = transcribe._get_model(Settings())
    assert transcribe._get_model(Settings()) is first
    assert transcribe._get_model(Settings(whisper_model="tiny")) is not first
    assert loads == [("small", "auto", "auto"), ("tiny", "auto", "auto")]


def test_audio_extraction_uses_pcm_mono_and_atomic_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FFmpeg is passed paths as arguments and writes a temporary WAV before replacement."""
    commands: list[list[str]] = []

    def ffmpeg(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        _write_wav(Path(args[-1]), np.zeros(16000, dtype=np.float32))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(transcribe.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(transcribe.subprocess, "run", ffmpeg)
    destination = tmp_path / "audio/E01.wav"
    source = tmp_path / "episode with spaces.mp4"
    assert transcribe.extract_audio(source, destination) == destination
    command = commands[0]
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[-1] != str(destination)
    assert list(destination.parent.iterdir()) == [destination]


def test_failed_extraction_preserves_previous_wav_and_cleans_temporary_file(
    audio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extraction error cannot replace a previously valid WAV with a partial output."""
    previous = audio.read_bytes()

    def fail(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(args[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, args, stderr="no audio stream")

    monkeypatch.setattr(transcribe.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(transcribe.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="no audio stream"):
        transcribe.extract_audio(Path("episode.mp4"), audio)
    assert audio.read_bytes() == previous
    assert list(audio.parent.iterdir()) == [audio]


def test_missing_ffmpeg_has_actionable_error(audio: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing executable fails before attempting to modify output files."""
    monkeypatch.setattr(transcribe.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="FFmpeg is required"):
        transcribe.extract_audio(Path("episode.mp4"), audio)


def test_mid_season_resume_preserves_completed_episode_and_saved_settings(
    ingested_job: JobState,
    model: _Model,
    extraction_calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After E02 fails, resume processes E02/E03 and keeps E01 plus its original settings."""
    model.fail_on_call = 2
    with pytest.raises(RuntimeError, match="decoding failure"):
        resume_job(ingested_job.job_id)
    failed = get_status(ingested_job.job_id)
    assert failed.stages.transcribe.status == "failed"
    assert failed.stages.transcribe.completed == ["E01"]
    assert failed.stages.transcribe.pending == ["E02", "E03"]
    assert len(failed.errors) == 1
    assert failed.errors[-1].episode == "E02"
    assert failed.errors[-1].stage == "transcribe"
    directory = Path("data/jobs") / failed.job_id
    paths = [
        directory / "audio/E01.wav",
        directory / "audio/E01.rms.json",
        directory / "transcripts/E01.transcript.json",
    ]
    previous = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    model.fail_on_call = None
    resumed = resume_job(failed.job_id)
    assert resumed.stages.transcribe.status == "done"
    assert resumed.stages.transcribe.completed == ["E01", "E02", "E03"]
    assert resumed.stages.transcribe.pending == []
    assert len(resumed.errors) == 1
    assert extraction_calls == ["E01", "E02", "E02", "E03"]
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == previous
    assert transcribe._MODEL[0] == ("tiny", "cpu", "int8")
    for stage in ("scenes", "nlp", "select", "render"):
        assert getattr(resumed.stages, stage).status == "pending"


@pytest.mark.parametrize("artifact", ["transcript", "rms", "wav", "malformed", "nonutf8"])
def test_unfinished_stage_repairs_missing_completed_artifact(
    ingested_job: JobState, model: _Model, extraction_calls: list[str], artifact: str
) -> None:
    """A checkpoint alone cannot cause a missing or corrupt artifact to be silently skipped."""
    completed = resume_job(ingested_job.job_id)
    directory = Path("data/jobs") / completed.job_id
    completed.stages.transcribe.status = "in_progress"
    write_json(directory / "state.json", completed)
    path = {
        "transcript": directory / "transcripts/E01.transcript.json",
        "rms": directory / "audio/E01.rms.json",
        "wav": directory / "audio/E01.wav",
        "malformed": directory / "transcripts/E01.transcript.json",
        "nonutf8": directory / "audio/E01.rms.json",
    }[artifact]
    if artifact == "malformed":
        path.write_text("{not json", encoding="utf-8")
    elif artifact == "nonutf8":
        path.write_bytes(b"\xff\xfeinvalid")
    else:
        path.unlink()
    extraction_calls.clear()
    assert resume_job(completed.job_id).stages.transcribe.status == "done"
    assert extraction_calls == ["E01"]
    assert path.exists()


def test_failed_sidecar_write_does_not_mark_episode_complete(
    ingested_job: JobState,
    model: _Model,
    extraction_calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disk errors during artifact persistence leave the episode pending for retry."""
    original = transcribe.write_json

    def fail_rms(path: Path, document: object) -> None:
        if path.name == "E01.rms.json":
            raise OSError("simulated disk error")
        original(path, document)

    monkeypatch.setattr(transcribe, "write_json", fail_rms)
    with pytest.raises(OSError, match="disk error"):
        resume_job(ingested_job.job_id)
    state = get_status(ingested_job.job_id)
    assert state.stages.transcribe.completed == []
    assert state.stages.transcribe.pending == ["E01", "E02", "E03"]
    assert state.errors[-1].episode == "E01"


def test_completed_transcription_resume_skips_model_and_audio_work(
    ingested_job: JobState, model: _Model, extraction_calls: list[str]
) -> None:
    """Repeated resume of a completed transcription stage is idempotent."""
    completed = resume_job(ingested_job.job_id)
    extraction_calls.clear()
    previous_calls = len(model.calls)
    assert resume_job(completed.job_id) == completed
    assert extraction_calls == []
    assert len(model.calls) == previous_calls


def test_default_cli_runs_transcription(
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    model: _Model,
    extraction_calls: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default CLI runs now stop after transcription while preserving --stage ingest."""
    archive = make_zip([("ep1.mp4", b"one")])
    assert main(["run", "--zip", str(archive)]) == 0
    assert "transcribe done" in capsys.readouterr().out
    assert list_jobs()[0].stages.transcribe.status == "done"
    assert extraction_calls == ["E01"]


@pytest.mark.parametrize("values", [[0.1], [0.1, -0.2], [0.1, float("nan")]])
def test_rms_schema_rejects_inconsistent_or_invalid_features(values: list[float]) -> None:
    """RMS documents require one finite nonnegative value per second, including the tail."""
    with pytest.raises(ValidationError):
        AudioFeaturesDoc(episode="E01", duration=1.5, rms_per_second=values)
