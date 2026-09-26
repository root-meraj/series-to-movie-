"""Episode-at-a-time audio extraction, RMS features, and filtered transcription."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from pydantic import ValidationError

from sts.config import Settings
from sts.schemas import (
    AudioFeaturesDoc,
    ErrorRecord,
    JobState,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Segment

# The only mutable module state is the currently loaded model singleton.
_MODEL: tuple[tuple[str, str, str], WhisperModel] | None = None


def _create_model(cfg: Settings) -> WhisperModel:
    from faster_whisper import WhisperModel

    return WhisperModel(
        cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute
    )


def _get_model(cfg: Settings) -> WhisperModel:
    global _MODEL
    key = (cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute)
    if _MODEL is None or _MODEL[0] != key:
        logging.getLogger(__name__).info(
            "transcribe unit-progress: loading Whisper %s (%s/%s)", *key
        )
        # Release an old model before allocating its replacement.
        _MODEL = None
        _MODEL = (key, _create_model(cfg))
    return _MODEL[1]


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Atomically extract the first audio stream as 16 kHz mono PCM WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required; install FFmpeg and add its bin directory to PATH")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=audio_path.parent, prefix=f".{audio_path.stem}.", suffix=".wav", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        metadata = sf.info(temporary_path)
        if metadata.samplerate != 16000 or metadata.channels != 1 or metadata.frames <= 0:
            raise ValueError("Extracted audio must be nonempty 16 kHz mono WAV")
        temporary_path.replace(audio_path)
        return audio_path
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"Audio extraction failed for {video_path.name}: {detail}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _rms_features(episode: str, wav: NDArray[np.float32], sample_rate: int) -> AudioFeaturesDoc:
    values = [
        float(np.sqrt(np.mean(np.square(wav[start : start + sample_rate], dtype=np.float64))))
        for start in range(0, len(wav), sample_rate)
    ]
    return AudioFeaturesDoc(
        episode=episode,
        sample_rate=sample_rate,
        duration=float(len(wav) / sample_rate),
        rms_per_second=values,
    )


def _filter_segments(segments: list[Segment], duration: float) -> list[TranscriptSegment]:
    accepted: list[TranscriptSegment] = []
    previous_text: str | None = None
    for segment in segments:
        if not all(
            math.isfinite(value)
            for value in (segment.start, segment.end, segment.no_speech_prob, segment.avg_logprob)
        ):
            continue
        if segment.no_speech_prob > 0.6 or segment.avg_logprob < -1.0:
            continue
        text = " ".join(segment.text.split())
        if not text or text.casefold() == previous_text:
            continue
        start, end = max(0.0, float(segment.start)), min(duration, float(segment.end))
        if end <= start:
            continue
        accepted.append(TranscriptSegment(start=start, end=end, text=text))
        previous_text = text.casefold()
    return accepted


def transcribe_episode(audio_path: Path, out_path: Path, cfg: Settings) -> TranscriptDoc:
    """Decode a WAV once, transcribe it, and atomically save transcript and RMS JSON.

    The RMS sidecar is written beside the WAV as E##.rms.json. Empty transcripts
    are valid. Job progress is committed by transcribe_job after both files exist.
    """
    wav, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if sample_rate != 16000 or wav.ndim != 1 or len(wav) == 0:
        raise ValueError("Transcription requires nonempty 16 kHz mono audio")
    if not np.isfinite(wav).all():
        raise ValueError("Audio samples must all be finite")
    features = _rms_features(audio_path.stem, wav, sample_rate)
    model = _get_model(cfg)
    segments, info = model.transcribe(
        wav, word_timestamps=False, vad_filter=True, condition_on_previous_text=False
    )
    segments = list(segments)  # Materialize immediately: decoding errors occur during iteration.
    document = TranscriptDoc(
        episode=audio_path.stem,
        language=info.language,
        segments=_filter_segments(segments, features.duration),
    )
    write_json(audio_path.with_suffix(".rms.json"), features)
    write_json(out_path, document)
    logging.getLogger(__name__).info(
        "transcribe unit-progress: %s kept %d/%d segments",
        document.episode,
        len(document.segments),
        len(segments),
    )
    return document


def _completed_artifacts_valid(job_dir: Path, episode: str) -> bool:
    try:
        transcript = read_json(
            job_dir / "transcripts" / f"{episode}.transcript.json", TranscriptDoc
        )
        features = read_json(job_dir / "audio" / f"{episode}.rms.json", AudioFeaturesDoc)
        audio = sf.info(job_dir / "audio" / f"{episode}.wav")
        return (
            transcript.episode == episode
            and features.episode == episode
            and audio.samplerate == 16000
            and audio.channels == 1
            and audio.frames > 0
            and math.isclose(features.duration, audio.frames / audio.samplerate, abs_tol=1e-6)
            and all(segment.end <= features.duration + 1e-6 for segment in transcript.segments)
        )
    except (OSError, RuntimeError, UnicodeError, ValidationError):
        return False


def transcribe_job(job_dir: Path) -> JobState:
    """Transcribe unfinished episodes using saved settings and checkpoint each success."""
    logger = logging.getLogger(__name__)
    state_path = job_dir / "state.json"
    state = read_json(state_path, JobState)
    stage = state.stages.transcribe
    if stage.status == "done":
        logger.info("transcribe done: skipping completed stage for %s", state.job_id)
        return state
    episode: str | None = None
    try:
        stage.status = "in_progress"
        write_json(state_path, state)
        logger.info("transcribe started: %s", state.job_id)
        if state.stages.ingest.status != "done" or not state.stages.ingest.episodes:
            raise ValueError("Transcription requires completed ingestion and an episode manifest")
        cfg = Settings(
            whisper_model=state.params.whisper_model,
            whisper_device=state.params.whisper_device,
            whisper_compute=state.params.whisper_compute,
        )
        episodes = sorted(state.stages.ingest.episodes, key=lambda item: item.episode_number)
        planned = [item.episode for item in episodes]
        stage.completed = [
            identifier
            for identifier in planned
            if identifier in stage.completed and _completed_artifacts_valid(job_dir, identifier)
        ]
        stage.pending = [identifier for identifier in planned if identifier not in stage.completed]
        write_json(state_path, state)
        for item in episodes:
            episode = item.episode
            if episode in stage.completed:
                logger.info("transcribe unit-progress: %s already complete", episode)
                continue
            logger.info("transcribe unit-progress: %s extracting audio", episode)
            audio_path = job_dir / "audio" / f"{episode}.wav"
            extract_audio(item.source_path, audio_path)
            transcribe_episode(
                audio_path, job_dir / "transcripts" / f"{episode}.transcript.json", cfg
            )
            completed = set(stage.completed) | {episode}
            stage.completed = [identifier for identifier in planned if identifier in completed]
            stage.pending = [identifier for identifier in planned if identifier not in completed]
            write_json(state_path, state)
            logger.info(
                "transcribe unit-progress: %s done (%d/%d)",
                episode,
                len(stage.completed),
                len(planned),
            )
        stage.status = "done"
        write_json(state_path, state)
        logger.info("transcribe done: %d episodes", len(stage.completed))
        return state
    except Exception as exc:
        stage.status = "failed"
        state.errors.append(ErrorRecord(stage="transcribe", episode=episode, message=str(exc)))
        write_json(state_path, state)
        logger.exception("transcribe failed: %s", state.job_id)
        raise
