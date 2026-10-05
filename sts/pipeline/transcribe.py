"""Episode-at-a-time audio extraction, RMS features, and filtered transcription."""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from pydantic import ValidationError

from sts.config import Settings
from sts.pipeline.control import check_stop, run_media
from sts.pipeline.subtitles import extract_subtitles
from sts.schemas import (
    AudioFeaturesDoc,
    ErrorRecord,
    JobState,
    TranscriptDoc,
    TranscriptionProgress,
    TranscriptSegment,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Segment

# The only mutable module state is the currently loaded model singleton.
_MODEL: tuple[tuple[str, str, str], WhisperModel] | None = None


def _prepare_cuda_runtime(cfg: Settings) -> None:
    if sys.platform != "win32" or cfg.whisper_device == "cpu":
        return
    # CTranslate2's backend uses Windows LoadLibrary, which needs these wheels on
    # the process PATH even when Python's add_dll_directory has been called.
    site = Path(sysconfig.get_path("purelib")) / "nvidia"
    directories = [site / vendor / "bin" for vendor in ("cublas", "cuda_nvrtc", "cudnn")]
    available = [str(path) for path in directories if path.is_dir() and any(path.glob("*.dll"))]
    if available:
        existing = os.environ.get("PATH", "").split(os.pathsep)
        known = {value.casefold() for value in existing}
        additions = [value for value in available if value.casefold() not in known]
        if additions:
            os.environ["PATH"] = os.pathsep.join([*additions, *existing])


def _create_model(cfg: Settings) -> WhisperModel:
    _prepare_cuda_runtime(cfg)
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
        if _MODEL is not None:
            release_model()
        _MODEL = (key, _create_model(cfg))
    return _MODEL[1]


def release_model() -> None:
    """Release the completed ASR stage's model, including a registered UI resource cache.

    Freeing GPU memory lets the subsequent local story model use a small graphics card.
    Call only after transcription has finished; active decoder objects must remain alive.
    """
    global _MODEL
    _MODEL = None
    clear = getattr(_create_model, "clear", None)
    if callable(clear):
        clear()


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
        run_media(
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
            audio_path.parent.parent,
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


def _track_segments(
    segments: Iterable[Segment],
    episode: str,
    duration: float,
    progress: Callable[[TranscriptionProgress], None] | None,
    job_dir: Path,
) -> Iterator[Segment]:
    processed = 0.0
    for segment in segments:
        check_stop(job_dir)
        if progress is not None and math.isfinite(segment.end):
            position = max(processed, min(duration, max(0.0, float(segment.end))))
            if position > processed:
                processed = position
                progress(
                    TranscriptionProgress(
                        episode=episode,
                        activity="decoding",
                        duration=duration,
                        processed_seconds=processed,
                    )
                )
        yield segment


def transcribe_episode(
    audio_path: Path,
    out_path: Path,
    cfg: Settings,
    *,
    progress: Callable[[TranscriptionProgress], None] | None = None,
) -> TranscriptDoc:
    """Decode a WAV once, transcribe it, and atomically save transcript and RMS JSON.

    The RMS sidecar is written beside the WAV as E##.rms.json. Empty transcripts
    are valid. The optional callback reports live decoding position while immediately
    materializing Whisper's generator. An episode is committed only after both files exist.
    """
    wav, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if sample_rate != 16000 or wav.ndim != 1 or len(wav) == 0:
        raise ValueError("Transcription requires nonempty 16 kHz mono audio")
    if not np.isfinite(wav).all():
        raise ValueError("Audio samples must all be finite")
    features = _rms_features(audio_path.stem, wav, sample_rate)
    if progress is not None:
        progress(
            TranscriptionProgress(
                episode=audio_path.stem, activity="loading_model", duration=features.duration
            )
        )
    check_stop(audio_path.parent.parent)
    model = _get_model(cfg)
    check_stop(audio_path.parent.parent)
    if progress is not None:
        progress(
            TranscriptionProgress(
                episode=audio_path.stem, activity="decoding", duration=features.duration
            )
        )
    segments, info = model.transcribe(
        wav, word_timestamps=False, vad_filter=True, condition_on_previous_text=False
    )
    # Materialize immediately: progress reports do not defer decoding or publish partial dialogue.
    segments = list(
        _track_segments(
            segments, audio_path.stem, features.duration, progress, audio_path.parent.parent
        )
    )
    document = TranscriptDoc(
        episode=audio_path.stem,
        language=info.language,
        source="whisper",
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


def _save_subtitle_episode(audio_path: Path, out_path: Path, document: TranscriptDoc) -> None:
    wav, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if sample_rate != 16000 or wav.ndim != 1 or len(wav) == 0 or not np.isfinite(wav).all():
        raise ValueError("Subtitle features require nonempty finite 16 kHz mono audio")
    features = _rms_features(audio_path.stem, wav, sample_rate)
    document = TranscriptDoc(
        episode=document.episode,
        language=document.language,
        source="embedded_subtitles",
        segments=[
            TranscriptSegment(
                start=segment.start, end=min(segment.end, features.duration), text=segment.text
            )
            for segment in document.segments
            if segment.start < min(segment.end, features.duration)
        ],
    )
    write_json(audio_path.with_suffix(".rms.json"), features)
    write_json(out_path, document)


def _completed_artifacts_valid(job_dir: Path, episode: str, cfg: Settings) -> bool:
    try:
        transcript = read_json(
            job_dir / "transcripts" / f"{episode}.transcript.json", TranscriptDoc
        )
        features = read_json(job_dir / "audio" / f"{episode}.rms.json", AudioFeaturesDoc)
        audio = sf.info(job_dir / "audio" / f"{episode}.wav")
        return (
            transcript.episode == episode
            and (
                cfg.transcript_source == "auto"
                or cfg.transcript_source == "subtitles"
                and transcript.source == "embedded_subtitles"
                or cfg.transcript_source == "whisper"
                and transcript.source != "embedded_subtitles"
            )
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
    """Import timed subtitles or transcribe unfinished episodes, checkpointing each success."""
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
        stage.current = None
        write_json(state_path, state)
        logger.info("transcribe started: %s", state.job_id)
        if state.stages.ingest.status != "done" or not state.stages.ingest.episodes:
            raise ValueError("Transcription requires completed ingestion and an episode manifest")
        cfg = Settings(
            whisper_model=state.params.whisper_model,
            whisper_device=state.params.whisper_device,
            whisper_compute=state.params.whisper_compute,
            transcript_source=state.params.transcript_source,
            subtitle_language=state.params.subtitle_language,
        )
        episodes = sorted(state.stages.ingest.episodes, key=lambda item: item.episode_number)
        planned = [item.episode for item in episodes]
        stage.completed = [
            identifier
            for identifier in planned
            if identifier in stage.completed
            and _completed_artifacts_valid(job_dir, identifier, cfg)
        ]
        stage.pending = [identifier for identifier in planned if identifier not in stage.completed]
        write_json(state_path, state)
        for item in episodes:
            episode = item.episode
            if episode in stage.completed:
                logger.info("transcribe unit-progress: %s already complete", episode)
                continue
            check_stop(job_dir)
            subtitles = None
            if cfg.transcript_source != "whisper":
                stage.current = TranscriptionProgress(
                    episode=episode, activity="extracting_subtitles", duration=item.duration
                )
                write_json(state_path, state)
                subtitles = extract_subtitles(
                    item.source_path, episode, item.duration, language=cfg.subtitle_language
                )
                if subtitles is None and cfg.transcript_source == "subtitles":
                    raise ValueError(
                        f"No usable {cfg.subtitle_language} text subtitles found for {episode}. "
                        "Choose Automatic or Whisper to transcribe its audio."
                    )
            logger.info("transcribe unit-progress: %s extracting audio", episode)
            stage.current = TranscriptionProgress(
                episode=episode, activity="extracting_audio", duration=item.duration
            )
            write_json(state_path, state)
            audio_path = job_dir / "audio" / f"{episode}.wav"
            extract_audio(item.source_path, audio_path)
            last_write = float("-inf")

            def update_progress(current: TranscriptionProgress) -> None:
                nonlocal last_write
                check_stop(job_dir)
                changed_activity = (
                    stage.current is None or stage.current.activity != current.activity
                )
                stage.current = current
                now = monotonic()
                if changed_activity or now - last_write >= 5.0:
                    write_json(state_path, state)
                    last_write = now

            out_path = job_dir / "transcripts" / f"{episode}.transcript.json"
            if subtitles is not None:
                document, stream_index = subtitles
                _save_subtitle_episode(audio_path, out_path, document)
                logger.info(
                    "transcribe unit-progress: %s imported %d subtitle cues from stream %d",
                    episode,
                    len(document.segments),
                    stream_index,
                )
            else:
                transcribe_episode(audio_path, out_path, cfg, progress=update_progress)
            completed = set(stage.completed) | {episode}
            stage.completed = [identifier for identifier in planned if identifier in completed]
            stage.pending = [identifier for identifier in planned if identifier not in completed]
            stage.current = None
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
        release_model()
        return state
    except Exception as exc:
        stage.status = "failed"
        state.errors.append(ErrorRecord(stage="transcribe", episode=episode, message=str(exc)))
        write_json(state_path, state)
        logger.exception("transcribe failed: %s", state.job_id)
        raise
