"""Opt-in Phase 2 integration smoke test; the only test allowed to load real Whisper."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf

from sts.schemas import AudioFeaturesDoc, JobState, TranscriptDoc, read_json


@pytest.mark.integration
def test_real_fixture_transcription() -> None:
    """Run the real fixture through the CLI with cached small Whisper on CPU/int8."""
    project = Path(__file__).resolve().parents[1]
    fixture = project / "tests/fixtures/show.zip"
    if not fixture.exists():
        subprocess.run([sys.executable, str(project / "scripts/make_fixture.py")], check=True)
    environment = os.environ.copy()
    environment.update(WHISPER_MODEL="small", WHISPER_DEVICE="cpu", WHISPER_COMPUTE="int8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sts.app.cli",
            "run",
            "--zip",
            str(fixture),
            "--target-minutes",
            "2",
            "--stage",
            "transcribe",
            "--name",
            "Phase 2 smoke",
        ],
        cwd=project,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    match = re.search(r"Job ([0-9a-f]{12}): transcribe done", result.stdout)
    assert match is not None, result.stdout
    directory = project / "data/jobs" / match[1]
    state = read_json(directory / "state.json", JobState)
    assert state.stages.ingest.status == "done"
    assert state.stages.transcribe.status == "done"
    assert state.stages.transcribe.completed == ["E01", "E02", "E03"]
    assert state.stages.transcribe.pending == []
    assert state.errors == []
    for episode in state.stages.transcribe.completed:
        transcript = read_json(
            directory / "transcripts" / f"{episode}.transcript.json", TranscriptDoc
        )
        features = read_json(directory / "audio" / f"{episode}.rms.json", AudioFeaturesDoc)
        audio = sf.info(directory / "audio" / f"{episode}.wav")
        assert transcript.episode == features.episode == episode
        assert (audio.samplerate, audio.channels) == (16000, 1)
        assert 89.9 <= features.duration <= 90.1
        assert len(features.rms_per_second) in (90, 91)
        assert all(segment.end <= features.duration for segment in transcript.segments)
    for name in ("scenes", "nlp", "select", "render"):
        assert getattr(state.stages, name).status == "pending"
