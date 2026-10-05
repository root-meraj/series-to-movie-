"""Shared fixtures keep unit tests independent of local settings and FFmpeg."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from sts.app import model_cache
from sts.config import Settings
from sts.nlp import embeddings, ner, scorer_llm
from sts.pipeline import transcribe


def pytest_addoption(parser: pytest.Parser) -> None:
    """Keep integration smoke tests using trained models opt-in."""
    parser.addoption("--run-model-smoke", action="store_true", help="Run trained-model smoke tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests requiring downloaded models unless explicitly requested."""
    if not config.getoption("--run-model-smoke"):
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(pytest.mark.skip(reason="use --run-model-smoke for trained models"))


@pytest.fixture(autouse=True)
def prevent_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block every external LLM transport, including during opt-in local-model smoke tests."""

    def fail_transport(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must use MockProvider; external LLM requests are forbidden")

    monkeypatch.setattr(scorer_llm, "_create_openai_client", fail_transport)
    monkeypatch.setattr(scorer_llm, "_post_ollama", fail_transport)


@pytest.fixture(autouse=True)
def prevent_real_models(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make accidental model loading an immediate failure in every unit test."""
    if request.node.get_closest_marker("integration"):
        return

    def fail_model(cfg: Settings) -> None:
        raise AssertionError("Unit tests must mock Whisper; real model loading is forbidden")

    def fail_ner() -> None:
        raise AssertionError("Unit tests must mock trained spaCy NER")

    def fail_embedding(name: str) -> None:
        raise AssertionError("Unit tests must mock SentenceTransformer")

    monkeypatch.setattr(transcribe, "_MODEL", None)
    # Model-focused unit tests use synthetic video bytes. Subtitle tests replace
    # this seam explicitly or exercise the extractor module with their own media doubles.
    monkeypatch.setattr(transcribe, "extract_subtitles", lambda *args, **kwargs: None)
    monkeypatch.setattr(transcribe, "_create_model", fail_model)
    monkeypatch.setattr(ner, "_NLP", None)
    monkeypatch.setattr(ner, "_create_nlp", fail_ner)
    monkeypatch.setattr(embeddings, "_MODEL", None)
    monkeypatch.setattr(embeddings, "_create_model", fail_embedding)
    monkeypatch.setattr(model_cache, "_WHISPER_FACTORY", fail_model)
    monkeypatch.setattr(model_cache, "_NER_FACTORY", fail_ner)
    monkeypatch.setattr(model_cache, "_SBERT_FACTORY", fail_embedding)


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent developer .env files and process settings from affecting unit tests."""
    monkeypatch.chdir(tmp_path)
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
        monkeypatch.delenv(field.lower(), raising=False)


@pytest.fixture
def probe_calls(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Replace FFprobe with a deterministic probe that records each file's bytes."""
    calls: list[bytes] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(Path(args[-1]).read_bytes())
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "90.0"}}),
            "",
        )

    monkeypatch.setattr("sts.pipeline.ingest.shutil.which", lambda name: "ffprobe")
    monkeypatch.setattr("sts.pipeline.ingest.subprocess.run", fake_run)
    return calls


@pytest.fixture
def make_zip(tmp_path: Path) -> Callable[[list[tuple[str, bytes]]], Path]:
    """Create a small ZIP with ordered names and distinctive stand-in video bytes."""
    from zipfile import ZipFile

    def create(entries: list[tuple[str, bytes]]) -> Path:
        path = tmp_path / "show.zip"
        with ZipFile(path, "w") as archive:
            for name, contents in entries:
                archive.writestr(name, contents)
        return path

    return create
