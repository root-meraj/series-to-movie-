"""Configuration defaults, .env parsing, validation, and secret handling."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sts.config import Settings


def test_defaults_match_specification() -> None:
    """The default settings match the project contract."""
    settings = Settings()
    assert (settings.whisper_model, settings.whisper_device, settings.whisper_compute) == (
        "small",
        "auto",
        "auto",
    )
    assert settings.target_minutes == 120.0
    assert settings.detector_threshold == 27.0
    assert settings.score_weights == (0.5, 0.3, 0.1, 0.1)
    assert settings.max_characters == 12
    assert settings.openai_api_key.get_secret_value() == ""
    assert settings.enable_openai is False
    assert settings.season_number == 1
    assert settings.transcript_source == "auto"
    assert settings.subtitle_language == "en"


def test_dotenv_and_nonsecret_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Comma-separated weights load from .env; process settings can override defaults."""
    (tmp_path / ".env").write_text(
        "TARGET_MINUTES=8\nWHISPER_MODEL=base\nSCORE_WEIGHTS=0.6, 0.2, 0.1, 0.1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TARGET_MINUTES", "4")
    settings = Settings()
    assert settings.target_minutes == 4.0
    assert settings.whisper_model == "base"
    assert settings.score_weights == (0.6, 0.2, 0.1, 0.1)


def test_secret_only_comes_from_dotenv_and_is_not_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credentials cannot leak through a settings dump or environment override."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-process-value")
    assert Settings().openai_api_key.get_secret_value() == ""
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dummy-file-value\n", encoding="utf-8")
    settings = Settings()
    assert settings.openai_api_key.get_secret_value() == "dummy-file-value"
    assert "openai_api_key" not in settings.model_dump()
    assert "dummy-file-value" not in repr(settings)


@pytest.mark.parametrize(
    "setting",
    [
        "TARGET_MINUTES=0",
        "TARGET_MINUTES=nan",
        "MIN_CLIP_SECONDS=-1",
        "WHISPER_DEVICE=bogus",
        "TRANSCRIPT_SOURCE=bogus",
        "SUBTITLE_LANGUAGE=English",
        "LLM_PROVIDER=bogus",
        "ENABLE_OPENAI=bogus",
        "SEASON_NUMBER=0",
        "OLLAMA_BASE_URL=file:///tmp/model",
        "OLLAMA_BASE_URL=http://user:password@localhost:11434",
        "MAX_CHARACTERS=0",
        "SCORE_WEIGHTS=0,0,0,0",
        "SCORE_WEIGHTS=1,2,3",
        "SCORE_WEIGHTS=-1,1,1,1",
        "SCORE_WEIGHTS=nan,1,1,1",
        "SCORE_WEIGHTS=inf,1,1,1",
    ],
)
def test_invalid_settings_fail_before_pipeline_work(tmp_path: Path, setting: str) -> None:
    """Invalid numeric values, choices, and weight vectors fail validation."""
    (tmp_path / ".env").write_text(setting, encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings()
