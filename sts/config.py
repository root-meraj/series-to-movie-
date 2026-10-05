"""Validated settings loaded from the working directory's .env file."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """Runtime configuration; model loading and API calls happen in later stages."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", allow_inf_nan=False
    )

    whisper_model: Literal["tiny", "base", "small", "medium"] = "small"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute: Literal["auto", "int8", "float16"] = "auto"
    transcript_source: Literal["auto", "whisper", "subtitles"] = "auto"
    subtitle_language: str = Field(default="en", min_length=2, max_length=3, pattern=r"^[a-z]+$")
    detector_threshold: float = Field(default=27.0, gt=0)
    min_scene_seconds: float = Field(default=5.0, gt=0)
    target_minutes: float = Field(default=120.0, gt=0)
    min_clip_seconds: float = Field(default=15.0, gt=0)
    llm_provider: Literal["openai", "ollama", "mock"] = "openai"
    llm_model: str = Field(default="gpt-4o-mini", min_length=1)
    enable_openai: bool = False
    season_number: int = Field(default=1, ge=1)
    ollama_base_url: str = Field(default="http://localhost:11434", min_length=1)
    openai_api_key: SecretStr = Field(default=SecretStr(""), exclude=True, repr=False)
    sbert_model: str = Field(default="all-MiniLM-L6-v2", min_length=1)
    score_weights: Annotated[tuple[float, float, float, float], NoDecode] = (0.5, 0.3, 0.1, 0.1)
    max_characters: int = Field(default=12, gt=0)

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        """Require an HTTP endpoint without embedded credentials or URL parameters."""
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "OLLAMA_BASE_URL must be an HTTP(S) URL without credentials or parameters"
            )
        return value.rstrip("/")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource | Callable[[], dict[str, Any]], ...]:
        """Read API secrets only from .env; allow environment overrides for other fields."""

        def without_secret(source: PydanticBaseSettingsSource) -> dict[str, Any]:
            values = source()
            values.pop("openai_api_key", None)
            return values

        return (
            lambda: without_secret(init_settings),
            lambda: without_secret(env_settings),
            dotenv_settings,
        )

    @field_validator("score_weights", mode="before")
    @classmethod
    def parse_score_weights(cls, value: object) -> object:
        """Accept the contract's comma-separated .env weights."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(","))
        return value

    @field_validator("score_weights")
    @classmethod
    def validate_score_weights(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Require four finite, nonnegative weights with a positive total."""
        if any(not math.isfinite(weight) or weight < 0 for weight in value) or sum(value) <= 0:
            raise ValueError("score_weights must be finite, nonnegative, and have a positive sum")
        return value
