"""Validated JSON contracts and atomic persistence shared by pipeline stages."""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

Seconds = Annotated[float, Field(ge=0)]
UnitScore = Annotated[float, Field(ge=0, le=1)]
EpisodeId = Annotated[str, Field(pattern=r"^E\d{2,}$")]
SceneId = Annotated[str, Field(pattern=r"^E\d{2,}_s\d{4,}$")]
StageName = Literal["ingest", "transcribe", "scenes", "nlp", "select", "render"]
StageStatus = Literal["pending", "in_progress", "done", "failed"]
Document = TypeVar("Document", bound=BaseModel)


def utc_now() -> datetime:
    """Return an aware UTC timestamp for job creation and error events."""
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    """Reject unknown contract keys and non-finite numeric values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class TimedRange(ContractModel):
    """A positive-length media interval expressed in float seconds."""

    start: Seconds
    end: Seconds

    @model_validator(mode="after")
    def validate_interval(self) -> TimedRange:
        """Reject reversed and zero-length intervals."""
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class TranscriptSegment(TimedRange):
    """One timestamped transcript segment."""

    text: str


class TranscriptDoc(ContractModel):
    """The transcripts/E##.transcript.json contract; empty segments are valid."""

    episode: EpisodeId
    language: str = Field(min_length=1)
    segments: list[TranscriptSegment] = Field(default_factory=list)


class AudioFeaturesDoc(ContractModel):
    """The audio/E##.rms.json contract, with one RMS value per second of audio."""

    episode: EpisodeId
    sample_rate: Literal[16000] = 16000
    duration: float = Field(gt=0)
    rms_per_second: list[Seconds]

    @model_validator(mode="after")
    def validate_rms_count(self) -> AudioFeaturesDoc:
        """Include a final partial second, without padding its RMS calculation."""
        if len(self.rms_per_second) != math.ceil(self.duration):
            raise ValueError("rms_per_second must contain ceil(duration) values")
        return self


class SceneRecord(TimedRange):
    """A detected scene enriched with aligned dialogue and audio features."""

    scene_id: SceneId
    episode: EpisodeId
    source_path: Path
    duration: float = Field(gt=0)
    transcript: str = ""
    dialogue_ratio: UnitScore = 0.0
    audio_rms: Seconds = 0.0
    characters: list[str] = Field(default_factory=list)
    is_first: bool = False
    is_last: bool = False
    episode_number: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_scene(self) -> SceneRecord:
        """Keep duration and episode identity consistent with the scene bounds."""
        if not math.isclose(self.duration, self.end - self.start, abs_tol=0.001):
            raise ValueError("duration must equal end - start")
        if not self.scene_id.startswith(f"{self.episode}_"):
            raise ValueError("scene_id must belong to episode")
        if int(self.episode[1:]) != self.episode_number:
            raise ValueError("episode_number must match episode")
        return self


class ScenesDoc(RootModel[list[SceneRecord]]):
    """The scenes.json top-level array."""


class CharacterRecord(ContractModel):
    """One canonical character with aliases and mention statistics."""

    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    mention_count: int = Field(ge=0)
    is_main: bool


class CharacterTable(ContractModel):
    """The characters.json contract."""

    characters: list[CharacterRecord] = Field(default_factory=list)


class CharacterMention(ContractModel):
    """An unmerged PERSON surface form and its per-episode frequency."""

    name: str = Field(min_length=1)
    mention_count: int = Field(ge=1)


class CharacterMentionsDoc(ContractModel):
    """The transcripts/E##.characters.json checkpoint before season-wide deduplication."""

    episode: EpisodeId
    mentions: list[CharacterMention] = Field(default_factory=list)


class SceneEmbeddingRecord(ContractModel):
    """A normalized embedding for a scene with nonblank cleaned dialogue."""

    scene_id: SceneId
    vector: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_normalization(self) -> SceneEmbeddingRecord:
        """Reject invalid cached vectors before they enter the similarity graph."""
        if not math.isclose(math.hypot(*self.vector), 1.0, abs_tol=1e-5):
            raise ValueError("Cached scene embeddings must have unit length")
        return self


class SceneEmbeddingsDoc(ContractModel):
    """The transcripts/E##.embeddings.json checkpoint used by baseline scoring."""

    episode: EpisodeId
    model_name: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings: list[SceneEmbeddingRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_records(self) -> SceneEmbeddingsDoc:
        """Require unique scenes from one episode and a consistent vector dimension."""
        identifiers = [record.scene_id for record in self.embeddings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Cached embeddings must have unique scene IDs")
        if any(not identifier.startswith(f"{self.episode}_") for identifier in identifiers):
            raise ValueError("Cached embeddings must belong to the checkpoint episode")
        if len({len(record.vector) for record in self.embeddings}) > 1:
            raise ValueError("Cached embedding dimensions must match")
        return self


class ScoreRecord(ContractModel):
    """Scene scores, with LLM fields omitted until the LLM stage runs."""

    scene_id: SceneId
    baseline_score: UnitScore = 0.0
    llm_score: Annotated[int, Field(ge=1, le=10)] | None = None
    llm_plot_advance: bool | None = None
    llm_reason: str | None = None
    final_score: UnitScore = 0.0

    @model_validator(mode="after")
    def validate_reason(self) -> ScoreRecord:
        """Limit the LLM explanation to the specified 15 words."""
        if self.llm_reason is not None and len(self.llm_reason.split()) > 15:
            raise ValueError("llm_reason must contain at most 15 words")
        return self


class ScoresDoc(RootModel[list[ScoreRecord]]):
    """The scores.json top-level array."""


class SelectedClip(TimedRange):
    """An interval selected for the condensed movie."""

    scene_id: SceneId
    episode: EpisodeId
    source_path: Path


class SelectionStats(ContractModel):
    """Counts and source runtime reported by scene selection."""

    scenes_kept: int = Field(ge=0)
    scenes_dropped: int = Field(ge=0)
    original_seconds: Seconds


class SelectionDoc(ContractModel):
    """The selected.json contract, including checked runtime totals."""

    target_seconds: float = Field(gt=0)
    selected_total_seconds: Seconds
    clips: list[SelectedClip] = Field(default_factory=list)
    stats: SelectionStats

    @model_validator(mode="after")
    def validate_selection(self) -> SelectionDoc:
        """Ensure reported runtime and clip counts agree with the selected intervals."""
        total = sum(clip.end - clip.start for clip in self.clips)
        if not math.isclose(self.selected_total_seconds, total, abs_tol=0.001):
            raise ValueError("selected_total_seconds must equal the sum of clip durations")
        if total > self.target_seconds + 1e-9:
            raise ValueError("selected clips exceed target_seconds")
        if self.stats.scenes_kept != len(self.clips):
            raise ValueError("scenes_kept must equal the number of clips")
        return self


class EpisodeInfo(ContractModel):
    """A validated episode, persisted in the ingestion checkpoint manifest."""

    episode: EpisodeId
    episode_number: int = Field(ge=1)
    source_path: Path
    original_name: str = Field(min_length=1)
    duration: float = Field(ge=60)


class JobParams(ContractModel):
    """Persisted run parameters, excluding credentials."""

    target_seconds: float = Field(default=7200.0, gt=0)
    whisper_model: Literal["tiny", "base", "small", "medium"] = "small"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute: Literal["auto", "int8", "float16"] = "auto"
    detector_threshold: float = Field(default=27.0, gt=0)
    min_scene_seconds: float = Field(default=5.0, gt=0)
    max_characters: int = Field(default=12, gt=0)
    sbert_model: str = Field(default="all-MiniLM-L6-v2", min_length=1)
    zip_path: Path | None = None
    zip_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    name: str | None = None


class StageState(ContractModel):
    """Status of a pipeline stage."""

    status: StageStatus = "pending"


class EpisodeStageState(StageState):
    """Per-episode checkpoints for resumable stages."""

    completed: list[EpisodeId] = Field(default_factory=list)
    pending: list[EpisodeId] = Field(default_factory=list)


class IngestState(EpisodeStageState):
    """Ingestion progress and the manifest of validated episode files."""

    episode_count: int = Field(default=0, ge=0)
    episodes: list[EpisodeInfo] = Field(default_factory=list)


class NLPState(StageState):
    """NLP progress; character preparation precedes the later scoring phases."""

    characters: EpisodeStageState = Field(default_factory=EpisodeStageState)
    baseline: EpisodeStageState = Field(default_factory=EpisodeStageState)


class JobStages(ContractModel):
    """The six stages, in pipeline execution order."""

    ingest: IngestState = Field(default_factory=IngestState)
    transcribe: EpisodeStageState = Field(default_factory=EpisodeStageState)
    scenes: EpisodeStageState = Field(default_factory=EpisodeStageState)
    nlp: NLPState = Field(default_factory=NLPState)
    select: StageState = Field(default_factory=StageState)
    render: StageState = Field(default_factory=StageState)


class ErrorRecord(ContractModel):
    """A persisted stage failure; earlier errors remain after successful resume."""

    stage: StageName
    episode: EpisodeId | None = None
    message: str
    ts: datetime = Field(default_factory=utc_now)


class JobState(ContractModel):
    """The state.json contract and all resumable stage checkpoints."""

    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    created_at: datetime = Field(default_factory=utc_now)
    params: JobParams = Field(default_factory=JobParams)
    stages: JobStages = Field(default_factory=JobStages)
    errors: list[ErrorRecord] = Field(default_factory=list)


def read_json(path: Path, document_type: type[Document]) -> Document:
    """Read and validate one JSON document against its contract."""
    return document_type.model_validate_json(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: BaseModel) -> None:
    """Atomically replace a JSON file after flushing a temporary sibling to disk."""
    payload = document.model_dump_json(indent=2, exclude_none=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                temporary_path.replace(path)
                break
            except PermissionError as exc:
                # Windows indexers/readers can briefly hold the destination open.
                # Never unlink it: the previous checkpoint must remain intact.
                if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                    raise
                time.sleep(0.05 * 2**attempt)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
