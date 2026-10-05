"""JSON contract validation, top-level shapes, and atomic checkpoint persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sts.schemas import (
    CharacterTable,
    JobState,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    ScoresDoc,
    SelectedClip,
    SelectionDoc,
    SelectionStats,
    TranscriptDoc,
    TranscriptionProgress,
    TranscriptSegment,
    read_json,
    write_json,
)


def test_original_state_contract_round_trips(tmp_path: Path) -> None:
    """The original specification's state payload remains accepted without extension fields."""
    payload = {
        "job_id": "abc123",
        "created_at": "2026-09-26T00:00:00Z",
        "params": {"target_seconds": 7200, "whisper_model": "small", "detector_threshold": 27.0},
        "stages": {
            "ingest": {"status": "done", "episode_count": 10},
            "transcribe": {"status": "in_progress", "completed": ["E01"], "pending": ["E02"]},
            "scenes": {"status": "pending"},
            "nlp": {"status": "pending"},
            "select": {"status": "pending"},
            "render": {"status": "pending"},
        },
        "errors": [
            {
                "stage": "transcribe",
                "episode": "E03",
                "message": "error",
                "ts": "2026-09-26T00:00:01Z",
            }
        ],
    }
    state = JobState.model_validate(payload)
    path = tmp_path / "state.json"
    write_json(path, state)
    assert read_json(path, JobState) == state
    assert isinstance(state.params.target_seconds, float)
    assert state.created_at.utcoffset().total_seconds() == 0
    assert state.stages.transcribe.current is None
    assert state.params.transcript_source == "whisper"
    assert state.params.subtitle_language == "en"


def test_active_transcription_progress_round_trips_without_committing_episode(
    tmp_path: Path,
) -> None:
    """Live decoding positions remain distinct from completed episode checkpoints."""
    state = JobState(
        job_id="active123",
        params={"target_seconds": 120},
        stages={
            "transcribe": {
                "status": "in_progress",
                "completed": ["E01"],
                "pending": ["E02"],
                "current": {
                    "episode": "E02",
                    "activity": "decoding",
                    "duration": 2930.0,
                    "processed_seconds": 1500.25,
                },
            }
        },
    )
    path = tmp_path / "state.json"
    write_json(path, state)
    restored = read_json(path, JobState)
    assert restored == state
    assert restored.stages.transcribe.completed == ["E01"]
    assert restored.stages.transcribe.pending == ["E02"]
    assert restored.stages.transcribe.current.processed_seconds == 1500.25


@pytest.mark.parametrize(
    "changes",
    [
        {"processed_seconds": -1.0},
        {"processed_seconds": 11.0},
        {"processed_seconds": float("nan")},
        {"duration": float("inf")},
        {"activity": "completed"},
        {"episode": "Episode 1"},
    ],
)
def test_invalid_active_transcription_progress_is_rejected(changes: dict[str, object]) -> None:
    """Active activity uses finite, bounded seconds and known episode/phase identifiers."""
    payload = {"episode": "E01", "activity": "decoding", "duration": 10.0, **changes}
    with pytest.raises(ValidationError):
        TranscriptionProgress.model_validate(payload)


def test_scenes_scores_and_characters_keep_contract_shapes(tmp_path: Path) -> None:
    """Scene/score arrays stay arrays and filesystem paths serialize to strings."""
    scene = SceneRecord(
        scene_id="E01_s0042",
        episode="E01",
        source_path=tmp_path / "E01.mp4",
        start=734.2,
        end=795.8,
        duration=61.6,
        transcript="We need to talk about the money.",
        dialogue_ratio=0.83,
        audio_rms=0.41,
        characters=["Walter", "Jesse"],
        episode_number=1,
    )
    scenes = ScenesDoc([scene])
    scores = ScoresDoc(
        [
            ScoreRecord(
                scene_id=scene.scene_id,
                baseline_score=0.63,
                llm_score=8,
                llm_plot_advance=True,
                llm_reason="Introduces the central conflict",
                final_score=0.81,
            )
        ]
    )
    for name, document in (("scenes", scenes), ("scores", scores)):
        path = tmp_path / f"{name}.json"
        write_json(path, document)
        assert isinstance(json.loads(path.read_text()), list)
        assert read_json(path, type(document)) == document
    assert isinstance(json.loads(scenes.model_dump_json())[0]["source_path"], str)
    characters = CharacterTable.model_validate(
        {
            "characters": [
                {
                    "name": "Walter White",
                    "aliases": ["Walter", "Mr. White"],
                    "mention_count": 341,
                    "is_main": True,
                }
            ]
        }
    )
    assert characters.characters[0].aliases == ["Walter", "Mr. White"]


def test_empty_transcript_is_valid() -> None:
    """Music or silence may produce no transcript segments."""
    document = TranscriptDoc(episode="E01", language="en", segments=[])
    assert document.segments == []


@pytest.mark.parametrize(("start", "end"), [(1, 1), (2, 1), (-1, 5), (0, float("inf"))])
def test_invalid_segment_bounds_are_rejected(start: float, end: float) -> None:
    """All media timestamps are finite seconds with positive-length intervals."""
    with pytest.raises(ValidationError):
        TranscriptSegment(start=start, end=end, text="Dialogue")


def test_unknown_keys_and_invalid_scores_are_rejected() -> None:
    """Misspelled contract keys and scores outside their ranges fail immediately."""
    with pytest.raises(ValidationError):
        TranscriptDoc.model_validate({"episode": "E01", "language": "en", "segment": []})
    with pytest.raises(ValidationError):
        ScoreRecord(scene_id="E01_s0001", baseline_score=1.1)
    with pytest.raises(ValidationError):
        ScoreRecord(scene_id="E01_s0001", llm_score=11)


def test_baseline_score_can_be_saved_before_llm_scoring() -> None:
    """Baseline-only scores omit LLM values instead of fabricating scores."""
    score = ScoreRecord(scene_id="E01_s0001", baseline_score=0.5)
    assert "llm_score" not in score.model_dump(exclude_none=True)


@pytest.mark.parametrize(
    ("target", "total", "kept"), [(5.0, 10.0, 1), (20.0, 9.0, 1), (20.0, 10.0, 0)]
)
def test_selection_rejects_inconsistent_totals(target: float, total: float, kept: int) -> None:
    """Selected clips cannot exceed the budget or misreport duration and count."""
    clip = SelectedClip(
        scene_id="E01_s0001", episode="E01", source_path=Path("E01.mp4"), start=0, end=10
    )
    with pytest.raises(ValidationError):
        SelectionDoc(
            target_seconds=target,
            selected_total_seconds=total,
            clips=[clip],
            stats=SelectionStats(scenes_kept=kept, scenes_dropped=0, original_seconds=90),
        )


def test_selection_round_trip() -> None:
    """A valid selection serializes the required clip and statistics keys."""
    selection = SelectionDoc(
        target_seconds=7200,
        selected_total_seconds=45.2,
        clips=[
            SelectedClip(
                scene_id="E01_s0001",
                episode="E01",
                source_path=Path("E01.mp4"),
                start=0.0,
                end=45.2,
            )
        ],
        stats=SelectionStats(scenes_kept=1, scenes_dropped=2, original_seconds=270.0),
    )
    assert SelectionDoc.model_validate_json(selection.model_dump_json()) == selection


def test_failed_atomic_replace_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed state write leaves the last complete JSON intact and cleans its temporary file."""
    path = tmp_path / "state.json"
    original = JobState(job_id="job")
    write_json(path, original)
    previous = path.read_bytes()

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    updated = JobState(job_id="newjob")
    with pytest.raises(OSError, match="simulated"):
        write_json(path, updated)
    assert path.read_bytes() == previous
    assert read_json(path, JobState) == original
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("permanent", [False, True])
def test_windows_checkpoint_lock_has_bounded_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, permanent: bool
) -> None:
    """A transient Windows sharing lock retries; a permanent one preserves the prior state."""
    path = tmp_path / "state.json"
    original = JobState(job_id="original")
    write_json(path, original)
    replace = Path.replace
    attempts: list[Path] = []
    delays: list[float] = []

    def locked_replace(self: Path, target: Path) -> Path:
        attempts.append(self)
        if permanent or len(attempts) == 1:
            error = PermissionError("simulated Windows sharing lock")
            error.winerror = 32
            raise error
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr("sts.schemas.time.sleep", delays.append)
    updated = JobState(job_id="updated")
    if permanent:
        with pytest.raises(PermissionError, match="sharing lock"):
            write_json(path, updated)
        assert read_json(path, JobState) == original
        assert len(attempts) == 5
        assert delays == [0.05, 0.1, 0.2, 0.4]
    else:
        write_json(path, updated)
        assert read_json(path, JobState) == updated
        assert len(attempts) == 2
        assert delays == [0.05]
    assert list(tmp_path.glob("*.tmp")) == []
