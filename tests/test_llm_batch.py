"""Strict batch coverage, fewer requests, and durable local-scoring recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sts.config import Settings
from sts.nlp import scorer_llm as llm
from sts.schemas import (
    CharacterTable,
    EpisodeInfo,
    EpisodeLLMDoc,
    JobParams,
    JobState,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    ScoresDoc,
    read_json,
    write_json,
)


def _scene(index: int) -> SceneRecord:
    return SceneRecord(
        scene_id=f"E01_s{index:04d}",
        episode="E01",
        episode_number=1,
        source_path=Path("episode.mkv"),
        start=(index - 1) * 6,
        end=index * 6,
        duration=6,
        transcript=f"The investigator discovers clue {index}.",
    )


def _batch(indices: list[int]) -> str:
    return json.dumps(
        {
            "scores": [
                {
                    "scene_id": f"E01_s{index:04d}",
                    "score": 8,
                    "plot_advance": True,
                    "reason": "Reveals evidence about the central conflict",
                }
                for index in indices
            ]
        }
    )


def test_batch_scoring_reduces_calls_and_preserves_every_scene() -> None:
    """Seventeen blocks use one summary plus three requests, rather than eighteen calls."""
    mock = llm.MockProvider()
    scenes = [_scene(index) for index in range(1, 18)]
    scores, summary = llm.score_scenes(
        scenes, CharacterTable(), "", Settings(llm_batch_size=8), provider=mock
    )
    assert len(mock.calls) == 4 and summary
    assert [item.scene_id for item in scores] == [item.scene_id for item in scenes]


def test_local_batch_schema_requires_each_requested_id_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small model cannot choose extra IDs or end its array before judging every block."""
    mock = llm.MockProvider()
    schemas: list[dict[str, Any]] = []

    def complete(system: str, user: str, schema: dict[str, Any]) -> str:
        schemas.append(schema)
        return _batch([1, 2])

    monkeypatch.setattr(mock, "complete_structured", complete)
    llm._score_batch([_scene(1), _scene(2)], CharacterTable(), "", Settings(), mock, "Show")
    scores = schemas[0]["properties"]["scores"]
    assert scores["minItems"] == scores["maxItems"] == 2
    assert scores["additionalItems"] is False
    assert [item["properties"]["scene_id"]["const"] for item in scores["items"]] == [
        "E01_s0001",
        "E01_s0002",
    ]
    assert all(
        list(item["properties"]) == ["scene_id", "reason", "plot_advance", "score"]
        for item in scores["items"]
    )


@pytest.mark.parametrize("damaged", [[1], [1, 3], [1, 1]])
def test_batch_missing_extra_and_duplicate_ids_get_bounded_retries(damaged: list[int]) -> None:
    """Valid-looking scores cannot silently omit a crucial event or judge a nonexistent one."""
    mock = llm.MockProvider([_batch(damaged), _batch([2, 1])])
    result = llm._score_batch(
        [_scene(1), _scene(2)],
        CharacterTable(),
        "Context.",
        Settings(llm_batch_size=8),
        mock,
        "Show",
    )
    assert [item.scene_id for item in result] == ["E01_s0001", "E01_s0002"]
    assert len(mock.calls) == 2 and "Invalid response" in mock.calls[1][1]


def test_batch_long_reasons_stay_strict_and_do_not_leak_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nested response failures retain the exact word limit and safe retry feedback."""
    damaged = json.loads(_batch([1]))
    damaged["scores"][0]["reason"] = "private-body " * 16
    mock = llm.MockProvider([json.dumps(damaged), _batch([1])])
    llm._score_batch([_scene(1)], CharacterTable(), "", Settings(), mock, "Show")
    assert "1 to 15 words" in mock.calls[1][1]
    assert "private-body" not in caplog.text + mock.calls[1][1]


def test_credit_blocks_never_trigger_scoring_requests() -> None:
    """Credit text cannot affect the episode summary or compete with story footage."""
    scenes = [_scene(1), _scene(2).model_copy(update={"is_credit": True})]
    mock = llm.MockProvider()
    scores, _ = llm.score_scenes(scenes, CharacterTable(), "", Settings(), provider=mock)
    assert len(mock.calls) == 2 and len(scores) == 1
    assert "E01_s0002" not in mock.calls[0][1] + mock.calls[1][1]


def test_batch_failure_resumes_after_last_atomic_batch(tmp_path: Path) -> None:
    """An interrupted second batch reuses the summary and all eight previously saved scores."""
    directory = tmp_path / "batch-job"
    state = JobState(
        job_id="batch-job",
        params=JobParams(llm_provider="mock", llm_model="mock", llm_batch_size=8),
    )
    state.stages.ingest.episodes = [
        EpisodeInfo(
            episode="E01",
            episode_number=1,
            source_path=tmp_path / "E01.mkv",
            original_name="episode.mkv",
            duration=120,
        )
    ]
    state.stages.ingest.episode_count = 1
    for stage in (
        state.stages.ingest,
        state.stages.transcribe,
        state.stages.scenes,
        state.stages.nlp.characters,
        state.stages.nlp.baseline,
    ):
        stage.status = "done"
        stage.completed = ["E01"]
    scenes = [_scene(index) for index in range(1, 18)]
    write_json(directory / "state.json", state)
    write_json(directory / "scenes.json", ScenesDoc(scenes))
    write_json(directory / "characters.json", CharacterTable())
    write_json(
        directory / "scores.json", ScoresDoc([ScoreRecord(scene_id=s.scene_id) for s in scenes])
    )
    mock = llm.MockProvider(
        [
            json.dumps({"summary": "The investigation begins."}),
            _batch(list(range(1, 9))),
            llm.ProviderError("Interrupted"),
        ]
    )
    with pytest.raises(llm.ProviderError):
        llm.build_llm(directory, provider=mock)
    saved = read_json(directory / "transcripts/E01.llm.json", EpisodeLLMDoc)
    assert len(saved.scores) == 8
    resumed = llm.MockProvider([_batch(list(range(9, 17))), _batch([17])])
    completed = llm.build_llm(directory, provider=resumed)
    assert completed.stages.nlp.status == "done" and len(resumed.calls) == 2
    final = read_json(directory / "transcripts/E01.llm.json", EpisodeLLMDoc)
    assert final.summary == saved.summary and final.scores[:8] == saved.scores
