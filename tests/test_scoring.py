"""Weighted TextRank graphs, score contracts, and resumable embedding units."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from sts.app.cli import main
from sts.config import Settings
from sts.nlp import scorer_textrank as baseline
from sts.pipeline.controller import get_status, resume_job, run_job
from sts.schemas import (
    CharacterTable,
    JobState,
    SceneEmbeddingRecord,
    SceneEmbeddingsDoc,
    SceneRecord,
    ScenesDoc,
    ScoresDoc,
    read_json,
    write_json,
)


def _scene(index: int, text: str, episode: int = 1) -> SceneRecord:
    return SceneRecord(
        scene_id=f"E{episode:02d}_s{index:04d}",
        episode=f"E{episode:02d}",
        source_path=Path(f"E{episode:02d}.mp4"),
        episode_number=episode,
        start=(index - 1) * 30,
        end=index * 30,
        duration=30,
        transcript=text,
    )


@pytest.fixture
def embedding_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], str | None]]:
    """Provide a weighted star graph: bridge joins two otherwise unrelated dialogue scenes."""
    calls: list[tuple[list[str], str | None]] = []
    vectors = {"bridge": [0.6, 0.8], "left": [1, 0], "right": [0, 1], "changed": [0, -1]}

    def embed(texts: list[str], *, model_name: str | None = None) -> NDArray[np.float64]:
        calls.append((texts, model_name))
        return np.asarray([vectors[text] for text in texts], dtype=np.float64)

    monkeypatch.setattr(baseline, "embed", embed)
    return calls


def test_weighted_pagerank_ranks_bridge_and_keeps_silent_scene(
    embedding_calls: list[tuple[list[str], str | None]],
) -> None:
    """The connecting scene ranks highest and the stronger edge raises its neighbor's score."""
    scenes = [_scene(1, "ALICE: bridge"), _scene(2, "left"), _scene(3, "right"), _scene(4, "")]
    scores = baseline.score_scenes(scenes, model_name="saved-model")
    assert list(scores) == [scene.scene_id for scene in scenes]
    assert scores["E01_s0001"] == 1.0
    assert scores["E01_s0003"] > scores["E01_s0002"] > scores["E01_s0004"] == 0.0
    assert all(isinstance(score, float) and math.isfinite(score) for score in scores.values())
    assert embedding_calls == [(["bridge", "left", "right"], "saved-model")]


def test_graph_has_strict_threshold_no_self_edges_and_cosine_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly 0.2, negative similarities, and self-similarities never become edges."""
    vectors = np.array(
        [
            [1, 0, 0],
            [0.2, math.sqrt(0.96), 0],
            [0.21, 0, math.sqrt(1 - 0.21**2)],
            [-1, 0, 0],
        ]
    )
    monkeypatch.setattr(baseline, "embed", lambda *args, **kwargs: vectors)

    def inspect(graph: nx.Graph, **kwargs: object) -> dict[str, float]:
        assert len(graph) == 4
        assert list(nx.selfloop_edges(graph)) == []
        assert set(graph.edges) == {("E01_s0001", "E01_s0003")}
        assert graph["E01_s0001"]["E01_s0003"]["weight"] == pytest.approx(0.21)
        assert kwargs["weight"] == "weight"
        return dict.fromkeys(graph, 0.25)

    monkeypatch.setattr(baseline.nx, "pagerank", inspect)
    scores = baseline.score_scenes([_scene(index, "text") for index in range(1, 5)])
    assert set(scores.values()) == {0.0}


@pytest.mark.parametrize("texts", [[], [""], [" ", "\n", "ALICE: "]])
def test_empty_and_silent_inputs_do_not_load_embedding_model(texts: list[str]) -> None:
    """Silence is valid and needs no downloaded model or invented ranking."""
    scores = baseline.score_scenes([_scene(index, text) for index, text in enumerate(texts, 1)])
    assert len(scores) == len(texts)
    assert all(score == 0.0 for score in scores.values())


@pytest.mark.parametrize("texts", [["left"], ["left", "left", "left"], ["left", "right"]])
def test_single_uniform_and_disconnected_graphs_have_finite_tied_scores(
    embedding_calls: list[tuple[list[str], str | None]],
    texts: list[str],
) -> None:
    """A zero min-max range yields zeros rather than NaNs or fabricated distinctions."""
    scores = baseline.score_scenes([_scene(index, text) for index, text in enumerate(texts, 1)])
    assert set(scores.values()) == {0.0}


def test_scene_permutation_does_not_change_ranking(
    embedding_calls: list[tuple[list[str], str | None]],
) -> None:
    """Semantic centrality belongs to scene IDs and is independent of input ordering."""
    scenes = [_scene(1, "bridge"), _scene(2, "left"), _scene(3, "right"), _scene(4, "")]
    assert baseline.score_scenes(scenes[::-1]) == pytest.approx(baseline.score_scenes(scenes))


def test_duplicate_scene_ids_fail_instead_of_overwriting_scores() -> None:
    """Every output key must identify exactly one input scene."""
    with pytest.raises(ValueError, match="unique scene IDs"):
        baseline.score_scenes([_scene(1, "left"), _scene(1, "right")])


@pytest.mark.parametrize("vector", [[], [0, 0], [2, 0], [float("nan"), 1], [float("inf"), 0]])
def test_cache_contract_rejects_invalid_vectors(vector: list[float]) -> None:
    """Cached vectors must be finite, nonempty, and L2 normalized."""
    with pytest.raises(ValidationError):
        SceneEmbeddingRecord(scene_id="E01_s0001", vector=vector)


@pytest.mark.parametrize("damage", ["duplicate", "wrong_episode", "dimension"])
def test_cache_contract_checks_scene_identity_and_dimensions(damage: str) -> None:
    """A cache cannot silently mix identities, episodes, or vector dimensions."""
    first = SceneEmbeddingRecord(scene_id="E01_s0001", vector=[1, 0])
    second = SceneEmbeddingRecord(
        scene_id="E01_s0001"
        if damage == "duplicate"
        else ("E02_s0001" if damage == "wrong_episode" else "E01_s0002"),
        vector=[1, 0, 0] if damage == "dimension" else [0, 1],
    )
    with pytest.raises(ValidationError):
        SceneEmbeddingsDoc(
            episode="E01", model_name="model", input_sha256="a" * 64, embeddings=[first, second]
        )


@pytest.fixture
def character_job(
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
) -> JobState:
    """Prepare persisted inputs through Phase 4 without trained models or video decoding."""
    state = run_job(
        make_zip([("ep1.mp4", b"one"), ("ep2.mp4", b"two")]),
        stage="ingest",
        cfg=Settings(sbert_model="saved-model"),
    )
    for stage in (state.stages.transcribe, state.stages.scenes, state.stages.nlp.characters):
        stage.status = "done"
        stage.completed = ["E01", "E02"]
    state.stages.nlp.status = "in_progress"
    directory = Path("data/jobs") / state.job_id
    write_json(directory / "state.json", state)
    write_json(directory / "characters.json", CharacterTable())
    write_json(
        directory / "scenes.json",
        ScenesDoc(
            [
                _scene(1, "ALICE: bridge"),
                _scene(2, "left"),
                _scene(1, "right", episode=2),
                _scene(2, "", episode=2),
            ]
        ),
    )
    return state


def test_baseline_writes_one_score_per_scene_and_leaves_future_phases_pending(
    character_job: JobState,
    embedding_calls: list[tuple[list[str], str | None]],
) -> None:
    """The season graph connects episodes and writes baseline-only score records in scene order."""
    directory = Path("data/jobs") / character_job.job_id
    scenes_before = (directory / "scenes.json").read_bytes()
    state = resume_job(character_job.job_id)
    records = read_json(directory / "scores.json", ScoresDoc).root
    assert [record.scene_id for record in records] == [
        "E01_s0001",
        "E01_s0002",
        "E02_s0001",
        "E02_s0002",
    ]
    assert records[0].baseline_score == 1.0
    assert records[2].baseline_score > records[1].baseline_score > records[3].baseline_score == 0.0
    assert all(record.final_score == 0.0 and record.llm_score is None for record in records)
    assert "llm_score" not in (directory / "scores.json").read_text(encoding="utf-8")
    assert (directory / "scenes.json").read_bytes() == scenes_before
    assert state.stages.nlp.baseline.status == "done"
    assert state.stages.nlp.baseline.completed == ["E01", "E02"]
    assert state.stages.nlp.baseline.pending == []
    assert state.stages.nlp.status == "in_progress"
    assert state.stages.select.status == state.stages.render.status == "pending"
    assert embedding_calls == [(["bridge", "left"], "saved-model"), (["right"], "saved-model")]


def test_episode_embedding_failure_resumes_with_saved_model(
    character_job: JobState,
    embedding_calls: list[tuple[list[str], str | None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second-episode failure preserves the first cache and resumes using persisted settings."""
    embed = baseline.embed

    def fail_second(texts: list[str], *, model_name: str | None = None) -> NDArray[np.float64]:
        if texts == ["right"]:
            raise RuntimeError("simulated embedding failure")
        return embed(texts, model_name=model_name)

    monkeypatch.setattr(baseline, "embed", fail_second)
    with pytest.raises(RuntimeError, match="embedding failure"):
        resume_job(character_job.job_id)
    failed = get_status(character_job.job_id)
    assert failed.stages.nlp.baseline.completed == ["E01"]
    assert failed.stages.nlp.baseline.pending == ["E02"]
    assert failed.stages.nlp.status == failed.stages.nlp.baseline.status == "failed"
    assert failed.errors[-1].stage == "nlp" and failed.errors[-1].episode == "E02"
    embedding_calls.clear()
    monkeypatch.setattr(baseline, "embed", embed)
    monkeypatch.setenv("SBERT_MODEL", "different-environment-model")
    state = resume_job(character_job.job_id)
    assert state.stages.nlp.baseline.status == "done"
    assert len(state.errors) == 1
    assert embedding_calls == [(["right"], "saved-model")]


@pytest.mark.parametrize("damage", ["missing", "invalid", "model", "hash", "missing_row", "vector"])
def test_unfinished_baseline_repairs_only_invalid_embedding_cache(
    character_job: JobState,
    embedding_calls: list[tuple[list[str], str | None]],
    damage: str,
) -> None:
    """Checkpoint validation prevents stale, missing, or malformed embeddings from being reused."""
    state = resume_job(character_job.job_id)
    directory = Path("data/jobs") / state.job_id
    state.stages.nlp.baseline.status = "in_progress"
    write_json(directory / "state.json", state)
    path = directory / "transcripts/E01.embeddings.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if damage == "missing":
        path.unlink()
    elif damage == "invalid":
        path.write_text("bad JSON", encoding="utf-8")
    else:
        if damage == "model":
            document["model_name"] = "wrong-model"
        elif damage == "hash":
            document["input_sha256"] = "0" * 64
        elif damage == "missing_row":
            document["embeddings"].pop()
        else:
            document["embeddings"][0]["vector"] = [0, 0]
        path.write_text(json.dumps(document), encoding="utf-8")
    embedding_calls.clear()
    assert resume_job(state.job_id).stages.nlp.baseline.status == "done"
    assert embedding_calls == [(["bridge", "left"], "saved-model")]


def test_changed_dialogue_invalidates_only_its_episode_cache(
    character_job: JobState,
    embedding_calls: list[tuple[list[str], str | None]],
) -> None:
    """A valid JSON checkpoint still needs to match the current scene IDs and cleaned text."""
    state = resume_job(character_job.job_id)
    directory = Path("data/jobs") / state.job_id
    state.stages.nlp.baseline.status = "in_progress"
    write_json(directory / "state.json", state)
    scenes = read_json(directory / "scenes.json", ScenesDoc)
    scenes.root[0].transcript = "changed"
    write_json(directory / "scenes.json", scenes)
    embedding_calls.clear()
    assert resume_job(state.job_id).stages.nlp.baseline.status == "done"
    assert embedding_calls == [(["changed", "left"], "saved-model")]


def test_pagerank_failure_reuses_all_embeddings_on_resume(
    character_job: JobState,
    embedding_calls: list[tuple[list[str], str | None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A graph failure records no episode and never repeats completed model inference."""
    pagerank = baseline.nx.pagerank

    def fail_graph(*args: object, **kwargs: object) -> dict[str, float]:
        raise nx.PowerIterationFailedConvergence(1000)

    monkeypatch.setattr(baseline.nx, "pagerank", fail_graph)
    with pytest.raises(nx.PowerIterationFailedConvergence):
        resume_job(character_job.job_id)
    state = get_status(character_job.job_id)
    assert state.stages.nlp.baseline.completed == ["E01", "E02"]
    assert state.errors[-1].episode is None
    assert not (Path("data/jobs") / state.job_id / "scores.json").exists()
    monkeypatch.setattr(baseline.nx, "pagerank", pagerank)
    embedding_calls.clear()
    assert resume_job(state.job_id).stages.nlp.baseline.status == "done"
    assert embedding_calls == []


def test_interrupt_after_score_write_recovers_without_reembedding(
    character_job: JobState,
    embedding_calls: list[tuple[list[str], str | None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scores written before the done checkpoint are safely regenerated on restart."""
    persist = baseline.write_json

    def interrupt(path: Path, document: BaseModel) -> None:
        if isinstance(document, JobState) and document.stages.nlp.baseline.status == "done":
            raise KeyboardInterrupt("simulated interruption")
        persist(path, document)

    monkeypatch.setattr(baseline, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        resume_job(character_job.job_id)
    directory = Path("data/jobs") / character_job.job_id
    before = read_json(directory / "scores.json", ScoresDoc)
    assert get_status(character_job.job_id).stages.nlp.baseline.status == "in_progress"
    monkeypatch.setattr(baseline, "write_json", persist)
    embedding_calls.clear()
    assert resume_job(character_job.job_id).stages.nlp.baseline.status == "done"
    assert read_json(directory / "scores.json", ScoresDoc) == before
    assert embedding_calls == []


@pytest.mark.parametrize("damage", ["missing_episode", "empty", "duplicate"])
def test_incomplete_or_duplicate_scenes_fail_before_embedding(
    character_job: JobState,
    damage: str,
) -> None:
    """Baseline output cannot silently omit an episode or overwrite a duplicate scene ID."""
    directory = Path("data/jobs") / character_job.job_id
    scenes = read_json(directory / "scenes.json", ScenesDoc).root
    scenes = (
        scenes[:2] if damage == "missing_episode" else ([] if damage == "empty" else scenes * 2)
    )
    write_json(directory / "scenes.json", ScenesDoc(scenes))
    with pytest.raises(ValueError):
        resume_job(character_job.job_id)
    assert get_status(character_job.job_id).stages.nlp.baseline.status == "failed"
    assert not (directory / "scores.json").exists()


def test_cli_default_baseline_stop_and_completed_stage_are_idempotent(
    character_job: JobState,
    embedding_calls: list[tuple[list[str], str | None]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default CLI resume finishes baseline scoring and skips every completed unit on repeat."""
    assert main(["resume", "--job", character_job.job_id]) == 0
    assert "baseline done" in capsys.readouterr().out
    directory = Path("data/jobs") / character_job.job_id
    paths = [directory / name for name in ("state.json", "scenes.json", "scores.json")]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    embedding_calls.clear()
    assert main(["resume", "--job", character_job.job_id]) == 0
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before
    assert embedding_calls == []
