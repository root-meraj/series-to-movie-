"""Weighted runtime budgets, bookend priority, semantic deduplication, and recovery."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from sts.app.cli import main
from sts.config import Settings
from sts.nlp import scorer_textrank as baseline
from sts.nlp.scorer_llm import MockProvider, build_llm
from sts.pipeline import selector
from sts.pipeline.controller import get_status, resume_job, run_job
from sts.schemas import (
    CharacterTable,
    JobParams,
    JobState,
    SceneEmbeddingRecord,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    ScoresDoc,
    SelectionDoc,
    read_json,
    write_json,
)


def _scene(index: int, duration: float = 20, text: str = "", episode: int = 1) -> SceneRecord:
    return SceneRecord(
        scene_id=f"E{episode:02d}_s{index:04d}",
        episode=f"E{episode:02d}",
        episode_number=episode,
        source_path=Path(f"E{episode:02d}.mp4"),
        start=(index - 1) * 100,
        end=(index - 1) * 100 + duration,
        duration=duration,
        transcript=text,
    )


def _scores(scenes: list[SceneRecord], values: list[float] | None = None) -> list[ScoreRecord]:
    return [
        ScoreRecord(
            scene_id=scene.scene_id,
            baseline_score=value,
            llm_score=5,
            llm_plot_advance=False,
            llm_reason="Mock score",
        )
        for scene, value in zip(scenes, values or [0.0] * len(scenes), strict=True)
    ]


def _ids(document: SelectionDoc) -> list[str]:
    return [clip.scene_id for clip in document.clips]


def _vectors(scenes: list[SceneRecord], values: list[list[float]]) -> list[SceneEmbeddingRecord]:
    return [
        SceneEmbeddingRecord(scene_id=scene.scene_id, vector=vector)
        for scene, vector in zip(scenes, values, strict=True)
    ]


def test_weighted_components_normalize_and_preserve_original_scores() -> None:
    """All four features contribute at their configured relative weights, independent of scale."""
    scenes = [_scene(index) for index in range(1, 4)]
    scores = _scores(scenes, [0.2, 0.5, 0.8])
    for index, (scene, score) in enumerate(zip(scenes, scores, strict=True)):
        scene.audio_rms = 10 + 10 * index
        scene.dialogue_ratio = 0.8 - 0.4 * index
        score.llm_score = 1 + 4 * index
    before = ScoresDoc(scores).model_dump()
    result = selector.combine_scores(scenes, scores[::-1], Settings())
    assert [score.final_score for score in result] == pytest.approx([0.1, 0.5, 0.9])
    assert ScoresDoc(scores).model_dump() == before
    assert [score.llm_reason for score in result] == ["Mock score"] * 3
    doubled = selector.combine_scores(scenes, scores, Settings(score_weights=(5, 3, 1, 1)))
    assert [score.final_score for score in doubled] == pytest.approx([0.1, 0.5, 0.9])


def test_constant_components_and_large_weights_remain_finite() -> None:
    """Constant inputs have no ranking signal and very large finite weights cannot overflow."""
    scenes = [_scene(index) for index in range(1, 4)]
    result = selector.combine_scores(scenes, _scores(scenes), Settings())
    assert [score.final_score for score in result] == [0, 0, 0]
    result = selector.combine_scores(
        scenes, _scores(scenes, [0, 0.5, 1]), Settings(score_weights=(1e308,) * 4)
    )
    assert [score.final_score for score in result] == pytest.approx([0, 0.125, 0.25])


def test_density_selection_reserves_bookends_and_restores_chronology() -> None:
    """Two shorter scenes beat a higher absolute score, with space reserved for both ends."""
    scenes = [_scene(index, duration) for index, duration in enumerate([20, 60, 20, 40, 20], 1)]
    scores = _scores(scenes, [0, 1, 0.6, 0.7, 0])
    document = selector.select(scenes[::-1], scores, 100, Settings(score_weights=(0, 1, 0, 0)))
    assert _ids(document) == [scenes[index].scene_id for index in (0, 2, 3, 4)]
    assert document.selected_total_seconds == 100
    assert document.stats.scenes_dropped == 1
    assert document.stats.original_seconds == 160


def test_scene_order_uses_numeric_episode_and_start() -> None:
    """Input order and score ties cannot change the chronological cut order."""
    scenes = [_scene(1, episode=10), _scene(2, episode=2), _scene(1, episode=2)]
    result = selector.select(scenes, _scores(scenes), 60, Settings())
    assert _ids(result) == ["E02_s0001", "E02_s0002", "E10_s0001"]


def test_minimum_length_filters_optional_clips_but_keeps_bookends() -> None:
    """Forced endpoints survive short length; ordinary clips at the exact minimum survive."""
    scenes = [_scene(index, duration) for index, duration in enumerate([5, 14, 15, 5], 1)]
    result = selector.select(scenes, _scores(scenes), 100, Settings())
    assert _ids(result) == [scenes[index].scene_id for index in (0, 2, 3)]
    assert result.selected_total_seconds == 25


@pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf"), 39.999])
def test_invalid_or_infeasible_target_fails_without_dropping_bookends(target: float) -> None:
    """Budget violations fail explicitly rather than quietly exceeding the target."""
    scenes = [_scene(1), _scene(2)]
    with pytest.raises(ValueError):
        selector.select(scenes, _scores(scenes), target, Settings())


def test_empty_singleton_and_exact_bookend_budget() -> None:
    """An empty input is valid; a single scene counts as both bookends only once."""
    assert selector.select([], [], 20, Settings()).stats.scenes_kept == 0
    for scenes in ([_scene(1)], [_scene(1), _scene(2)]):
        target = 20 * len(scenes)
        result = selector.select(scenes, _scores(scenes), target, Settings())
        assert result.selected_total_seconds == target
        assert len(result.clips) == len(scenes)


def test_dialogue_dedup_only_compares_to_previous_kept_clip() -> None:
    """Repeated dialogue separated by a different retained clip stays, and silence is valid."""
    scenes = [
        _scene(index, text=text) for index, text in enumerate(["", "A", "A", "B", "A", ""], 1)
    ]
    records = _vectors(scenes[1:-1], [[1, 0], [1, 0], [0, 1], [1, 0]])
    result = selector.select(scenes, _scores(scenes), 120, Settings(), embedding_records=records)
    assert _ids(result) == [scenes[index].scene_id for index in (0, 1, 3, 4, 5)]
    assert result.selected_total_seconds == 100


def test_duplicate_chain_compares_against_retained_scene_after_drop() -> None:
    """A dropped intermediary cannot allow another duplicate of the retained scene through."""
    scenes = [_scene(index, text="text" if index < 4 else "") for index in range(1, 5)]
    records = _vectors(
        scenes[:3], [[1, 0], [0.93, math.sqrt(1 - 0.93**2)], [0.93, -math.sqrt(1 - 0.93**2)]]
    )
    result = selector.select(scenes, _scores(scenes), 80, Settings(), embedding_records=records)
    assert _ids(result) == [scenes[0].scene_id, scenes[-1].scene_id]


@pytest.mark.parametrize("cosine,kept", [(0.92, 3), (0.920001, 2), (-1, 3)])
def test_dedup_threshold_is_strict(cosine: float, kept: int) -> None:
    """Only cosine similarity strictly above 0.92 removes an optional clip."""
    scenes = [_scene(1, text="A"), _scene(2, text="B"), _scene(3)]
    records = _vectors(scenes[:2], [[1, 0], [cosine, math.sqrt(1 - cosine**2)]])
    result = selector.select(scenes, _scores(scenes), 60, Settings(), embedding_records=records)
    assert result.stats.scenes_kept == kept


def test_closing_scene_displaces_optional_duplicate_and_both_bookends_survive() -> None:
    """The forced closing scene replaces a duplicate predecessor, including a duplicate chain."""
    scenes = [_scene(index, text="text") for index in range(1, 5)]
    records = _vectors(scenes, [[1, 0], [0.8, 0.6], [0.8, -0.6], [1, 0]])
    # Both optional scenes are not duplicates of each other or the first, but
    # make the last match the prior optional scene in a separate close-up case.
    records[-1] = SceneEmbeddingRecord(scene_id=scenes[-1].scene_id, vector=[0.8, -0.6])
    result = selector.select(scenes, _scores(scenes), 80, Settings(), embedding_records=records)
    assert _ids(result) == [scenes[index].scene_id for index in (0, 1, 3)]
    records = _vectors(scenes, [[1, 0]] * 4)
    result = selector.select(scenes, _scores(scenes), 80, Settings(), embedding_records=records)
    assert _ids(result) == [scenes[0].scene_id, scenes[-1].scene_id]


def test_silent_scenes_never_load_models_or_deduplicate() -> None:
    """Speaker-only and blank transcripts remain eligible and require no model inference."""
    scenes = [_scene(index, text=text) for index, text in enumerate(["", "ALICE: ", "\n"], 1)]
    assert len(selector.select(scenes, _scores(scenes), 60, Settings()).clips) == 3


def test_direct_selection_embeds_cleaned_dialogue_one_episode_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct callers use the saved model and bounded episode batches."""
    calls: list[tuple[list[str], str | None]] = []

    def embed(texts: list[str], *, model_name: str | None = None) -> NDArray[np.float64]:
        calls.append((texts, model_name))
        return np.tile([1.0, 0.0], (len(texts), 1))

    monkeypatch.setattr(selector, "embed", embed)
    scenes = [_scene(1, text="ALICE: hello"), _scene(1, text="BOB: goodbye", episode=2)]
    result = selector.select(scenes, _scores(scenes), 40, Settings(sbert_model="saved-model"))
    assert calls == [(["hello"], "saved-model"), (["goodbye"], "saved-model")]
    assert len(result.clips) == 2


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "missing_llm"])
def test_invalid_score_coverage_is_rejected(damage: str) -> None:
    """Scores must match scene identities exactly, and active LLM weighting requires LLM data."""
    scenes = [_scene(1), _scene(2)]
    scores = _scores(scenes)
    if damage == "missing":
        scores.pop()
    elif damage == "extra":
        scores.extend(_scores([_scene(3)]))
    elif damage == "duplicate":
        scores.append(scores[0])
    else:
        scores[0].llm_score = None
    with pytest.raises(ValueError):
        selector.select(scenes, scores, 40, Settings())


def test_baseline_only_selection_allows_zero_llm_weight() -> None:
    """A standalone caller can intentionally select using baseline-only scores."""
    scenes = [_scene(1)]
    result = selector.select(
        scenes, [ScoreRecord(scene_id=scenes[0].scene_id)], 20, Settings(score_weights=(0, 1, 0, 0))
    )
    assert len(result.clips) == 1


@pytest.mark.parametrize("damage", ["duplicate", "overlap", "bad_first", "bad_last"])
def test_ambiguous_scene_contracts_are_rejected(damage: str) -> None:
    """Duplicate IDs, overlapping source intervals, and wrong endpoint flags fail clearly."""
    scenes = [_scene(1), _scene(2)]
    if damage == "duplicate":
        scenes.append(scenes[0])
    elif damage == "overlap":
        scenes[1] = scenes[1].model_copy(update={"start": 10, "end": 30})
    elif damage == "bad_first":
        scenes[1].is_first = True
    else:
        scenes[0].is_last = True
    with pytest.raises(ValueError):
        selector.select(scenes, _scores(scenes), 100, Settings())


@pytest.mark.parametrize("damage", ["missing", "duplicate", "dimension"])
def test_invalid_embedding_contracts_are_rejected(damage: str) -> None:
    """Missing or inconsistent embeddings must never silently weaken deduplication."""
    scenes = [_scene(1, text="A"), _scene(2, text="B")]
    records = _vectors(scenes, [[1, 0], [0, 1]])
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records.append(records[0])
    else:
        records[1] = SceneEmbeddingRecord(scene_id=scenes[1].scene_id, vector=[1, 0, 0])
    with pytest.raises(ValueError):
        selector.select(scenes, _scores(scenes), 40, Settings(), embedding_records=records)


@pytest.fixture
def embedding_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Provide orthogonal episode dialogue vectors without a downloaded model."""
    calls: list[list[str]] = []
    vectors = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1], "D": [-1, 0, 0]}

    def embed(texts: list[str], *, model_name: str | None = None) -> NDArray[np.float64]:
        assert model_name == "saved-model"
        calls.append(texts)
        return np.asarray([vectors[text] for text in texts], dtype=np.float64)

    monkeypatch.setattr(baseline, "embed", embed)
    return calls


@pytest.fixture
def scored_job(
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    embedding_calls: list[list[str]],
) -> Path:
    """Create real baseline and MockProvider artifacts using synthetic scene records."""
    state = run_job(
        make_zip([("ep1.mp4", b"one"), ("ep2.mp4", b"two")]),
        stage="ingest",
        target_minutes=1,
        cfg=Settings(llm_provider="mock", sbert_model="saved-model", score_weights=(0, 0, 1, 0)),
    )
    directory = Path("data/jobs") / state.job_id
    for stage in (state.stages.transcribe, state.stages.scenes, state.stages.nlp.characters):
        stage.status = "done"
    write_json(directory / "state.json", state)
    scenes = [
        _scene(1, text="A"),
        _scene(2, text="B"),
        _scene(1, text="C", episode=2),
        _scene(2, text="D", episode=2),
    ]
    scenes[0].is_first = True
    scenes[-1].is_last = True
    scenes[2].audio_rms = 1
    write_json(directory / "scenes.json", ScenesDoc(scenes))
    write_json(directory / "characters.json", CharacterTable())
    baseline.build_baseline(directory)
    build_llm(directory, provider=MockProvider())
    embedding_calls.clear()
    return directory


def test_resume_uses_saved_selection_settings_and_preserves_nlp_artifacts(
    scored_job: Path,
    embedding_calls: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment changes cannot alter a job's saved weights/minimum; valid caches are reused."""
    monkeypatch.setenv("SCORE_WEIGHTS", "1,0,0,0")
    monkeypatch.setenv("MIN_CLIP_SECONDS", "99")
    original = read_json(scored_job / "scores.json", ScoresDoc)
    scene_bytes = (scored_job / "scenes.json").read_bytes()
    state = resume_job(scored_job.name, stage="select")
    assert state.stages.select.status == "done"
    assert state.stages.render.status == "pending"
    assert state.stages.select.completed == ["E01", "E02"]
    assert state.stages.select.pending == []
    assert embedding_calls == []
    result = read_json(scored_job / "selected.json", SelectionDoc)
    assert _ids(result) == ["E01_s0001", "E02_s0001", "E02_s0002"]
    assert result.selected_total_seconds == 60
    updated = read_json(scored_job / "scores.json", ScoresDoc)
    assert [score.final_score for score in updated.root] == [0, 0, 1, 0]
    for old, new in zip(original.root, updated.root, strict=True):
        assert old.model_dump(exclude={"final_score"}) == new.model_dump(exclude={"final_score"})
    assert (scored_job / "scenes.json").read_bytes() == scene_bytes


def test_explicit_cli_select_stop_and_completed_stage_are_idempotent(
    scored_job: Path,
    embedding_calls: list[list[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit select stop publishes selection, then skips all work on subsequent calls."""
    assert main(["resume", "--job", scored_job.name, "--stage", "select"]) == 0
    assert "select done" in capsys.readouterr().out
    files = [path for path in scored_job.rglob("*.json")]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in files]
    assert main(["resume", "--job", scored_job.name, "--stage", "select"]) == 0
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in files] == before
    assert embedding_calls == []


@pytest.mark.parametrize("damage", ["missing", "corrupt", "stale"])
def test_missing_invalid_or_stale_cache_repairs_only_affected_episode(
    scored_job: Path,
    embedding_calls: list[list[str]],
    damage: str,
) -> None:
    """Selection validates baseline caches against their model, IDs, input hash, and schema."""
    cache = scored_job / "transcripts/E02.embeddings.json"
    if damage == "missing":
        cache.unlink()
    elif damage == "corrupt":
        cache.write_text("{", encoding="utf-8")
    else:
        cache.write_text(
            cache.read_text(encoding="utf-8").replace("saved-model", "old-model"), encoding="utf-8"
        )
    resume_job(scored_job.name, stage="select")
    assert embedding_calls == [["C", "D"]]


def test_failed_episode_reuses_completed_embedding_on_resume(
    scored_job: Path,
    embedding_calls: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed episode is recorded and resume reuses the earlier committed cache."""
    for path in scored_job.glob("transcripts/*.embeddings.json"):
        path.unlink()
    original = selector.episode_embeddings

    def fail_second(scenes: list[SceneRecord], directory: Path, model: str) -> object:
        if scenes[0].episode == "E02":
            raise RuntimeError("simulated model failure")
        return original(scenes, directory, model)

    monkeypatch.setattr(selector, "episode_embeddings", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        resume_job(scored_job.name, stage="select")
    failed = get_status(scored_job.name)
    assert failed.stages.select.completed == ["E01"]
    assert failed.stages.select.pending == ["E02"]
    assert failed.errors[-1].stage == "select"
    assert failed.errors[-1].episode == "E02"
    monkeypatch.setattr(selector, "episode_embeddings", original)
    assert resume_job(scored_job.name, stage="select").stages.select.status == "done"
    assert embedding_calls == [["A", "B"], ["C", "D"]]


@pytest.mark.parametrize("boundary", ["scores", "selection", "checkpoint"])
def test_publication_failure_resumes_without_model_calls(
    scored_job: Path,
    embedding_calls: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    """Artifacts committed before an interrupted publication are reconstructed deterministically."""
    original = selector.write_json

    def interrupt(path: Path, document: BaseModel) -> None:
        if (
            (boundary == "scores" and path.name == "scores.json")
            or (boundary == "selection" and path.name == "selected.json")
            or (
                boundary == "checkpoint"
                and isinstance(document, JobState)
                and document.stages.select.status == "done"
            )
        ):
            raise OSError("simulated disk failure")
        original(path, document)

    monkeypatch.setattr(selector, "write_json", interrupt)
    with pytest.raises(OSError, match="simulated"):
        resume_job(scored_job.name, stage="select")
    failed = get_status(scored_job.name)
    assert failed.stages.select.status == "failed"
    assert failed.stages.nlp.status == "done"
    assert failed.errors[-1].episode is None
    monkeypatch.setattr(selector, "write_json", original)
    state = resume_job(scored_job.name, stage="select")
    assert state.stages.select.status == "done"
    assert state.errors == failed.errors
    assert read_json(scored_job / "selected.json", SelectionDoc).selected_total_seconds == 60
    assert embedding_calls == []


def test_infeasible_job_records_error_and_never_publishes_selection(scored_job: Path) -> None:
    """Both artifacts remain unpublished by selection when forced bookends cannot fit."""
    state = get_status(scored_job.name)
    state.params.target_seconds = 30
    write_json(scored_job / "state.json", state)
    before = (scored_job / "scores.json").read_bytes()
    with pytest.raises(ValueError, match="at least 40s"):
        resume_job(scored_job.name, stage="select")
    assert get_status(scored_job.name).stages.select.status == "failed"
    assert not (scored_job / "selected.json").exists()
    assert (scored_job / "scores.json").read_bytes() == before


def test_legacy_job_adopts_selection_settings_once(
    scored_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Jobs from earlier phases persist current selection settings before first selection work."""
    state = get_status(scored_job.name)
    state.params.min_clip_seconds = None
    state.params.score_weights = None
    write_json(scored_job / "state.json", state)
    monkeypatch.setenv("MIN_CLIP_SECONDS", "25")
    monkeypatch.setenv("SCORE_WEIGHTS", "0,1,0,0")
    state = resume_job(scored_job.name, stage="select")
    assert state.params.min_clip_seconds == 25
    assert state.params.score_weights == (0, 1, 0, 0)
    assert read_json(scored_job / "selected.json", SelectionDoc).selected_total_seconds == 40


@pytest.mark.parametrize(
    "weights",
    [(0, 0, 0, 0), (-1, 1, 0, 0), (1, 2, 3), (float("inf"), 0, 0, 0), (float("nan"), 0, 0, 0)],
)
def test_persisted_weights_reject_invalid_values(weights: tuple[float, ...]) -> None:
    """Invalid saved weight vectors are rejected at the JSON boundary."""
    with pytest.raises(ValidationError):
        JobParams(score_weights=weights)
