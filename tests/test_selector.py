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
    RequiredRange,
    SceneEmbeddingRecord,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    ScoresDoc,
    SelectionDoc,
    read_json,
    write_json,
)


def _density_settings(**values: object) -> Settings:
    return Settings(selection_strategy="density", **values)


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
    result = selector.combine_scores(scenes, scores[::-1], _density_settings())
    assert [score.final_score for score in result] == pytest.approx([0.1, 0.5, 0.9])
    assert ScoresDoc(scores).model_dump() == before
    assert [score.llm_reason for score in result] == ["Mock score"] * 3
    doubled = selector.combine_scores(scenes, scores, _density_settings(score_weights=(5, 3, 1, 1)))
    assert [score.final_score for score in doubled] == pytest.approx([0.1, 0.5, 0.9])


def test_constant_components_and_large_weights_remain_finite() -> None:
    """Constant inputs have no ranking signal and very large finite weights cannot overflow."""
    scenes = [_scene(index) for index in range(1, 4)]
    result = selector.combine_scores(scenes, _scores(scenes), _density_settings())
    assert [score.final_score for score in result] == [0, 0, 0]
    result = selector.combine_scores(
        scenes, _scores(scenes, [0, 0.5, 1]), _density_settings(score_weights=(1e308,) * 4)
    )
    assert [score.final_score for score in result] == pytest.approx([0, 0.125, 0.25])


def test_density_selection_reserves_bookends_and_restores_chronology() -> None:
    """Two shorter scenes beat a higher absolute score, with space reserved for both ends."""
    scenes = [_scene(index, duration) for index, duration in enumerate([20, 60, 20, 40, 20], 1)]
    scores = _scores(scenes, [0, 1, 0.6, 0.7, 0])
    document = selector.select(
        scenes[::-1], scores, 100, _density_settings(score_weights=(0, 1, 0, 0))
    )
    assert _ids(document) == [scenes[index].scene_id for index in (0, 2, 3, 4)]
    assert document.selected_total_seconds == 100
    assert document.stats.scenes_dropped == 1
    assert document.stats.original_seconds == 160


def test_density_credits_cannot_replace_narrative_bookends() -> None:
    """Explicit credit markers override legacy density selection's physical bookends."""
    scenes = [_scene(index, duration) for index, duration in enumerate([60, 8, 20, 9, 90], 1)]
    scenes[0].is_credit = scenes[0].is_first = True
    scenes[-1].is_credit = scenes[-1].is_last = True
    scores = _scores(scenes, [1, 0.1, 0.9, 0.1, 1])
    scores[0] = ScoreRecord(scene_id=scenes[0].scene_id, baseline_score=1)
    scores[-1] = ScoreRecord(scene_id=scenes[-1].scene_id, baseline_score=1)
    document = selector.select(scenes, scores, 37, _density_settings())
    assert _ids(document) == [scene.scene_id for scene in scenes[1:-1]]
    assert document.selected_total_seconds == 37
    assert document.stats.scenes_dropped == 2
    assert document.stats.original_seconds == 187


def test_density_all_credit_input_stays_empty() -> None:
    """A wholly excluded episode cannot force a staff card into the film."""
    scenes = [_scene(1), _scene(2)]
    for scene in scenes:
        scene.is_credit = True
    scores = [ScoreRecord(scene_id=scene.scene_id) for scene in scenes]
    document = selector.select(scenes, scores, 60, _density_settings())
    assert document.clips == []
    assert document.selected_total_seconds == 0


def test_density_credit_filter_accepts_narrative_boundary_flags() -> None:
    """Validated narrative endpoint flags remain valid with explicitly excluded tail credits."""
    scenes = [_scene(1), _scene(2), _scene(3)]
    scenes[0].is_first = True
    scenes[1].is_last = True
    scenes[2].is_credit = True
    scores = _scores(scenes)
    scores[-1] = ScoreRecord(scene_id=scenes[-1].scene_id)
    result = selector.select(scenes, scores, 40, _density_settings())
    assert _ids(result) == [scene.scene_id for scene in scenes[:2]]


def test_excluded_credit_values_do_not_change_story_score_normalization() -> None:
    """Unscored credits cannot create ranking signals absent from the narrative footage."""
    scenes = [_scene(1), _scene(2)]
    scenes[0].audio_rms = 10
    scenes[1].audio_rms = 20
    scores = _scores(scenes, [0.2, 0.8])
    original = selector.combine_scores(scenes, scores, _density_settings())
    credit = _scene(3)
    credit.is_credit = True
    credit.audio_rms = 10000
    credit.dialogue_ratio = 1
    credit_score = ScoreRecord(scene_id=credit.scene_id, baseline_score=1)
    updated = selector.combine_scores(
        [*scenes, credit], [*scores, credit_score], _density_settings()
    )
    assert [score.final_score for score in updated[:-1]] == [
        score.final_score for score in original
    ]
    assert updated[-1].final_score == 0


def test_scene_order_uses_numeric_episode_and_start() -> None:
    """Input order and score ties cannot change the chronological cut order."""
    scenes = [_scene(1, episode=10), _scene(2, episode=2), _scene(1, episode=2)]
    result = selector.select(scenes, _scores(scenes), 60, _density_settings())
    assert _ids(result) == ["E02_s0001", "E02_s0002", "E10_s0001"]


def test_minimum_length_filters_optional_clips_but_keeps_bookends() -> None:
    """Forced endpoints survive short length; ordinary clips at the exact minimum survive."""
    scenes = [_scene(index, duration) for index, duration in enumerate([5, 14, 15, 5], 1)]
    result = selector.select(scenes, _scores(scenes), 100, _density_settings())
    assert _ids(result) == [scenes[index].scene_id for index in (0, 2, 3)]
    assert result.selected_total_seconds == 25


@pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf"), 39.999])
def test_invalid_or_infeasible_target_fails_without_dropping_bookends(target: float) -> None:
    """Budget violations fail explicitly rather than quietly exceeding the target."""
    scenes = [_scene(1), _scene(2)]
    with pytest.raises(ValueError):
        selector.select(scenes, _scores(scenes), target, _density_settings())


def test_empty_singleton_and_exact_bookend_budget() -> None:
    """An empty input is valid; a single scene counts as both bookends only once."""
    assert selector.select([], [], 20, _density_settings()).stats.scenes_kept == 0
    for scenes in ([_scene(1)], [_scene(1), _scene(2)]):
        target = 20 * len(scenes)
        result = selector.select(scenes, _scores(scenes), target, _density_settings())
        assert result.selected_total_seconds == target
        assert len(result.clips) == len(scenes)


def test_dialogue_dedup_only_compares_to_previous_kept_clip() -> None:
    """Repeated dialogue separated by a different retained clip stays, and silence is valid."""
    scenes = [
        _scene(index, text=text) for index, text in enumerate(["", "A", "A", "B", "A", ""], 1)
    ]
    records = _vectors(scenes[1:-1], [[1, 0], [1, 0], [0, 1], [1, 0]])
    result = selector.select(
        scenes, _scores(scenes), 120, _density_settings(), embedding_records=records
    )
    assert _ids(result) == [scenes[index].scene_id for index in (0, 1, 3, 4, 5)]
    assert result.selected_total_seconds == 100


def test_duplicate_chain_compares_against_retained_scene_after_drop() -> None:
    """A dropped intermediary cannot allow another duplicate of the retained scene through."""
    scenes = [_scene(index, text="text" if index < 4 else "") for index in range(1, 5)]
    records = _vectors(
        scenes[:3], [[1, 0], [0.93, math.sqrt(1 - 0.93**2)], [0.93, -math.sqrt(1 - 0.93**2)]]
    )
    result = selector.select(
        scenes, _scores(scenes), 80, _density_settings(), embedding_records=records
    )
    assert _ids(result) == [scenes[0].scene_id, scenes[-1].scene_id]


@pytest.mark.parametrize("cosine,kept", [(0.92, 3), (0.920001, 2), (-1, 3)])
def test_dedup_threshold_is_strict(cosine: float, kept: int) -> None:
    """Only cosine similarity strictly above 0.92 removes an optional clip."""
    scenes = [_scene(1, text="A"), _scene(2, text="B"), _scene(3)]
    records = _vectors(scenes[:2], [[1, 0], [cosine, math.sqrt(1 - cosine**2)]])
    result = selector.select(
        scenes, _scores(scenes), 60, _density_settings(), embedding_records=records
    )
    assert result.stats.scenes_kept == kept


def test_closing_scene_displaces_optional_duplicate_and_both_bookends_survive() -> None:
    """The forced closing scene replaces a duplicate predecessor, including a duplicate chain."""
    scenes = [_scene(index, text="text") for index in range(1, 5)]
    records = _vectors(scenes, [[1, 0], [0.8, 0.6], [0.8, -0.6], [1, 0]])
    # Both optional scenes are not duplicates of each other or the first, but
    # make the last match the prior optional scene in a separate close-up case.
    records[-1] = SceneEmbeddingRecord(scene_id=scenes[-1].scene_id, vector=[0.8, -0.6])
    result = selector.select(
        scenes, _scores(scenes), 80, _density_settings(), embedding_records=records
    )
    assert _ids(result) == [scenes[index].scene_id for index in (0, 1, 3)]
    records = _vectors(scenes, [[1, 0]] * 4)
    result = selector.select(
        scenes, _scores(scenes), 80, _density_settings(), embedding_records=records
    )
    assert _ids(result) == [scenes[0].scene_id, scenes[-1].scene_id]


def test_silent_scenes_never_load_models_or_deduplicate() -> None:
    """Speaker-only and blank transcripts remain eligible and require no model inference."""
    scenes = [_scene(index, text=text) for index, text in enumerate(["", "ALICE: ", "\n"], 1)]
    assert len(selector.select(scenes, _scores(scenes), 60, _density_settings()).clips) == 3


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
    result = selector.select(
        scenes, _scores(scenes), 40, _density_settings(sbert_model="saved-model")
    )
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
        selector.select(scenes, scores, 40, _density_settings())


def test_baseline_only_selection_allows_zero_llm_weight() -> None:
    """A standalone caller can intentionally select using baseline-only scores."""
    scenes = [_scene(1)]
    result = selector.select(
        scenes,
        [ScoreRecord(scene_id=scenes[0].scene_id)],
        20,
        _density_settings(score_weights=(0, 1, 0, 0)),
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
        selector.select(scenes, _scores(scenes), 100, _density_settings())


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
        selector.select(scenes, _scores(scenes), 40, _density_settings(), embedding_records=records)


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
        cfg=_density_settings(
            llm_provider="mock", sbert_model="saved-model", score_weights=(0, 0, 1, 0)
        ),
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


def _continuous(
    durations: list[float], *, episode: int = 1, offset: float = 0
) -> list[SceneRecord]:
    scenes: list[SceneRecord] = []
    for index, duration in enumerate(durations, 1):
        scenes.append(
            _scene(index, duration, episode=episode).model_copy(
                update={"start": offset, "end": offset + duration}
            )
        )
        offset += duration
    return scenes


def test_story_camera_shots_fill_two_hour_budget_without_embedding_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short camera cuts become continuous story blocks, filling a realistic season target."""
    scenes = [scene for episode in range(1, 9) for scene in _continuous([8] * 180, episode=episode)]
    scores = _scores(scenes)
    for index, score in enumerate(scores):
        score.llm_score = 8 if index % 13 == 0 else 3
        score.llm_plot_advance = index % 13 == 0

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("Story selection must not load an embedding model")

    monkeypatch.setattr(selector, "embed", forbidden)
    result = selector.select(scenes[::-1], scores, 7200, Settings(selection_strategy="story"))
    assert 7200 * 0.98 <= result.selected_total_seconds <= 7200
    assert {clip.episode for clip in result.clips} == {f"E{episode:02d}" for episode in range(1, 9)}
    assert len(result.clips) < len(scenes) / 5
    assert all(clip.end - clip.start >= 15 for clip in result.clips)
    assert [(clip.episode, clip.start) for clip in result.clips] == sorted(
        (clip.episode, clip.start) for clip in result.clips
    )


def test_story_short_plot_outcome_keeps_surrounding_setup_instead_of_empty_aftermath() -> None:
    """An eight-second scored plot event survives with the nearby setup camera shots."""
    scenes = [_scene(1, 60)]
    setup = _continuous([(354.772 - 213.673) / 20] * 20, offset=213.673)
    # Stable IDs need not encode the helper's temporary shot index.
    for index, scene in enumerate(setup, 2):
        scene.scene_id = f"E01_s{index:04d}"
    setup[-1].end = 354.772
    event = _scene(22, 8.175, "The central incident changes the investigation.").model_copy(
        update={"start": 354.772, "end": 362.947}
    )
    aftermath = _scene(23, 16.85).model_copy(update={"start": 362.947, "end": 379.797})
    scenes.extend([*setup, event, aftermath, _scene(24, 60)])
    scores = _scores(scenes)
    for score in scores:
        score.llm_score = 2
    scores[-3].llm_score = 8
    scores[-3].llm_plot_advance = True
    result = selector.select(scenes, scores, 180, Settings(selection_strategy="story"))
    retained = [
        clip for clip in result.clips if clip.start <= event.start and clip.end >= event.end
    ]
    assert len(retained) == 1
    assert retained[0].start <= event.start - 30
    assert result.selected_total_seconds <= 180


def test_story_credit_bookends_are_excluded_and_missing_credit_llm_is_valid() -> None:
    """Names and ending credits never become the forced season closing scene."""
    scenes = _continuous([20, 60, 60, 100])
    scenes[0].is_credit = True
    scenes[0].is_first = True
    scenes[-1].is_credit = True
    scenes[-1].is_last = True
    scenes[-2].is_last = True
    scores = _scores(scenes)
    scores[0].llm_score = None
    scores[-1].llm_score = None
    result = selector.select(scenes, scores, 120, Settings(selection_strategy="story"))
    assert [(clip.start, clip.end) for clip in result.clips] == [(20, 80), (80, 140)]
    assert result.selected_total_seconds == 120
    assert result.stats.scenes_dropped == 2
    weighted = selector.combine_scores(scenes, scores, Settings(selection_strategy="story"))
    assert weighted[0].final_score == weighted[-1].final_score == 0


def test_story_empty_or_all_credit_input_returns_no_clips() -> None:
    """An empty or entirely excluded source does not manufacture a forced credit clip."""
    cfg = Settings(selection_strategy="story")
    assert selector.select([], [], 120, cfg).clips == []
    scene = _scene(1, 90)
    scene.is_credit = True
    result = selector.select([scene], _scores([scene]), 120, cfg)
    assert result.clips == []
    assert result.stats.scenes_dropped == 1


def test_story_episode_coverage_survives_one_episode_dominating_scores() -> None:
    """Several high-scoring events in episode one cannot consume the entire season cut."""
    scenes = [
        _scene(index, 45, episode=episode) for episode in range(1, 9) for index in range(1, 5)
    ]
    scores = _scores(scenes)
    for scene, score in zip(scenes, scores, strict=True):
        score.llm_score = 9 if scene.episode == "E01" else 3
        score.llm_plot_advance = scene.episode == "E01"
    result = selector.select(scenes, scores, 450, Settings(selection_strategy="story"))
    assert result.selected_total_seconds == 450
    assert {clip.episode for clip in result.clips} == {f"E{episode:02d}" for episode in range(1, 9)}


def test_story_similar_dialogue_embeddings_do_not_remove_distinct_plot_events() -> None:
    """Semantic similarity describes related conversations rather than actual repeated footage."""
    scenes = [_scene(index, 30, f"Different plot dialogue {index}.") for index in range(1, 6)]
    records = _vectors(scenes, [[1, 0]] * 5)
    result = selector.select(
        scenes,
        _scores(scenes),
        150,
        Settings(selection_strategy="story"),
        embedding_records=records,
    )
    assert len(result.clips) == 5
    assert result.selected_total_seconds == 150


def test_story_exact_repeated_dialogue_is_removed_before_filling_released_time() -> None:
    """Only a substantial actual dialogue repeat is excluded, and another block fills its time."""
    repeat = "The team already knows where the missing shipment went after the long investigation."
    scenes = [_scene(index, 30, text=repeat if index in (2, 3) else "") for index in range(1, 7)]
    result = selector.select(scenes, _scores(scenes), 150, Settings(selection_strategy="story"))
    assert result.selected_total_seconds == 150
    assert scenes[1].scene_id in _ids(result)
    assert scenes[2].scene_id not in _ids(result)
    assert scenes[4].scene_id in _ids(result)


def test_story_repeated_words_with_new_plot_progress_are_not_deduplicated() -> None:
    """A later plot-bearing repetition remains eligible even when its dialogue is verbatim."""
    repeat = "The team already knows where the missing shipment went after the long investigation."
    scenes = [_scene(index, 30, text=repeat) for index in range(1, 5)]
    scores = _scores(scenes)
    scores[2].llm_plot_advance = True
    result = selector.select(scenes, scores, 120, Settings(selection_strategy="story"))
    assert scenes[2].scene_id in _ids(result)


def test_story_whole_block_packing_repairs_a_greedy_underfilled_budget() -> None:
    """Two whole 30-second blocks can replace one 45-second block to reach the target."""
    scenes = [_scene(index, duration) for index, duration in enumerate([20, 45, 30, 30, 20], 1)]
    scores = _scores(scenes, [0, 1, 0.6, 0.6, 0])
    cfg = Settings(selection_strategy="story", score_weights=(0, 1, 0, 0))
    result = selector.select(scenes, scores, 100, cfg)
    assert result.selected_total_seconds == 100
    assert _ids(result) == [scenes[index].scene_id for index in (0, 2, 3, 4)]


def test_story_manual_required_range_preserves_complete_incident_and_context() -> None:
    """A user-kept interval protects every overlapping whole block and its dialogue boundaries."""
    scenes = _continuous([10] * 30)
    scores = _scores(scenes)
    cfg = Settings(
        selection_strategy="story",
        must_keep_ranges=[RequiredRange(episode="E01", start=112, end=171)],
    )
    result = selector.select(scenes, scores, 200, cfg)
    intervals = [(clip.start, clip.end) for clip in result.clips]
    assert (100, 150) in intervals
    assert (150, 200) in intervals
    assert result.selected_total_seconds == 200
    assert all(clip.start % 10 == clip.end % 10 == 0 for clip in result.clips)


@pytest.mark.parametrize("damage", ["unknown_episode", "outside", "gap", "credits", "budget"])
def test_story_invalid_or_infeasible_required_ranges_fail_explicitly(damage: str) -> None:
    """Manual inclusions cannot silently omit unavailable footage or exceed the runtime budget."""
    scenes = _continuous([60, 60, 60])
    requested = RequiredRange(episode="E01", start=65, end=115)
    target = 120
    if damage == "unknown_episode":
        requested.episode = "E02"
    elif damage == "outside":
        requested.end = 200
    elif damage == "gap":
        scenes[1] = scenes[1].model_copy(update={"start": 65, "end": 125})
        scenes[2] = scenes[2].model_copy(update={"start": 125, "end": 185})
        requested.start = 60
    elif damage == "credits":
        scenes[1].is_credit = True
    with pytest.raises(ValueError):
        selector.select(
            scenes,
            _scores(scenes),
            target,
            Settings(selection_strategy="story", must_keep_ranges=[requested]),
        )


def test_story_contiguous_shots_report_clip_counts_consistently() -> None:
    """Three joined camera shots form one editing unit in the existing selected.json contract."""
    scenes = _continuous([20, 20, 20])
    result = selector.select(scenes, _scores(scenes), 60, Settings(selection_strategy="story"))
    assert len(result.clips) == result.stats.scenes_kept == 1
    assert result.stats.scenes_dropped == 0
    assert result.clips[0].scene_id == scenes[0].scene_id
    assert result.clips[0].start == 0
    assert result.clips[0].end == 60


def test_story_saved_job_selection_does_not_recompute_embeddings(
    scored_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A story edit reuses the saved scores and needs no model calls during selection."""
    state = read_json(scored_job / "state.json", JobState)
    state.params.selection_strategy = "story"
    write_json(scored_job / "state.json", state)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("Story selection must not validate or rebuild embedding caches")

    monkeypatch.setattr(selector, "episode_embeddings", forbidden)
    result = resume_job(scored_job.name, stage="select")
    assert result.stages.select.status == "done"
    assert result.stages.select.completed == ["E01", "E02"]
    assert read_json(scored_job / "selected.json", SelectionDoc).selected_total_seconds == 60
