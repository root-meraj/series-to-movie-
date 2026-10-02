"""Weighted scene selection with chronological deduplication and durable checkpoints."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from sts.config import Settings
from sts.nlp.embeddings import embed
from sts.nlp.preprocess import preprocess
from sts.nlp.scorer_textrank import episode_embeddings
from sts.schemas import (
    ErrorRecord,
    JobState,
    SceneEmbeddingRecord,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    ScoresDoc,
    SelectedClip,
    SelectionDoc,
    SelectionStats,
    read_json,
    write_json,
)


def _duration(scene: SceneRecord) -> float:
    # Rendered bounds are authoritative even when stored duration was rounded.
    return scene.end - scene.start


def _chronological(scenes: list[SceneRecord]) -> list[SceneRecord]:
    ordered = sorted(scenes, key=lambda scene: (scene.episode_number, scene.start, scene.scene_id))
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("Selection requires unique scene IDs")
    for index, scene in enumerate(ordered):
        if (scene.is_first and index != 0) or (scene.is_last and index != len(ordered) - 1):
            raise ValueError("First/last flags must match chronological season boundaries")
        if index and scene.episode == ordered[index - 1].episode:
            if scene.start < ordered[index - 1].end:
                raise ValueError("Selection requires non-overlapping scenes within each episode")
    return ordered


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def combine_scores(
    scenes: list[SceneRecord], scores: list[ScoreRecord], cfg: Settings
) -> list[ScoreRecord]:
    """Return copies in scene order with min-max components mixed by normalized weights.

    Constant components contribute zero. Missing LLM scores are allowed only
    when the LLM weight is zero. Input score records are never mutated.
    """
    _chronological(scenes)
    by_id = {score.scene_id: score for score in scores}
    if len(by_id) != len(scores) or set(by_id) != {scene.scene_id for scene in scenes}:
        raise ValueError("Scores must contain exactly one record for every scene")
    aligned = [by_id[scene.scene_id] for scene in scenes]
    if cfg.score_weights[0] > 0 and any(score.llm_score is None for score in aligned):
        raise ValueError("Nonzero LLM weight requires an LLM score for every scene")
    components = [
        _normalize([float(score.llm_score or 0) for score in aligned]),
        _normalize([score.baseline_score for score in aligned]),
        _normalize([scene.audio_rms for scene in scenes]),
        _normalize([scene.dialogue_ratio for scene in scenes]),
    ]
    # Scale first so even individually finite, very large weights cannot overflow.
    scaled = [weight / max(cfg.score_weights) for weight in cfg.score_weights]
    weights = [weight / math.fsum(scaled) for weight in scaled]
    return [
        score.model_copy(
            update={
                "final_score": min(
                    1.0,
                    max(
                        0.0,
                        math.fsum(
                            weight * values[index]
                            for weight, values in zip(weights, components, strict=True)
                        ),
                    ),
                )
            }
        )
        for index, score in enumerate(aligned)
    ]


def _candidates(
    scenes: list[SceneRecord], scores: list[ScoreRecord], target_seconds: float, cfg: Settings
) -> tuple[list[SceneRecord], set[str]]:
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("target_seconds must be finite and positive")
    ordered = _chronological(scenes)
    if not ordered:
        return [], set()
    forced = {ordered[0].scene_id, ordered[-1].scene_id}
    chosen = [scene for scene in ordered if scene.scene_id in forced]
    minimum = math.fsum(_duration(scene) for scene in chosen)
    if minimum > target_seconds:
        raise ValueError(
            f"Target {target_seconds:g}s cannot fit the first and last scenes; "
            f"at least {minimum:g}s is required. Start a job with a larger target."
        )
    values = {score.scene_id: score.final_score for score in scores}
    candidates = [
        scene
        for scene in ordered
        if scene.scene_id not in forced and _duration(scene) >= cfg.min_clip_seconds
    ]
    candidates.sort(
        key=lambda scene: (
            -values[scene.scene_id] / _duration(scene),
            scene.episode_number,
            scene.start,
            scene.scene_id,
        )
    )
    durations = [_duration(scene) for scene in chosen]
    for scene in candidates:
        if math.fsum([*durations, _duration(scene)]) <= target_seconds:
            chosen.append(scene)
            durations.append(_duration(scene))
    return _chronological(chosen), forced


def _deduplicate(
    chosen: list[SceneRecord], forced: set[str], records: list[SceneEmbeddingRecord]
) -> list[SceneRecord]:
    vectors: dict[str, list[float]] = {}
    for record in records:
        if record.scene_id in vectors:
            raise ValueError("Deduplication requires unique embedding scene IDs")
        norm = math.hypot(*record.vector)
        vectors[record.scene_id] = [value / norm for value in record.vector]
    if len({len(vector) for vector in vectors.values()}) > 1:
        raise ValueError("Embedding dimensions differ across episodes")

    def duplicate(first: SceneRecord, second: SceneRecord) -> bool:
        left, right = vectors.get(first.scene_id), vectors.get(second.scene_id)
        if left is None or right is None:
            return False
        cosine = math.fsum(a * b for a, b in zip(left, right, strict=True))
        return cosine > 0.92

    kept: list[SceneRecord] = []
    for scene in chosen:
        while kept and duplicate(kept[-1], scene):
            if scene.scene_id in forced:
                if kept[-1].scene_id in forced:
                    break  # Both season bookends must survive, even if identical.
                kept.pop()  # The closing scene takes priority over an optional duplicate.
            else:
                break
        else:
            kept.append(scene)
            continue
        if scene.scene_id in forced:
            kept.append(scene)
    return kept


def _selection(
    scenes: list[SceneRecord],
    scores: list[ScoreRecord],
    target_seconds: float,
    cfg: Settings,
    embedding_records: list[SceneEmbeddingRecord] | None,
) -> SelectionDoc:
    chosen, forced = _candidates(scenes, scores, target_seconds, cfg)
    inputs = [(scene, " ".join(preprocess(scene.transcript))) for scene in chosen]
    nonblank = [(scene, text) for scene, text in inputs if text]
    if embedding_records is None:
        embedding_records = []
        for episode in dict.fromkeys(scene.episode for scene, _ in nonblank):
            batch = [(scene, text) for scene, text in nonblank if scene.episode == episode]
            vectors = embed([text for _, text in batch], model_name=cfg.sbert_model)
            embedding_records.extend(
                SceneEmbeddingRecord(scene_id=scene.scene_id, vector=vector.tolist())
                for (scene, _), vector in zip(batch, vectors, strict=True)
            )
    wanted = {scene.scene_id for scene, _ in nonblank}
    if not wanted.issubset({record.scene_id for record in embedding_records}):
        raise ValueError("Missing embeddings for selected scenes with dialogue")
    # Silence must never be removed by a stray cached vector.
    relevant = [record for record in embedding_records if record.scene_id in wanted]
    kept = _deduplicate(chosen, forced, relevant)
    return SelectionDoc(
        target_seconds=target_seconds,
        selected_total_seconds=math.fsum(_duration(scene) for scene in kept),
        clips=[
            SelectedClip(
                scene_id=scene.scene_id,
                episode=scene.episode,
                source_path=scene.source_path,
                start=scene.start,
                end=scene.end,
            )
            for scene in kept
        ],
        stats=SelectionStats(
            scenes_kept=len(kept),
            scenes_dropped=len(scenes) - len(kept),
            original_seconds=math.fsum(_duration(scene) for scene in scenes),
        ),
    )


def select(
    scenes: list[SceneRecord],
    scores: list[ScoreRecord],
    target_seconds: float,
    cfg: Settings,
    *,
    embedding_records: list[SceneEmbeddingRecord] | None = None,
) -> SelectionDoc:
    """Reserve season bookends, greedily fill by score density, then deduplicate in order.

    Bookends override the minimum-length and duplicate filters, but never the
    runtime budget: an infeasible budget raises ValueError. Other clips must
    meet the configured minimum. Deduplication leaves released time unused.
    Optional validated embedding records avoid loading a model for saved jobs.
    """
    weighted = combine_scores(scenes, scores, cfg)
    return _selection(scenes, weighted, target_seconds, cfg, embedding_records)


def build_selection(job_dir: Path) -> JobState:
    """Resume embedding validation per episode, then publish scores and selected.json.

    Final scoring and selection publication are one retryable unit. A job is
    marked done only after both artifacts are committed; completed stages skip.
    """
    logger = logging.getLogger(__name__)
    state_path = job_dir / "state.json"
    state = read_json(state_path, JobState)
    stage = state.stages.select
    if stage.status == "done":
        logger.info("select done: skipping completed selection for %s", state.job_id)
        return state
    episode: str | None = None
    try:
        stage.status = "in_progress"
        write_json(state_path, state)
        logger.info("select started: %s", state.job_id)
        if state.stages.scenes.status != "done" or state.stages.nlp.status != "done":
            raise ValueError("Selection requires completed scenes and NLP scoring")
        settings = Settings()
        if state.params.min_clip_seconds is None:
            state.params.min_clip_seconds = settings.min_clip_seconds
        if state.params.score_weights is None:
            state.params.score_weights = settings.score_weights
        write_json(state_path, state)
        cfg = Settings(
            min_clip_seconds=state.params.min_clip_seconds,
            score_weights=state.params.score_weights,
            sbert_model=state.params.sbert_model,
        )
        scenes = read_json(job_dir / "scenes.json", ScenesDoc).root
        scores = combine_scores(scenes, read_json(job_dir / "scores.json", ScoresDoc).root, cfg)
        planned = [
            item.episode
            for item in sorted(state.stages.ingest.episodes, key=lambda item: item.episode_number)
        ]
        if not planned or set(planned) != {scene.episode for scene in scenes}:
            raise ValueError("Scene episodes must match the nonempty ingestion manifest")
        # Check the budget before any model/cache work.
        _candidates(scenes, scores, state.params.target_seconds, cfg)
        stage.completed = []
        stage.pending = planned.copy()
        write_json(state_path, state)
        records: list[SceneEmbeddingRecord] = []
        for episode in planned:
            document = episode_embeddings(
                [scene for scene in scenes if scene.episode == episode], job_dir, cfg.sbert_model
            )
            records.extend(document.embeddings)
            stage.completed.append(episode)
            stage.pending.remove(episode)
            write_json(state_path, state)
            logger.info(
                "select unit-progress: %s embeddings ready (%d/%d)",
                episode,
                len(stage.completed),
                len(planned),
            )
        episode = None
        selection = _selection(scenes, scores, state.params.target_seconds, cfg, records)
        write_json(job_dir / "scores.json", ScoresDoc(scores))
        write_json(job_dir / "selected.json", selection)
        stage.status = "done"
        write_json(state_path, state)
        logger.info(
            "select done: %d/%d scenes, %.3fs / %.3fs",
            len(selection.clips),
            len(scenes),
            selection.selected_total_seconds,
            selection.target_seconds,
        )
        return state
    except Exception as exc:
        stage.status = "failed"
        state.errors.append(ErrorRecord(stage="select", episode=episode, message=str(exc)))
        write_json(state_path, state)
        logger.exception("select failed: %s", state.job_id)
        raise
