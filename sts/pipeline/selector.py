"""Weighted scene selection with chronological deduplication and durable checkpoints."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path

from sts.config import Settings
from sts.nlp.embeddings import embed
from sts.nlp.preprocess import preprocess
from sts.nlp.scorer_textrank import episode_embeddings
from sts.pipeline.control import check_stop
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


def _chronological(
    scenes: list[SceneRecord], *, story_boundaries: bool = False
) -> list[SceneRecord]:
    ordered = sorted(scenes, key=lambda scene: (scene.episode_number, scene.start, scene.scene_id))
    if len({scene.scene_id for scene in scenes}) != len(scenes):
        raise ValueError("Selection requires unique scene IDs")
    eligible = [scene for scene in ordered if not getattr(scene, "is_credit", False)]
    first = {ordered[0].scene_id} if ordered else set()
    last = {ordered[-1].scene_id} if ordered else set()
    if story_boundaries and eligible:
        first.add(eligible[0].scene_id)
        last.add(eligible[-1].scene_id)
    for index, scene in enumerate(ordered):
        if (scene.is_first and scene.scene_id not in first) or (
            scene.is_last and scene.scene_id not in last
        ):
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

    Constant components contribute zero. Excluded credits do not influence
    normalization and need no LLM score. Narrative scenes need an LLM score
    when its weight is nonzero. Input score records are never mutated.
    """
    _chronological(
        scenes,
        story_boundaries=cfg.selection_strategy == "story"
        or any(scene.is_credit for scene in scenes),
    )
    by_id = {score.scene_id: score for score in scores}
    if len(by_id) != len(scores) or set(by_id) != {scene.scene_id for scene in scenes}:
        raise ValueError("Scores must contain exactly one record for every scene")
    aligned = [by_id[scene.scene_id] for scene in scenes]
    if cfg.score_weights[0] > 0 and any(
        score.llm_score is None and not scene.is_credit
        for scene, score in zip(scenes, aligned, strict=True)
    ):
        raise ValueError("Nonzero LLM weight requires an LLM score for every scene")
    story_indices = [index for index, scene in enumerate(scenes) if not scene.is_credit]

    def narrative_normalize(values: list[float]) -> list[float]:
        normalized = _normalize([values[index] for index in story_indices])
        by_index = dict(zip(story_indices, normalized, strict=True))
        return [by_index.get(index, 0.0) for index in range(len(scenes))]

    components = [
        narrative_normalize([float(score.llm_score or 0) for score in aligned]),
        narrative_normalize([score.baseline_score for score in aligned]),
        narrative_normalize([scene.audio_rms for scene in scenes]),
        narrative_normalize([scene.dialogue_ratio for scene in scenes]),
    ]
    # Scale first so even individually finite, very large weights cannot overflow.
    scaled = [weight / max(cfg.score_weights) for weight in cfg.score_weights]
    weights = [weight / math.fsum(scaled) for weight in scaled]
    return [
        score.model_copy(
            update={
                "final_score": 0.0
                if scenes[index].is_credit
                else min(
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
    ordered = [
        scene for scene in _chronological(scenes, story_boundaries=True) if not scene.is_credit
    ]
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
    if cfg.selection_strategy == "story":
        return _story_selection(scenes, scores, target_seconds, cfg)
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


@dataclass(frozen=True)
class _StoryBlock:
    scenes: tuple[SceneRecord, ...]
    priority: float
    plot_advance: bool

    @property
    def scene_id(self) -> str:
        """Use the leading constituent scene's stable ID for rendering."""
        return self.scenes[0].scene_id

    @property
    def duration(self) -> float:
        """Measure the full continuous video interval, including shot boundaries."""
        return self.scenes[-1].end - self.scenes[0].start

    @property
    def episode(self) -> str:
        """Return the episode shared by every constituent shot."""
        return self.scenes[0].episode


def _adjacent(first: SceneRecord, second: SceneRecord) -> bool:
    return (
        first.episode == second.episode
        and first.source_path == second.source_path
        and math.isclose(first.end, second.start, abs_tol=0.05)
    )


def _story_blocks(
    scenes: list[SceneRecord], scores: list[ScoreRecord], cfg: Settings
) -> list[_StoryBlock]:
    ordered = _chronological(scenes, story_boundaries=True)
    by_id = {score.scene_id: score for score in scores}
    blocks: list[_StoryBlock] = []

    def make_block(group: list[SceneRecord]) -> _StoryBlock:
        duration = math.fsum(_duration(scene) for scene in group)
        values = [by_id[scene.scene_id] for scene in group]
        average = (
            math.fsum(
                score.final_score * _duration(scene)
                for scene, score in zip(group, values, strict=True)
            )
            / duration
        )
        peak = max(score.final_score for score in values)
        plot = any(score.llm_plot_advance for score in values)
        llm = max((score.llm_score or 1) - 1 for score in values) / 9
        # Absolute model evidence survives uniform min-max components. A short
        # revelation retains its importance within the surrounding camera shots.
        priority = 0.4 * average + 0.2 * peak + 0.2 * llm + 0.2 * plot
        return _StoryBlock(tuple(group), priority, plot)

    group: list[SceneRecord] = []
    for scene in ordered:
        if scene.is_credit:
            if group:
                blocks.append(make_block(group))
            group = []
            continue
        if group and (
            not _adjacent(group[-1], scene)
            or group[-1].end - group[0].start >= 45.0
            or (
                scene.end - group[0].start > 90.0
                and group[-1].end - group[0].start >= cfg.min_clip_seconds
            )
        ):
            blocks.append(make_block(group))
            group = []
        group.append(scene)
    if group:
        blocks.append(make_block(group))
    # Keep a short ending attached to the preceding continuous story, rather
    # than throwing away its dialogue merely because it is below 15 seconds.
    merged: list[_StoryBlock] = []
    for block in blocks:
        if (
            merged
            and block.duration < cfg.min_clip_seconds
            and _adjacent(merged[-1].scenes[-1], block.scenes[0])
            and merged[-1].duration + block.duration <= 90.0
        ):
            prior = merged.pop()
            joined = list(prior.scenes + block.scenes)
            merged.append(make_block(joined))
        else:
            merged.append(block)
    contextual = merged.copy()
    for index, block in enumerate(merged):
        if not any(
            by_id[scene.scene_id].llm_plot_advance and (by_id[scene.scene_id].llm_score or 0) >= 8
            for scene in block.scenes
        ):
            continue
        for neighbor, fraction in ((index - 1, 0.85), (index + 1, 0.70)):
            if 0 <= neighbor < len(merged):
                other = merged[neighbor]
                first, second = (other, block) if neighbor < index else (block, other)
                if _adjacent(first.scenes[-1], second.scenes[0]):
                    contextual[neighbor] = replace(
                        contextual[neighbor],
                        priority=max(contextual[neighbor].priority, block.priority * fraction),
                    )
    return contextual


def _story_rank(block: _StoryBlock) -> tuple[float, int, float, str]:
    return (-block.priority, block.scenes[0].episode_number, block.scenes[0].start, block.scene_id)


def _story_candidates(
    scenes: list[SceneRecord], scores: list[ScoreRecord], target_seconds: float, cfg: Settings
) -> tuple[list[_StoryBlock], set[str], int]:
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("target_seconds must be finite and positive")
    blocks = _story_blocks(scenes, scores, cfg)
    required: set[str] = set()
    for requested in cfg.must_keep_ranges:
        matching = [
            block
            for block in blocks
            if block.episode == requested.episode
            and block.scenes[0].start < requested.end
            and block.scenes[-1].end > requested.start
        ]
        covered = requested.start
        for block in matching:
            if block.scenes[0].start > covered + 0.05:
                break
            covered = max(covered, block.scenes[-1].end)
        if covered < requested.end - 0.05:
            raise ValueError(
                f"Required range {requested.episode} {requested.start:g}-{requested.end:g}s "
                "is outside available story footage or includes credits/a missing interval"
            )
        required.update(block.scene_id for block in matching)
    if not blocks:
        return [], set(), 0
    forced = {blocks[0].scene_id, blocks[-1].scene_id, *required}
    minimum = math.fsum(block.duration for block in blocks if block.scene_id in forced)
    if minimum > target_seconds:
        raise ValueError(
            f"Target {target_seconds:g}s cannot fit the required story blocks and bookends; "
            f"at least {minimum:g}s is required. Start a job with a larger target."
        )
    by_id = {score.scene_id: score for score in scores}
    eligible = [
        block
        for block in blocks
        if block.duration >= cfg.min_clip_seconds
        or block.plot_advance
        or block.scene_id in forced
        or any((by_id[scene.scene_id].llm_score or 0) >= 7 for scene in block.scenes)
    ]
    # Similar embeddings often describe consecutive, equally necessary pieces
    # of the same conversation. Only substantial verbatim repeats without plot
    # progress are candidates for removal, before filling the runtime budget.
    unique: dict[str, _StoryBlock] = {}
    dropped: set[str] = set()
    for block in eligible:
        text = " ".join(" ".join(preprocess(scene.transcript)) for scene in block.scenes).casefold()
        key = " ".join(text.split())
        if block.plot_advance or len(key.split()) < 12:
            continue
        previous = unique.get(key)
        if previous is None:
            unique[key] = block
        elif block.scene_id in forced:
            if previous.scene_id not in forced:
                dropped.add(previous.scene_id)
            unique[key] = block
        elif previous.scene_id in forced or previous.priority >= block.priority:
            dropped.add(block.scene_id)
        else:
            dropped.add(previous.scene_id)
            unique[key] = block
    return [block for block in eligible if block.scene_id not in dropped], forced, len(blocks)


def _repair_story_budget(
    chosen: dict[str, _StoryBlock], remaining: list[_StoryBlock], forced: set[str], target: float
) -> None:
    # Ordinarily 45-second blocks leave well under 2% unused in a two-hour cut.
    # For uneven source scenes, replace at most one optional block by one or two
    # whole blocks. This bounded packing pass avoids arbitrary dialogue cuts.
    for _ in range(4):
        total = math.fsum(block.duration for block in chosen.values())
        if total >= target * 0.98 or not remaining:
            return
        by_duration = sorted(remaining, key=lambda block: (block.duration, _story_rank(block)))
        removable = [
            None,
            *sorted(
                (block for block in chosen.values() if block.scene_id not in forced),
                key=lambda block: (block.priority, block.duration, block.scene_id),
            ),
        ]
        best: tuple[float, float, str, _StoryBlock | None, tuple[_StoryBlock, ...]] | None = None
        for removed in removable:
            allowance = target - total + (removed.duration if removed else 0.0)
            left, right = 0, len(by_duration) - 1
            options: list[tuple[_StoryBlock, ...]] = []
            for block in reversed(by_duration):
                if block.duration <= allowance:
                    options.append((block,))
                    break
            while left < right:
                pair = (by_duration[left], by_duration[right])
                duration = math.fsum(block.duration for block in pair)
                if duration <= allowance:
                    options.append(pair)
                    left += 1
                else:
                    right -= 1
            for added in options:
                new_total = math.fsum(
                    [
                        total,
                        -(removed.duration if removed else 0),
                        *(block.duration for block in added),
                    ]
                )
                quality = math.fsum(block.priority for block in added) - (
                    removed.priority if removed else 0.0
                )
                if new_total <= total + 1e-9 or new_total > target:
                    continue
                # Do not replace a clear plot event solely to add low-value filler.
                if (
                    removed
                    and removed.plot_advance
                    and not any(block.plot_advance for block in added)
                ):
                    continue
                identity = ",".join(block.scene_id for block in added)
                candidate = (new_total, quality, identity, removed, added)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        if best is None:
            return
        _, _, _, removed, added = best
        if removed:
            chosen.pop(removed.scene_id)
            remaining.append(removed)
        added_ids = {block.scene_id for block in added}
        remaining[:] = [block for block in remaining if block.scene_id not in added_ids]
        chosen.update((block.scene_id, block) for block in added)


def _story_selection(
    scenes: list[SceneRecord], scores: list[ScoreRecord], target_seconds: float, cfg: Settings
) -> SelectionDoc:
    blocks, forced, block_count = _story_candidates(scenes, scores, target_seconds, cfg)
    chosen = {block.scene_id: block for block in blocks if block.scene_id in forced}

    def add(block: _StoryBlock) -> bool:
        if block.scene_id in chosen:
            return True
        if (
            math.fsum([*(item.duration for item in chosen.values()), block.duration])
            > target_seconds
        ):
            return False
        chosen[block.scene_id] = block
        return True

    episodes = list(dict.fromkeys(block.episode for block in blocks))
    anchors = forced.copy()
    # Give each episode its strongest remaining event before allocating the rest
    # of the film, preserving the season's setup, developments, and conclusion.
    for episode in episodes:
        if any(block.episode == episode for block in chosen.values()):
            continue
        for block in sorted((item for item in blocks if item.episode == episode), key=_story_rank):
            if add(block):
                anchors.add(block.scene_id)
                break
    source_total = math.fsum(block.duration for block in blocks)
    shares = {
        episode: target_seconds
        * math.fsum(block.duration for block in blocks if block.episode == episode)
        / source_total
        for episode in episodes
    }
    ranked = sorted(blocks, key=_story_rank)
    for block in ranked:
        episode_total = math.fsum(
            item.duration for item in chosen.values() if item.episode == block.episode
        )
        if episode_total + block.duration <= shares[block.episode] * 1.25:
            add(block)
    for block in ranked:
        add(block)
    remaining = [block for block in ranked if block.scene_id not in chosen]
    _repair_story_budget(chosen, remaining, anchors, target_seconds)
    kept = sorted(
        chosen.values(),
        key=lambda block: (block.scenes[0].episode_number, block.scenes[0].start, block.scene_id),
    )
    return SelectionDoc(
        target_seconds=target_seconds,
        selected_total_seconds=math.fsum(block.duration for block in kept),
        clips=[
            SelectedClip(
                scene_id=block.scene_id,
                episode=block.episode,
                source_path=block.scenes[0].source_path,
                start=block.scenes[0].start,
                end=block.scenes[-1].end,
            )
            for block in kept
        ],
        stats=SelectionStats(
            scenes_kept=len(kept),
            scenes_dropped=block_count - len(kept) + sum(scene.is_credit for scene in scenes),
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
    """Select chronological footage using the job's saved editing strategy.

    Story selection groups neighboring shots, protects requested ranges and
    episode coverage, and fills the runtime after excluding credits and actual
    dialogue repeats. Density selection preserves the earlier weighted greedy
    algorithm and its semantic deduplication. Neither exceeds the runtime budget.
    """
    weighted = combine_scores(scenes, scores, cfg)
    return _selection(scenes, weighted, target_seconds, cfg, embedding_records)


def build_selection(job_dir: Path) -> JobState:
    """Prepare each episode using saved settings, then publish scores and selected.json.

    Final scoring and selection publication are one retryable unit. A job is
    marked done only after both artifacts are committed; completed stages skip.
    Only legacy density selection validates embeddings; story editing requires
    no additional model inference after the NLP stage.
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
            selection_strategy=state.params.selection_strategy,
            must_keep_ranges=state.params.must_keep_ranges,
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
        if cfg.selection_strategy == "story":
            _story_candidates(scenes, scores, state.params.target_seconds, cfg)
        else:
            _candidates(scenes, scores, state.params.target_seconds, cfg)
        stage.completed = []
        stage.pending = planned.copy()
        write_json(state_path, state)
        records: list[SceneEmbeddingRecord] = []
        for episode in planned:
            check_stop(job_dir)
            if cfg.selection_strategy == "density":
                document = episode_embeddings(
                    [scene for scene in scenes if scene.episode == episode],
                    job_dir,
                    cfg.sbert_model,
                )
                records.extend(document.embeddings)
            stage.completed.append(episode)
            stage.pending.remove(episode)
            write_json(state_path, state)
            logger.info(
                "select unit-progress: %s selection inputs ready (%d/%d)",
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
