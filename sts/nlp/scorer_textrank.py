"""Season-wide TextRank scores with resumable per-episode embedding checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import networkx as nx
import numpy as np
from pydantic import ValidationError

from sts.nlp.embeddings import embed
from sts.nlp.preprocess import preprocess
from sts.schemas import (
    ErrorRecord,
    JobState,
    SceneEmbeddingRecord,
    SceneEmbeddingsDoc,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    ScoresDoc,
    read_json,
    write_json,
)


def _scene_inputs(scenes: list[SceneRecord]) -> list[tuple[str, str]]:
    identifiers = [scene.scene_id for scene in scenes]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("TextRank requires unique scene IDs")
    return [(scene.scene_id, " ".join(preprocess(scene.transcript))) for scene in scenes]


def _embed_inputs(
    inputs: list[tuple[str, str]], model_name: str | None
) -> list[SceneEmbeddingRecord]:
    nonblank = [(identifier, text) for identifier, text in inputs if text]
    if not nonblank:
        return []
    vectors = embed([text for _, text in nonblank], model_name=model_name)
    return [
        SceneEmbeddingRecord(scene_id=identifier, vector=vector.tolist())
        for (identifier, _), vector in zip(nonblank, vectors, strict=True)
    ]


def _rank_embeddings(
    identifiers: list[str], embeddings: list[SceneEmbeddingRecord]
) -> dict[str, float]:
    if not identifiers:
        return {}
    dimensions = {len(record.vector) for record in embeddings}
    if len(dimensions) > 1:
        raise ValueError("Embedding dimensions differ across episodes")
    dimension = next(iter(dimensions), 0)
    vectors = np.zeros((len(identifiers), dimension), dtype=np.float64)
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    for record in embeddings:
        vectors[positions[record.scene_id]] = record.vector
    # Normalize again to compute cosine exactly despite float32 serialization roundoff.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    graph = nx.Graph()
    graph.add_nodes_from(identifiers)  # Keep isolated and silent scenes in PageRank.
    for index, identifier in enumerate(identifiers):
        # One row at a time avoids allocating an additional season-sized N x N matrix.
        similarities = np.clip(vectors[index + 1 :] @ vectors[index], -1.0, 1.0)
        graph.add_weighted_edges_from(
            (identifier, identifiers[index + 1 + offset], float(similarities[offset]))
            for offset in np.flatnonzero(similarities > 0.2)
        )
    ranks = nx.pagerank(graph, alpha=0.85, weight="weight", max_iter=1000, tol=1e-10)
    low, high = min(ranks.values()), max(ranks.values())
    if math.isclose(high, low, rel_tol=0.0, abs_tol=1e-12):
        return dict.fromkeys(identifiers, 0.0)
    return {
        identifier: float((ranks[identifier] - low) / (high - low)) for identifier in identifiers
    }


def score_scenes(scenes: list[SceneRecord], *, model_name: str | None = None) -> dict[str, float]:
    """Rank all scenes by weighted cosine-graph PageRank and min-max normalize to [0, 1].

    Edges require cosine similarity strictly above 0.2 and never connect a scene
    to itself. Silent scenes remain isolated nodes. An empty input returns an
    empty mapping; a single scene or a tied ranking receives zero baseline scores.
    """
    inputs = _scene_inputs(scenes)
    return _rank_embeddings(
        [identifier for identifier, _ in inputs], _embed_inputs(inputs, model_name)
    )


def _input_hash(inputs: list[tuple[str, str]]) -> str:
    payload = json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_valid(path: Path, episode: str, model_name: str, inputs: list[tuple[str, str]]) -> bool:
    try:
        document = read_json(path, SceneEmbeddingsDoc)
        return (
            document.episode == episode
            and document.model_name == model_name
            and document.input_sha256 == _input_hash(inputs)
            and [record.scene_id for record in document.embeddings]
            == [identifier for identifier, text in inputs if text]
        )
    except (OSError, UnicodeError, ValidationError):
        return False


def build_baseline(job_dir: Path) -> JobState:
    """Checkpoint each episode's embeddings, then atomically write season-wide scores.json.

    Completed embedding units are validated and reused on resume. Graph ranking
    and score publication form one final unit. Overall NLP stays in_progress
    until the later LLM scoring phase is implemented.
    """
    logger = logging.getLogger(__name__)
    state_path = job_dir / "state.json"
    state = read_json(state_path, JobState)
    stage = state.stages.nlp.baseline
    if stage.status == "done":
        logger.info("nlp baseline done: skipping completed scoring for %s", state.job_id)
        return state
    episode: str | None = None
    try:
        stage.status = "in_progress"
        state.stages.nlp.status = "in_progress"
        write_json(state_path, state)
        logger.info("nlp baseline started: %s", state.job_id)
        if state.stages.scenes.status != "done" or state.stages.nlp.characters.status != "done":
            raise ValueError("Baseline scoring requires completed scenes and character preparation")
        scenes = read_json(job_dir / "scenes.json", ScenesDoc).root
        inputs = _scene_inputs(scenes)
        planned = [
            item.episode
            for item in sorted(state.stages.ingest.episodes, key=lambda item: item.episode_number)
        ]
        if not planned or set(planned) != {scene.episode for scene in scenes}:
            raise ValueError("Scene episodes must match the nonempty ingestion manifest")
        grouped: dict[str, list[tuple[str, str]]] = {identifier: [] for identifier in planned}
        for scene, scene_input in zip(scenes, inputs, strict=True):
            grouped[scene.episode].append(scene_input)
        model_name = state.params.sbert_model
        paths = {
            identifier: job_dir / "transcripts" / f"{identifier}.embeddings.json"
            for identifier in planned
        }
        stage.completed = [
            identifier
            for identifier in planned
            if identifier in stage.completed
            and _cache_valid(paths[identifier], identifier, model_name, grouped[identifier])
        ]
        stage.pending = [identifier for identifier in planned if identifier not in stage.completed]
        write_json(state_path, state)
        for episode in planned:
            if episode in stage.completed:
                logger.info("nlp baseline unit-progress: %s embeddings already complete", episode)
                continue
            document = SceneEmbeddingsDoc(
                episode=episode,
                model_name=model_name,
                input_sha256=_input_hash(grouped[episode]),
                embeddings=_embed_inputs(grouped[episode], model_name),
            )
            write_json(paths[episode], document)
            completed = set(stage.completed) | {episode}
            stage.completed = [identifier for identifier in planned if identifier in completed]
            stage.pending = [identifier for identifier in planned if identifier not in completed]
            write_json(state_path, state)
            logger.info(
                "nlp baseline unit-progress: %s embeddings done (%d/%d)",
                episode,
                len(stage.completed),
                len(planned),
            )
        episode = None
        logger.info("nlp baseline unit-progress: ranking %d scenes across the season", len(scenes))
        scores = _rank_embeddings(
            [scene.scene_id for scene in scenes],
            [
                record
                for identifier in planned
                for record in read_json(paths[identifier], SceneEmbeddingsDoc).embeddings
            ],
        )
        write_json(
            job_dir / "scores.json",
            ScoresDoc(
                [
                    ScoreRecord(scene_id=scene.scene_id, baseline_score=scores[scene.scene_id])
                    for scene in scenes
                ]
            ),
        )
        stage.status = "done"
        write_json(state_path, state)
        logger.info("nlp baseline done: %d scene scores", len(scores))
        return state
    except Exception as exc:
        stage.status = "failed"
        state.stages.nlp.status = "failed"
        state.errors.append(ErrorRecord(stage="nlp", episode=episode, message=str(exc)))
        write_json(state_path, state)
        logger.exception("nlp baseline failed: %s", state.job_id)
        raise
