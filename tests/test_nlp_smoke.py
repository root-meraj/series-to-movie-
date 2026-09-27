"""Opt-in smoke test for installed spaCy NER and the cached MiniLM embedding model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sts.nlp import embeddings, ner
from sts.schemas import (
    CharacterTable,
    EpisodeInfo,
    JobState,
    SceneRecord,
    ScenesDoc,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)


@pytest.mark.integration
def test_real_character_artifacts_and_cached_normalized_embeddings(tmp_path: Path) -> None:
    """Produce character contracts from synthetic dialogue and reuse both trained models."""
    directory = tmp_path / "nlp-smoke"
    text = "Barack Obama met Angela Merkel in Berlin. Barack Obama spoke to Angela Merkel."
    episode = EpisodeInfo(
        episode="E01",
        episode_number=1,
        source_path=directory / "episodes/E01.mp4",
        original_name="synthetic.mp4",
        duration=90,
    )
    state = JobState(job_id="nlp-smoke")
    state.stages.ingest.episodes = [episode]
    state.stages.ingest.episode_count = 1
    for stage in (state.stages.ingest, state.stages.transcribe, state.stages.scenes):
        stage.status = "done"
        stage.completed = ["E01"]
    write_json(directory / "state.json", state)
    transcript = TranscriptDoc(
        episode="E01", language="en", segments=[TranscriptSegment(start=0, end=10, text=text)]
    )
    write_json(directory / "transcripts/E01.transcript.json", transcript)
    write_json(
        directory / "scenes.json",
        ScenesDoc(
            [
                SceneRecord(
                    scene_id="E01_s0001",
                    episode="E01",
                    source_path=episode.source_path,
                    start=0,
                    end=90,
                    duration=90,
                    episode_number=1,
                    transcript=text,
                    is_first=True,
                    is_last=True,
                )
            ]
        ),
    )
    result = ner.build_characters(directory)
    assert result.stages.nlp.characters.status == "done"
    table = read_json(directory / "characters.json", CharacterTable)
    names = {character.name for character in table.characters}
    assert {"Barack Obama", "Angela Merkel"} <= names
    assert set(read_json(directory / "scenes.json", ScenesDoc).root[0].characters) == names
    nlp_cache = ner._NLP
    ner.extract_mentions(transcript)
    assert ner._NLP is nlp_cache

    vectors = embeddings.embed([text, "", "The two leaders held a meeting."])
    assert vectors.shape == (3, 384)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors[[0, 2]], axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(vectors[1], np.zeros(384))
    embedding_cache = embeddings._MODEL
    again = embeddings.embed([text])
    assert embeddings._MODEL is embedding_cache
    np.testing.assert_allclose(again[0], vectors[0], atol=1e-6)
