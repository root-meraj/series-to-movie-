"""Cached spaCy PERSON extraction and resumable season-wide character aggregation."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError
from rapidfuzz.fuzz import token_sort_ratio

from sts.config import Settings
from sts.nlp.preprocess import preprocess
from sts.pipeline.control import check_stop
from sts.schemas import (
    CharacterMention,
    CharacterMentionsDoc,
    CharacterRecord,
    CharacterTable,
    ErrorRecord,
    JobState,
    SceneRecord,
    ScenesDoc,
    TranscriptDoc,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    from spacy.language import Language

_NLP: Language | None = None


def _create_nlp() -> Language:
    import spacy

    try:
        return spacy.load(
            "en_core_web_sm", disable=["parser", "tagger", "lemmatizer", "attribute_ruler"]
        )
    except OSError as exc:
        raise RuntimeError(
            "spaCy en_core_web_sm is missing. During setup, run: "
            "python -m spacy download en_core_web_sm"
        ) from exc


def _get_nlp() -> Language:
    global _NLP
    if _NLP is None:
        logging.getLogger(__name__).info("nlp unit-progress: loading spaCy en_core_web_sm")
        _NLP = _create_nlp()
    return _NLP


def extract_mentions(transcript: TranscriptDoc) -> CharacterMentionsDoc:
    """Extract all PERSON surface forms from one cleaned episode, without applying a cap."""
    sentences = preprocess("\n".join(segment.text for segment in transcript.segments))
    if not sentences:
        return CharacterMentionsDoc(episode=transcript.episode)
    counts: Counter[str] = Counter()
    for document in _get_nlp().pipe(sentences, batch_size=32):
        for entity in document.ents:
            name = " ".join(entity.text.split())
            if entity.label_ == "PERSON" and name:
                counts[name] += 1
    return CharacterMentionsDoc(
        episode=transcript.episode,
        mentions=[
            CharacterMention(name=name, mention_count=count)
            for name, count in sorted(
                counts.items(), key=lambda item: (item[0].casefold(), item[0])
            )
        ],
    )


def merge_characters(
    documents: Iterable[CharacterMentionsDoc], *, max_characters: int | None = None
) -> CharacterTable:
    """Aggregate episode counts, fuzzy-dedupe names above 88, and retain the top characters.

    The most frequent observed spelling is canonical, with longer names then
    alphabetical order breaking ties. Similarity is compared to that canonical
    spelling to avoid transitive alias chains. Retained top-ranked characters are
    marked main. No first-name or nickname relationship is inferred.
    """
    limit = Settings().max_characters if max_characters is None else max_characters
    if limit <= 0:
        raise ValueError("max_characters must be positive")
    counts: Counter[str] = Counter()
    for document in documents:
        for mention in document.mentions:
            counts[mention.name] += mention.mention_count
    groups: list[CharacterRecord] = []
    ordered = sorted(
        counts,
        key=lambda name: (-counts[name], -len(name.split()), -len(name), name.casefold(), name),
    )
    for name in ordered:
        scores = [token_sort_ratio(name.casefold(), group.name.casefold()) for group in groups]
        best = max(range(len(groups)), key=lambda index: scores[index]) if groups else None
        if best is not None and scores[best] > 88:
            groups[best].mention_count += counts[name]
            groups[best].aliases.append(name)
        else:
            groups.append(
                CharacterRecord(name=name, aliases=[], mention_count=counts[name], is_main=True)
            )
    groups.sort(key=lambda group: (-group.mention_count, group.name.casefold(), group.name))
    return CharacterTable(characters=groups[:limit])


def extract_characters(
    all_transcripts: Iterable[TranscriptDoc], *, max_characters: int | None = None
) -> CharacterTable:
    """Extract characters from an episode stream, then deduplicate and rank their mentions."""
    return merge_characters(
        (extract_mentions(transcript) for transcript in all_transcripts),
        max_characters=max_characters,
    )


def characters_in_text(text: str, characters: CharacterTable) -> list[str]:
    """Return canonical character names mentioned through whole-word names or observed aliases."""
    cleaned = " ".join(preprocess(text))
    return [
        character.name
        for character in characters.characters
        if any(
            re.search(
                r"(?<!\w)" + r"\s+".join(re.escape(word) for word in name.split()) + r"(?!\w)",
                cleaned,
                flags=re.IGNORECASE,
            )
            for name in (character.name, *character.aliases)
        )
    ]


def _cached_mentions_valid(path: Path, episode: str) -> bool:
    try:
        return read_json(path, CharacterMentionsDoc).episode == episode
    except (OSError, UnicodeError, ValidationError):
        return False


def build_characters(job_dir: Path) -> JobState:
    """Checkpoint per-episode NER, write characters.json, and enrich scene character mentions.

    Character preparation is one NLP substage. The overall NLP status stays
    in_progress until the later baseline and LLM scoring phases are implemented.
    """
    logger = logging.getLogger(__name__)
    state_path = job_dir / "state.json"
    state = read_json(state_path, JobState)
    stage = state.stages.nlp.characters
    if stage.status == "done":
        logger.info("nlp characters done: skipping completed preparation for %s", state.job_id)
        return state
    episode: str | None = None
    try:
        state.stages.nlp.status = "in_progress"
        stage.status = "in_progress"
        write_json(state_path, state)
        logger.info("nlp characters started: %s", state.job_id)
        if state.stages.scenes.status != "done" or not state.stages.ingest.episodes:
            raise ValueError(
                "Character preparation requires completed scenes and an episode manifest"
            )
        episodes = sorted(state.stages.ingest.episodes, key=lambda item: item.episode_number)
        planned = [item.episode for item in episodes]
        cache_paths = {
            identifier: job_dir / "transcripts" / f"{identifier}.characters.json"
            for identifier in planned
        }
        stage.completed = [
            identifier
            for identifier in planned
            if identifier in stage.completed
            and _cached_mentions_valid(cache_paths[identifier], identifier)
        ]
        stage.pending = [identifier for identifier in planned if identifier not in stage.completed]
        write_json(state_path, state)
        for episode in planned:
            check_stop(job_dir)
            if episode in stage.completed:
                logger.info("nlp characters unit-progress: %s already complete", episode)
                continue
            transcript = read_json(
                job_dir / "transcripts" / f"{episode}.transcript.json", TranscriptDoc
            )
            if transcript.episode != episode:
                raise ValueError(f"Transcript episode identifier must match {episode}")
            document = extract_mentions(transcript)
            write_json(cache_paths[episode], document)
            completed = set(stage.completed) | {episode}
            stage.completed = [identifier for identifier in planned if identifier in completed]
            stage.pending = [identifier for identifier in planned if identifier not in completed]
            write_json(state_path, state)
            logger.info(
                "nlp characters unit-progress: %s done (%d/%d)",
                episode,
                len(stage.completed),
                len(planned),
            )
        episode = None
        table = merge_characters(
            (read_json(cache_paths[identifier], CharacterMentionsDoc) for identifier in planned),
            max_characters=state.params.max_characters,
        )
        scene_path = job_dir / "scenes.json"
        records = read_json(scene_path, ScenesDoc).root
        enriched = [
            SceneRecord.model_validate(
                {**record.model_dump(), "characters": characters_in_text(record.transcript, table)}
            )
            for record in records
        ]
        write_json(job_dir / "characters.json", table)
        write_json(scene_path, ScenesDoc(enriched))
        stage.status = "done"
        write_json(state_path, state)
        logger.info("nlp characters done: %d characters", len(table.characters))
        return state
    except Exception as exc:
        stage.status = "failed"
        state.stages.nlp.status = "failed"
        state.errors.append(ErrorRecord(stage="nlp", episode=episode, message=str(exc)))
        write_json(state_path, state)
        logger.exception("nlp characters failed: %s", state.job_id)
        raise
