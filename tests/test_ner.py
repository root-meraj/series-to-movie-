"""Character aggregation, aliases, scene attribution, and episode checkpoint recovery."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import BaseModel

from sts.app.cli import main
from sts.config import Settings
from sts.nlp import ner
from sts.pipeline.controller import get_status, resume_job, run_job
from sts.schemas import (
    CharacterMention,
    CharacterMentionsDoc,
    CharacterRecord,
    CharacterTable,
    JobState,
    SceneRecord,
    ScenesDoc,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)


@dataclass
class _Entity:
    text: str
    label_: str = "PERSON"


@dataclass
class _Doc:
    ents: list[_Entity]


@dataclass
class _Pipeline:
    sentences: list[str] = field(default_factory=list)

    def pipe(self, sentences: list[str], *, batch_size: int) -> Iterator[_Doc]:
        """Recognize fixed names and an organization without trained spaCy weights."""
        assert batch_size == 32
        for sentence in sentences:
            self.sentences.append(sentence)
            yield _Doc(
                [
                    _Entity(match.group(), "ORG" if match.group() == "Acme" else "PERSON")
                    for match in re.finditer(
                        r"Walter White|Walter Whte|Jesse Pinkman|Acme", sentence
                    )
                ]
            )


@pytest.fixture
def nlp(monkeypatch: pytest.MonkeyPatch) -> _Pipeline:
    """Supply a predictable NER pipeline and allow singleton cache checks."""
    pipeline = _Pipeline()
    monkeypatch.setattr(ner, "_create_nlp", lambda: pipeline)
    return pipeline


def _transcript(episode: str, text: str) -> TranscriptDoc:
    return TranscriptDoc(
        episode=episode, language="en", segments=[TranscriptSegment(start=0, end=10, text=text)]
    )


def _mentions(episode: str, **counts: int) -> CharacterMentionsDoc:
    return CharacterMentionsDoc(
        episode=episode,
        mentions=[
            CharacterMention(name=name, mention_count=count) for name, count in counts.items()
        ],
    )


def test_person_extraction_strips_speakers_and_ignores_organizations(nlp: _Pipeline) -> None:
    """Only PERSON mentions from cleaned dialogue contribute to character counts."""
    document = ner.extract_mentions(
        _transcript("E01", "WALTER: Walter White works at Acme.\nJESSE: Walter White left.")
    )
    assert document == _mentions("E01", **{"Walter White": 2})
    assert nlp.sentences == ["Walter White works at Acme.", "Walter White left."]
    assert ner._NLP is nlp
    ner.extract_mentions(_transcript("E02", "Jesse Pinkman left."))
    assert ner._NLP is nlp


def test_empty_transcripts_do_not_load_ner() -> None:
    """Silent seasons produce an empty character table without model construction."""
    assert ner.extract_characters([TranscriptDoc(episode="E01", language="en")]).characters == []
    assert ner._NLP is None


def test_episode_stream_aggregates_aliases_before_applying_cap(nlp: _Pipeline) -> None:
    """A frequent character across episodes outranks a locally frequent alternative."""
    transcripts = (
        _transcript("E01", "Walter White. Walter Whte. Jesse Pinkman. Jesse Pinkman."),
        _transcript("E02", "Walter White. Walter Whte."),
    )
    table = ner.extract_characters(iter(transcripts), max_characters=1)
    assert table.characters == [
        CharacterRecord(name="Walter White", aliases=["Walter Whte"], mention_count=4, is_main=True)
    ]


@pytest.mark.parametrize(("similarity", "group_count"), [(88.0, 2), (88.01, 1)])
def test_alias_threshold_is_strictly_greater_than_88(
    monkeypatch: pytest.MonkeyPatch, similarity: float, group_count: int
) -> None:
    """The specified strict threshold determines whether a spelling joins a character."""
    monkeypatch.setattr(ner, "token_sort_ratio", lambda left, right: similarity)
    table = ner.merge_characters([_mentions("E01", Alice=2, Alicia=1)])
    assert len(table.characters) == group_count
    assert sum(character.mention_count for character in table.characters) == 3


def test_cap_is_applied_after_season_aggregation_and_uses_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep candidates until every episode contributes; honor the configured final cap."""
    monkeypatch.setenv("MAX_CHARACTERS", "1")
    table = ner.merge_characters(
        [_mentions("E01", Alice=3, Bob=2), _mentions("E02", Claire=3, Bob=2)]
    )
    assert [(item.name, item.mention_count) for item in table.characters] == [("Bob", 4)]
    with pytest.raises(ValueError, match="must be positive"):
        ner.merge_characters([], max_characters=0)


def test_scene_names_use_whole_words_observed_aliases_and_canonical_order() -> None:
    """Substring matches and speaker labels cannot invent scene character mentions."""
    table = CharacterTable(
        characters=[
            CharacterRecord(
                name="Walter White", aliases=["Walter Whte"], mention_count=3, is_main=True
            ),
            CharacterRecord(name="Ann", mention_count=1, is_main=True),
        ]
    )
    assert ner.characters_in_text("ANN: Anna met WALTER   WHTE.", table) == ["Walter White"]
    assert ner.characters_in_text("Ann met Walter White.", table) == ["Walter White", "Ann"]
    assert ner.characters_in_text("", table) == []


@pytest.fixture
def scene_job(
    make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> JobState:
    """Prepare a two-episode job using persisted scene and transcript contracts."""
    state = run_job(
        make_zip([("ep1.mp4", b"one"), ("ep2.mp4", b"two")]),
        stage="ingest",
        cfg=Settings(max_characters=1, sbert_model="saved-model"),
    )
    directory = Path("data/jobs") / state.job_id
    records: list[SceneRecord] = []
    for episode in state.stages.ingest.episodes:
        text = "Walter White met Jesse Pinkman. Walter White left."
        write_json(
            directory / "transcripts" / f"{episode.episode}.transcript.json",
            _transcript(episode.episode, text),
        )
        records.append(
            SceneRecord(
                scene_id=f"{episode.episode}_s0001",
                episode=episode.episode,
                source_path=episode.source_path,
                start=0,
                end=90,
                duration=90,
                episode_number=episode.episode_number,
                transcript=text,
                is_first=episode.episode_number == 1,
                is_last=episode.episode_number == 2,
            )
        )
    write_json(directory / "scenes.json", ScenesDoc(records))
    for stage in (state.stages.transcribe, state.stages.scenes):
        stage.status = "done"
        stage.completed = ["E01", "E02"]
    write_json(directory / "state.json", state)
    return state


def test_character_stage_writes_contracts_and_preserves_other_scene_fields(
    scene_job: JobState,
    nlp: _Pipeline,
) -> None:
    """Default resume builds characters and enriches scenes while leaving scoring unfinished."""
    directory = Path("data/jobs") / scene_job.job_id
    before = read_json(directory / "scenes.json", ScenesDoc).root
    result = resume_job(scene_job.job_id)
    assert result.stages.nlp.status == "in_progress"
    assert result.stages.nlp.characters.status == "done"
    assert result.stages.nlp.characters.completed == ["E01", "E02"]
    assert result.stages.nlp.characters.pending == []
    assert result.params.sbert_model == "saved-model"
    table = read_json(directory / "characters.json", CharacterTable)
    assert [(item.name, item.mention_count) for item in table.characters] == [("Walter White", 4)]
    after = read_json(directory / "scenes.json", ScenesDoc).root
    assert all(item.characters == ["Walter White"] for item in after)
    assert [item.model_dump(exclude={"characters"}) for item in before] == [
        item.model_dump(exclude={"characters"}) for item in after
    ]
    assert result.stages.select.status == result.stages.render.status == "pending"
    assert not (directory / "scores.json").exists()


def test_mid_season_failure_resumes_only_unfinished_episodes_with_saved_cap(
    scene_job: JobState,
    nlp: _Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed NER survives a later episode failure and changing process settings."""
    extract = ner.extract_mentions

    def fail_second(transcript: TranscriptDoc) -> CharacterMentionsDoc:
        if transcript.episode == "E02":
            raise RuntimeError("simulated NER failure")
        return extract(transcript)

    monkeypatch.setattr(ner, "extract_mentions", fail_second)
    with pytest.raises(RuntimeError, match="NER failure"):
        resume_job(scene_job.job_id)
    failed = get_status(scene_job.job_id)
    assert failed.stages.nlp.status == failed.stages.nlp.characters.status == "failed"
    assert failed.stages.nlp.characters.completed == ["E01"]
    assert failed.stages.nlp.characters.pending == ["E02"]
    assert failed.errors[-1].stage == "nlp" and failed.errors[-1].episode == "E02"
    nlp.sentences.clear()
    monkeypatch.setattr(ner, "extract_mentions", extract)
    monkeypatch.setenv("MAX_CHARACTERS", "12")
    result = resume_job(scene_job.job_id)
    assert result.stages.nlp.characters.status == "done"
    assert len(nlp.sentences) == 2
    assert len(result.errors) == 1
    table = read_json(Path("data/jobs") / result.job_id / "characters.json", CharacterTable)
    assert len(table.characters) == 1


def test_final_write_failure_reuses_all_episode_caches(
    scene_job: JobState,
    nlp: _Pipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interruption while enriching scenes does not repeat any completed NER inference."""
    persist = ner.write_json

    def fail_scenes(path: Path, document: BaseModel) -> None:
        if path.name == "scenes.json":
            raise OSError("simulated scene write failure")
        persist(path, document)

    monkeypatch.setattr(ner, "write_json", fail_scenes)
    with pytest.raises(OSError, match="scene write failure"):
        resume_job(scene_job.job_id)
    failed = get_status(scene_job.job_id)
    assert failed.stages.nlp.characters.completed == ["E01", "E02"]
    assert failed.errors[-1].episode is None
    nlp.sentences.clear()
    monkeypatch.setattr(ner, "write_json", persist)
    assert resume_job(scene_job.job_id).stages.nlp.characters.status == "done"
    assert nlp.sentences == []


@pytest.mark.parametrize("damage", ["missing", "invalid", "wrong_episode"])
def test_unfinished_character_stage_repairs_damaged_episode_cache(
    scene_job: JobState,
    nlp: _Pipeline,
    damage: str,
) -> None:
    """Resume validates previously checkpointed files before skipping expensive extraction."""
    completed = resume_job(scene_job.job_id)
    directory = Path("data/jobs") / scene_job.job_id
    completed.stages.nlp.characters.status = "in_progress"
    write_json(directory / "state.json", completed)
    path = directory / "transcripts/E01.characters.json"
    if damage == "missing":
        path.unlink()
    elif damage == "invalid":
        path.write_text("invalid JSON", encoding="utf-8")
    else:
        write_json(path, _mentions("E02", Wrong=1))
    nlp.sentences.clear()
    assert resume_job(scene_job.job_id).stages.nlp.characters.status == "done"
    assert len(nlp.sentences) == 2
    assert read_json(path, CharacterMentionsDoc).episode == "E01"


def test_wrong_transcript_episode_records_failure(scene_job: JobState) -> None:
    """A misplaced transcript cannot contribute character mentions to another episode."""
    directory = Path("data/jobs") / scene_job.job_id
    write_json(directory / "transcripts/E01.transcript.json", _transcript("E02", "Wrong."))
    with pytest.raises(ValueError, match="identifier must match E01"):
        resume_job(scene_job.job_id)
    failed = get_status(scene_job.job_id)
    assert failed.errors[-1].episode == "E01"
    assert failed.stages.nlp.characters.completed == []


def test_cli_default_stop_and_completed_preparation_are_idempotent(
    scene_job: JobState,
    nlp: _Pipeline,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default CLI resume reaches Phase 4, then a second resume leaves all artifacts untouched."""
    assert main(["resume", "--job", scene_job.job_id]) == 0
    assert "characters done" in capsys.readouterr().out
    directory = Path("data/jobs") / scene_job.job_id
    paths = [directory / name for name in ("state.json", "scenes.json", "characters.json")]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    nlp.sentences.clear()
    assert main(["resume", "--job", scene_job.job_id]) == 0
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before
    assert nlp.sentences == []
