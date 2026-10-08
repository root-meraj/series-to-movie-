"""Scene checkpoints carry exact credit boundaries into scoring and story selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from sts.config import Settings
from sts.nlp.scorer_llm import MockProvider, score_scenes
from sts.pipeline import scenes
from sts.pipeline.selector import select
from sts.schemas import (
    AudioFeaturesDoc,
    CharacterTable,
    CreditRange,
    EpisodeCreditsDoc,
    EpisodeInfo,
    JobParams,
    JobState,
    SceneRecord,
    ScenesDoc,
    ScoreRecord,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)


@pytest.fixture
def credit_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a typed single-episode job with entirely mocked media detection."""
    job_dir = tmp_path / "credit-job"
    state = JobState(
        job_id="credit-job",
        params=JobParams(scene_method="visual", exclude_credits=True, selection_strategy="story"),
    )
    info = EpisodeInfo(
        episode="E01",
        episode_number=1,
        source_path=job_dir / "episodes" / "E01.mkv",
        original_name="example.S01E01.mkv",
        duration=120.0,
    )
    state.stages.ingest.status = state.stages.transcribe.status = "done"
    state.stages.ingest.episode_count = 1
    state.stages.ingest.episodes = [info]
    state.stages.ingest.completed = state.stages.transcribe.completed = ["E01"]
    write_json(job_dir / "state.json", state)
    write_json(
        job_dir / "transcripts" / "E01.transcript.json",
        TranscriptDoc(
            episode="E01",
            language="en",
            segments=[
                TranscriptSegment(start=10, end=13, text="We found the missing witness."),
                TranscriptSegment(start=96, end=98, text="The witness survived."),
                TranscriptSegment(start=104, end=106, text="EXECUTIVE PRODUCER JOHN DOE"),
            ],
        ),
    )
    write_json(
        job_dir / "audio" / "E01.rms.json",
        AudioFeaturesDoc(episode="E01", duration=120, rms_per_second=[0.2] * 120),
    )
    monkeypatch.setattr(scenes, "detect_scenes", lambda *args: [(0.0, 60.0), (60.0, 120.0)])
    return job_dir


def _credit_tail() -> EpisodeCreditsDoc:
    return EpisodeCreditsDoc(
        episode="E01",
        duration=120,
        ranges=[CreditRange(start=100.5, end=120, evidence="chapter: End Credits")],
    )


def _built(job_dir: Path) -> list[SceneRecord]:
    scenes.build_scenes(job_dir)
    return read_json(job_dir / "scenes.json", ScenesDoc).root


def test_credit_cache_splits_story_at_exact_boundary(
    credit_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A coarse scene crossing credits becomes separate narrative and credit records."""
    monkeypatch.setattr(scenes, "detect_credits", lambda *args, **kwargs: _credit_tail())
    records = _built(credit_job)
    assert [(item.start, item.end, item.is_credit) for item in records] == [
        (0.0, 60.0, False),
        (60.0, 100.5, False),
        (100.5, 120.0, True),
    ]
    assert "witness survived" in records[1].transcript
    assert "EXECUTIVE PRODUCER" in records[2].transcript
    assert records[0].is_first
    assert [item.scene_id for item in records if item.is_last] == [records[-1].scene_id]
    assert (
        read_json(credit_job / "transcripts" / "E01.credits.json", EpisodeCreditsDoc)
        == _credit_tail()
    )


def test_credit_text_skips_model_and_story_bookend_preserves_actual_ending(
    credit_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final story scene is retained while staff names never reach the scoring model."""
    monkeypatch.setattr(scenes, "detect_credits", lambda *args, **kwargs: _credit_tail())
    records = _built(credit_job)
    provider = MockProvider()
    cfg = Settings(selection_strategy="story", llm_batch_size=1, min_clip_seconds=15)
    scores, _ = score_scenes(records, CharacterTable(), "", cfg, provider=provider)
    assert {score.scene_id for score in scores} == {
        record.scene_id for record in records if not record.is_credit
    }
    assert len(provider.calls) == 3  # One story summary and two narrative scores.
    assert all("EXECUTIVE PRODUCER JOHN DOE" not in user for _, user, _ in provider.calls)
    scores.append(ScoreRecord(scene_id=records[-1].scene_id))
    result = select(records, scores, 100.5, cfg)
    assert result.selected_total_seconds == 100.5
    assert result.clips[0].start == 0.0
    assert result.clips[-1].end == 100.5
    assert all(clip.end <= 100.5 for clip in result.clips)


def test_resume_reuses_committed_credit_cache_after_alignment_failure(
    credit_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after detection keeps its cache and resume avoids costly frame rescans."""
    calls: list[str] = []

    def detect(*args: object, **kwargs: object) -> EpisodeCreditsDoc:
        calls.append("E01")
        return _credit_tail()

    monkeypatch.setattr(scenes, "detect_credits", detect)
    original = scenes.align

    def fail_alignment(*args: object) -> list[SceneRecord]:
        raise RuntimeError("alignment interrupted")

    monkeypatch.setattr(scenes, "align", fail_alignment)
    with pytest.raises(RuntimeError, match="alignment interrupted"):
        scenes.build_scenes(credit_job)
    assert (credit_job / "transcripts" / "E01.credits.json").is_file()
    assert read_json(credit_job / "state.json", JobState).stages.scenes.completed == []
    monkeypatch.setattr(scenes, "align", original)
    records = _built(credit_job)
    assert calls == ["E01"]
    assert records[-1].is_credit
    assert read_json(credit_job / "state.json", JobState).stages.scenes.status == "done"


def test_postcredit_scene_remains_a_separate_story_block(
    credit_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exclusion ending before the episode leaves post-credit dialogue/action eligible."""
    document = EpisodeCreditsDoc(
        episode="E01",
        duration=120,
        ranges=[CreditRange(start=100.5, end=110, evidence="verified credit frames")],
    )
    write_json(
        credit_job / "transcripts" / "E01.transcript.json",
        TranscriptDoc(
            episode="E01",
            language="en",
            segments=[TranscriptSegment(start=115, end=118, text="The killer is still alive.")],
        ),
    )
    monkeypatch.setattr(scenes, "detect_credits", lambda *args, **kwargs: document)
    records = _built(credit_job)
    assert [(item.start, item.end, item.is_credit) for item in records][-2:] == [
        (100.5, 110.0, True),
        (110.0, 120.0, False),
    ]
    assert records[-1].is_last
    assert "killer is still alive" in records[-1].transcript
    cfg = Settings(selection_strategy="story", min_clip_seconds=15, llm_batch_size=1)
    scores, _ = score_scenes(records, CharacterTable(), "", cfg, provider=MockProvider())
    scores.extend(ScoreRecord(scene_id=item.scene_id) for item in records if item.is_credit)
    result = select(records, scores, 110.5, cfg)
    assert result.clips[-1].start == 110
    assert result.clips[-1].end == 120
    assert all(not (clip.start < 110 and clip.end > 100.5) for clip in result.clips)


def test_disabled_credit_exclusion_never_reads_or_scans_credit_metadata(
    credit_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved disabled settings keep the previous complete-video behavior."""
    state = read_json(credit_job / "state.json", JobState)
    state.params.exclude_credits = False
    write_json(credit_job / "state.json", state)
    monkeypatch.setattr(
        scenes, "detect_credits", lambda *args, **kwargs: pytest.fail("exclusion disabled")
    )
    records = _built(credit_job)
    assert len(records) == 2
    assert not any(item.is_credit for item in records)


def test_wrong_episode_cache_is_regenerated(
    credit_job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An otherwise valid cache from another episode cannot exclude this episode's story."""
    write_json(
        credit_job / "transcripts" / "E01.credits.json",
        EpisodeCreditsDoc(episode="E02", duration=120),
    )
    calls: list[str] = []

    def detect(*args: object, **kwargs: object) -> EpisodeCreditsDoc:
        calls.append("E01")
        return _credit_tail()

    monkeypatch.setattr(scenes, "detect_credits", detect)
    records = _built(credit_job)
    assert calls == ["E01"]
    assert records[-1].is_credit
