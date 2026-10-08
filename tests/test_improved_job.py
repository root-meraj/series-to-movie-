"""Refined edits reuse saved source work without altering the previous movie."""

from __future__ import annotations

from pathlib import Path

import pytest

from sts.pipeline import controller
from sts.schemas import (
    AudioFeaturesDoc,
    CharacterMention,
    CharacterMentionsDoc,
    CreditRange,
    EpisodeCreditsDoc,
    EpisodeInfo,
    JobParams,
    JobState,
    RequiredRange,
    TranscriptDoc,
    TranscriptSegment,
    read_json,
    write_json,
)


@pytest.fixture
def saved_season(tmp_path: Path) -> tuple[Path, JobState]:
    """Create completed contracts and tiny fake media without decoding or model calls."""
    jobs_dir = tmp_path / "jobs"
    directory = jobs_dir / "original"
    state = JobState(
        job_id="original",
        params=JobParams(
            name="Saved Season",
            target_seconds=7200.0,
            zip_path=tmp_path / "removed-source.zip",
            zip_sha256="a" * 64,
            transcript_source="subtitles",
            subtitle_language="en",
            whisper_device="cuda",
            whisper_compute="float16",
            llm_provider="ollama",
            llm_model="saved-local-model:latest",
            ollama_base_url="http://127.0.0.1:11434",
            season_number=2,
            must_keep_ranges=[RequiredRange(episode="E01", start=25.0, end=30.0)],
        ),
    )
    episodes: list[EpisodeInfo] = []
    for index in range(1, 3):
        episode = f"E{index:02d}"
        video_path = directory / "episodes" / f"{episode}.mkv"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"stand-in video {episode}".encode())
        episodes.append(
            EpisodeInfo(
                episode=episode,
                episode_number=index,
                source_path=video_path,
                original_name=f"show.S02{episode}.mkv",
                duration=90.0,
            )
        )
        audio_path = directory / "audio" / f"{episode}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(f"stand-in audio {episode}".encode())
        write_json(
            directory / "audio" / f"{episode}.rms.json",
            AudioFeaturesDoc(episode=episode, duration=90.0, rms_per_second=[0.5] * 90),
        )
        write_json(
            directory / "transcripts" / f"{episode}.transcript.json",
            TranscriptDoc(
                episode=episode,
                language="en",
                source="embedded_subtitles",
                segments=[TranscriptSegment(start=5, end=10, text="Alice has a plan.")],
            ),
        )
        write_json(
            directory / "transcripts" / f"{episode}.characters.json",
            CharacterMentionsDoc(
                episode=episode, mentions=[CharacterMention(name="Alice", mention_count=1)]
            ),
        )
        write_json(
            directory / "transcripts" / f"{episode}.credits.json",
            EpisodeCreditsDoc(
                episode=episode,
                duration=90.0,
                ranges=[CreditRange(start=80, end=90, evidence="chapter: End Credits")],
            ),
        )
    identifiers = [episode.episode for episode in episodes]
    state.stages.ingest.status = "done"
    state.stages.ingest.episodes = episodes
    state.stages.ingest.completed = identifiers
    state.stages.ingest.episode_count = len(episodes)
    state.stages.transcribe.status = "done"
    state.stages.transcribe.completed = identifiers
    for name in ("scenes", "nlp", "select", "render"):
        getattr(state.stages, name).status = "done"
    for substage in ("characters", "baseline", "llm"):
        getattr(state.stages.nlp, substage).status = "done"
    state.stages.render.video_done = True
    state.stages.render.subtitles_done = True
    output_dir = directory / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "condensed_movie.mp4").write_bytes(b"previous rendered movie")
    (output_dir / "condensed.srt").write_text("previous subtitles", encoding="utf-8")
    write_json(directory / "state.json", state)
    return jobs_dir, state


def test_improved_job_reuses_capture_and_preserves_original_bytes_and_paths(
    saved_season: tuple[Path, JobState], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new edit starts later work fresh while every earlier project file stays identical."""
    jobs_dir, previous = saved_season
    source_dir = jobs_dir / previous.job_id
    before = {
        path.relative_to(source_dir): path.read_bytes()
        for path in source_dir.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "different-environment-model")
    refined = controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    directory = jobs_dir / refined.job_id
    assert refined.job_id != previous.job_id
    assert refined.params.parent_job_id == previous.job_id
    assert refined.params.name == "Saved Season · refined"
    assert refined.params.target_seconds == 7200.0
    assert refined.params.zip_sha256 == previous.params.zip_sha256
    assert refined.params.llm_provider == "ollama"
    assert refined.params.llm_model == "saved-local-model:latest"
    assert refined.params.ollama_base_url == "http://127.0.0.1:11434"
    assert refined.params.season_number == 2
    assert refined.params.transcript_source == "subtitles"
    assert refined.params.whisper_device == "cuda"
    assert refined.params.whisper_compute == "float16"
    assert refined.stages.ingest.status == refined.stages.transcribe.status == "done"
    assert refined.stages.ingest.completed == refined.stages.transcribe.completed == ["E01", "E02"]
    assert refined.stages.ingest.pending == refined.stages.transcribe.pending == []
    for name in ("scenes", "nlp", "select", "render"):
        assert getattr(refined.stages, name).status == "pending"
    for substage in ("characters", "baseline", "llm"):
        assert getattr(refined.stages.nlp, substage).status == "pending"
    assert not refined.stages.render.video_done
    assert not refined.stages.render.subtitles_done
    assert list((directory / "output").iterdir()) == []
    assert read_json(directory / "state.json", JobState) == refined
    assert {
        path.relative_to(source_dir): path.read_bytes()
        for path in source_dir.rglob("*")
        if path.is_file()
    } == before
    for original, new in zip(
        previous.stages.ingest.episodes, refined.stages.ingest.episodes, strict=True
    ):
        assert new.source_path.is_absolute()
        assert new.source_path.parent == directory / "episodes"
        assert new.source_path != original.source_path
        assert new.source_path.read_bytes() == original.source_path.read_bytes()
        assert new.episode == original.episode
        assert new.episode_number == original.episode_number
        for relative in (
            f"audio/{new.episode}.wav",
            f"audio/{new.episode}.rms.json",
            f"transcripts/{new.episode}.transcript.json",
            f"transcripts/{new.episode}.characters.json",
            f"transcripts/{new.episode}.credits.json",
        ):
            assert (directory / relative).read_bytes() == (source_dir / relative).read_bytes()


def test_refined_job_uses_current_fast_strategy_independent_of_legacy_environment(
    saved_season: tuple[Path, JobState], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refinement explicitly enables the new scene, score, selection, and encoder defaults."""
    jobs_dir, previous = saved_season
    monkeypatch.setenv("SCENE_METHOD", "visual")
    monkeypatch.setenv("SELECTION_STRATEGY", "density")
    monkeypatch.setenv("EXCLUDE_CREDITS", "false")
    monkeypatch.setenv("LLM_BATCH_SIZE", "1")
    monkeypatch.setenv("RENDER_ENCODER", "libx264")
    refined = controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    assert refined.params.scene_method == "subtitles"
    assert refined.params.selection_strategy == "story"
    assert refined.params.exclude_credits is True
    assert refined.params.llm_batch_size == 8
    assert refined.params.render_encoder == "auto"


def test_refinement_accepts_target_and_required_story_moment(
    saved_season: tuple[Path, JobState],
) -> None:
    """The requested moment and revised runtime are persisted on the new edit only."""
    jobs_dir, previous = saved_season
    ranges = [RequiredRange(episode="E02", start=20.0, end=25.0)]
    refined = controller.create_improved_job(
        previous.job_id, jobs_dir=jobs_dir, target_minutes=100.0, must_keep_ranges=ranges
    )
    assert refined.params.target_seconds == 6000.0
    assert refined.params.must_keep_ranges == ranges
    assert controller.get_status(previous.job_id, jobs_dir=jobs_dir) == previous


@pytest.mark.parametrize(
    "interval",
    [
        RequiredRange(episode="E03", start=20.0, end=25.0),
        RequiredRange(episode="E01", start=85.0, end=95.0),
    ],
)
def test_unknown_or_out_of_source_required_moment_does_not_create_a_job(
    saved_season: tuple[Path, JobState], interval: RequiredRange
) -> None:
    """Required moments must refer to a saved episode and stay within its source runtime."""
    jobs_dir, previous = saved_season
    with pytest.raises(ValueError, match="required moment|known episode"):
        controller.create_improved_job(
            previous.job_id, jobs_dir=jobs_dir, must_keep_ranges=[interval]
        )
    assert [path.name for path in jobs_dir.iterdir()] == [previous.job_id]


def test_refinement_inherits_required_moments_and_explicit_empty_list_clears_them(
    saved_season: tuple[Path, JobState],
) -> None:
    """Omitted required ranges preserve earlier choices; an explicit empty list resets them."""
    jobs_dir, previous = saved_season
    inherited = controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    cleared = controller.create_improved_job(
        previous.job_id, jobs_dir=jobs_dir, must_keep_ranges=[]
    )
    assert inherited.params.must_keep_ranges == previous.params.must_keep_ranges
    assert cleared.params.must_keep_ranges == []


@pytest.mark.parametrize("stage", ["ingest", "transcribe"])
def test_missing_capture_prerequisites_do_not_create_an_orphan_job(
    saved_season: tuple[Path, JobState], stage: str
) -> None:
    """Only a completed source import and dialogue capture can be refined."""
    jobs_dir, previous = saved_season
    getattr(previous.stages, stage).status = "in_progress"
    write_json(jobs_dir / previous.job_id / "state.json", previous)
    with pytest.raises(ValueError, match="import|capture|Complete"):
        controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    assert [path.name for path in jobs_dir.iterdir()] == [previous.job_id]


def test_empty_source_manifest_does_not_create_an_orphan_job(
    saved_season: tuple[Path, JobState],
) -> None:
    """A completed flag without source episodes cannot produce an unusable new edit."""
    jobs_dir, previous = saved_season
    previous.stages.ingest.episodes = []
    write_json(jobs_dir / previous.job_id / "state.json", previous)
    with pytest.raises(ValueError, match="import|capture|Complete"):
        controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    assert [path.name for path in jobs_dir.iterdir()] == [previous.job_id]


@pytest.mark.parametrize(
    "relative",
    ["episodes/E02.mkv", "audio/E02.wav", "audio/E02.rms.json", "transcripts/E02.transcript.json"],
)
def test_missing_required_source_artifact_is_rejected_before_destination_creation(
    saved_season: tuple[Path, JobState], relative: str
) -> None:
    """Preflight checks cover later episodes before any destination files are committed."""
    jobs_dir, previous = saved_season
    (jobs_dir / previous.job_id / relative).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    assert [path.name for path in jobs_dir.iterdir()] == [previous.job_id]


def test_missing_optional_character_and_credit_caches_still_allow_refinement(
    saved_season: tuple[Path, JobState],
) -> None:
    """Earlier builds without optional caches can recompute them on the new edit."""
    jobs_dir, previous = saved_season
    source_dir = jobs_dir / previous.job_id
    for episode in previous.stages.ingest.episodes:
        for name in ("characters", "credits"):
            (source_dir / "transcripts" / f"{episode.episode}.{name}.json").unlink()
    refined = controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    assert refined.stages.transcribe.status == "done"
    assert refined.stages.nlp.characters.status == "pending"


def test_hardlink_unsupported_falls_back_to_independent_media_copies(
    saved_season: tuple[Path, JobState], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filesystems without hard-link support receive ordinary playable-source copies."""
    jobs_dir, previous = saved_season
    calls: list[tuple[Path, Path]] = []

    def unsupported(source: Path, destination: Path) -> None:
        calls.append((source, destination))
        raise OSError("Hard links unavailable")

    monkeypatch.setattr(controller.os, "link", unsupported)
    refined = controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    assert len(calls) == 4
    for original, new in zip(
        previous.stages.ingest.episodes, refined.stages.ingest.episodes, strict=True
    ):
        assert not original.source_path.samefile(new.source_path)
        assert original.source_path.read_bytes() == new.source_path.read_bytes()
        source_audio = jobs_dir / previous.job_id / "audio" / f"{original.episode}.wav"
        target_audio = jobs_dir / refined.job_id / "audio" / f"{new.episode}.wav"
        assert not source_audio.samefile(target_audio)
        assert source_audio.read_bytes() == target_audio.read_bytes()


def test_refined_media_remains_accessible_after_original_directory_entries_are_deleted(
    saved_season: tuple[Path, JobState],
) -> None:
    """Local hard links reuse bytes without relying on the original file paths at resume."""
    jobs_dir, previous = saved_season
    refined = controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    for original, new in zip(
        previous.stages.ingest.episodes, refined.stages.ingest.episodes, strict=True
    ):
        expected_video = original.source_path.read_bytes()
        original.source_path.unlink()
        assert new.source_path.read_bytes() == expected_video
        source_audio = jobs_dir / previous.job_id / "audio" / f"{original.episode}.wav"
        target_audio = jobs_dir / refined.job_id / "audio" / f"{new.episode}.wav"
        expected_audio = source_audio.read_bytes()
        source_audio.unlink()
        assert target_audio.read_bytes() == expected_audio
    assert controller.get_status(refined.job_id, jobs_dir=jobs_dir) == refined


def test_mutable_checkpoint_json_is_copied_and_new_changes_do_not_touch_original(
    saved_season: tuple[Path, JobState],
) -> None:
    """JSON checkpoints have independent storage even when media bytes use hard links."""
    jobs_dir, previous = saved_season
    source_dir = jobs_dir / previous.job_id
    refined = controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir)
    directory = jobs_dir / refined.job_id
    for relative in (
        "audio/E01.rms.json",
        "transcripts/E01.transcript.json",
        "transcripts/E01.characters.json",
        "transcripts/E01.credits.json",
    ):
        original, new = source_dir / relative, directory / relative
        before = original.read_bytes()
        assert not original.samefile(new)
        new.write_text("changed checkpoint", encoding="utf-8")
        assert original.read_bytes() == before


@pytest.mark.parametrize("minutes", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_refined_target_is_rejected_without_creating_a_job(
    saved_season: tuple[Path, JobState], minutes: float
) -> None:
    """Invalid runtime requests fail validation before directories or media are created."""
    jobs_dir, previous = saved_season
    with pytest.raises(ValueError):
        controller.create_improved_job(previous.job_id, jobs_dir=jobs_dir, target_minutes=minutes)
    assert [path.name for path in jobs_dir.iterdir()] == [previous.job_id]
