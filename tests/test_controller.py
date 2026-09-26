"""Persistent job state, mid-ingestion recovery, and controller boundaries."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from sts.pipeline.controller import get_status, list_jobs, resume_job, run_job
from sts.schemas import write_json


def test_run_persists_parameters_and_leaves_later_stages_pending(
    tmp_path: Path, make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> None:
    """An ingest-only job never claims that transcription or downstream work ran."""
    state = run_job(make_zip([("ep1.mp4", b"one")]), 2, name="Test Show")
    loaded = get_status(state.job_id)
    assert loaded == state
    assert loaded.params.target_seconds == 120.0
    assert loaded.params.name == "Test Show"
    assert loaded.params.zip_path.is_absolute()
    assert loaded.params.zip_sha256 is not None
    assert loaded.stages.ingest.status == "done"
    for name in ("transcribe", "scenes", "nlp", "select", "render"):
        assert getattr(loaded.stages, name).status == "pending"
    assert loaded.errors == []
    assert probe_calls == [b"one"]
    assert all(
        (tmp_path / "data/jobs" / state.job_id / name).is_dir()
        for name in ("episodes", "audio", "transcripts", "output")
    )


def test_mid_episode_failure_resumes_without_reprobing_completed_episode(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second-episode failure preserves E01 and resumes only unfinished units."""
    archive = make_zip([("ep3.mp4", b"three"), ("ep1.mp4", b"one"), ("ep2.mp4", b"two")])
    probe = subprocess.run

    def fail_second(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if Path(args[-1]).read_bytes() == b"two":
            raise subprocess.TimeoutExpired(args, 120)
        return probe(args, **kwargs)

    monkeypatch.setattr("sts.pipeline.ingest.subprocess.run", fail_second)
    with pytest.raises(subprocess.TimeoutExpired):
        run_job(archive)
    failed = list_jobs()[0]
    assert failed.stages.ingest.status == "failed"
    assert failed.stages.ingest.completed == ["E01"]
    assert failed.stages.ingest.pending == ["E02", "E03"]
    assert len(failed.errors) == 1
    assert failed.errors[0].episode == "E02"
    completed = tmp_path / "data/jobs" / failed.job_id / "episodes/E01.mp4"
    modified = completed.stat().st_mtime_ns
    monkeypatch.setattr("sts.pipeline.ingest.subprocess.run", probe)
    resumed = resume_job(failed.job_id)
    assert resumed.stages.ingest.status == "done"
    assert resumed.stages.ingest.completed == ["E01", "E02", "E03"]
    assert resumed.stages.ingest.pending == []
    assert len(resumed.errors) == 1
    assert completed.stat().st_mtime_ns == modified
    assert probe_calls == [b"one", b"two", b"three"]


def test_completed_job_resume_skips_all_io_work(
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed ingestion remains inspectable/resumable without the original ZIP or FFprobe."""
    archive = make_zip([("ep1.mp4", b"one")])
    state = run_job(archive)
    archive.unlink()
    monkeypatch.setattr("sts.pipeline.ingest.shutil.which", lambda name: None)
    assert resume_job(state.job_id) == state
    assert probe_calls == [b"one"]


@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_unfinished_checkpoint_restores_damaged_episode_from_another_working_directory(
    tmp_path: Path,
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    """Unfinished jobs verify completed bytes and restore missing or changed media on resume."""
    archive = make_zip([("ep1.mp4", b"one"), ("ep2.mp4", b"two")])
    jobs_dir = tmp_path / "jobs"
    state = run_job(archive, jobs_dir=jobs_dir)
    # Simulate interruption after the last episode checkpoint, before stage completion.
    state.stages.ingest.status = "in_progress"
    write_json(jobs_dir / state.job_id / "state.json", state)
    first = state.stages.ingest.episodes[0].source_path
    if damage == "missing":
        first.unlink()
    else:
        first.write_bytes(b"bad")  # Same size: CRC verification must detect the change.
    probe_calls.clear()
    other_directory = tmp_path / "elsewhere"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    resumed = resume_job(state.job_id, jobs_dir=jobs_dir)
    assert resumed.stages.ingest.status == "done"
    assert resumed.stages.ingest.completed == ["E01", "E02"]
    assert first.read_bytes() == b"one"
    assert probe_calls == [b"one"]


def test_changed_zip_cannot_mix_with_existing_checkpoint(
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume must use the exact archive saved at the start of the job."""
    archive = make_zip([("ep1.mp4", b"original")])
    monkeypatch.setattr("sts.pipeline.ingest.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError):
        run_job(archive)
    job_id = list_jobs()[0].job_id
    make_zip([("ep1.mp4", b"changed")])
    with pytest.raises(ValueError, match="Source ZIP changed"):
        resume_job(job_id)
    assert get_status(job_id).stages.ingest.completed == []
    assert probe_calls == []


def test_missing_input_is_recorded_as_failed_job(tmp_path: Path) -> None:
    """A bad input path leaves an inspectable error instead of an incomplete mystery directory."""
    with pytest.raises(FileNotFoundError):
        run_job(tmp_path / "missing.zip")
    state = list_jobs()[0]
    assert state.stages.ingest.status == "failed"
    assert len(state.errors) == 1


@pytest.mark.parametrize("minutes", [0, -1, float("nan"), float("inf")])
def test_invalid_target_does_not_create_job(tmp_path: Path, minutes: float) -> None:
    """Reject invalid budgets before creating job directories."""
    with pytest.raises(ValidationError):
        run_job(tmp_path / "show.zip", minutes)
    assert list_jobs() == []


@pytest.mark.parametrize("job_id", ["../outside", "..\\outside", "/absolute", "C:drive", ""])
def test_job_lookup_cannot_escape_root(job_id: str) -> None:
    """Only safe single-component job IDs can resolve to saved state files."""
    with pytest.raises(ValueError, match="Invalid job ID"):
        get_status(job_id)


def test_list_jobs_is_newest_first_and_missing_root_is_empty(
    make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> None:
    """Job discovery handles a clean workspace and returns newest runs first."""
    assert list_jobs() == []
    archive = make_zip([("ep1.mp4", b"one")])
    first = run_job(archive)
    second = run_job(archive)
    assert [job.job_id for job in list_jobs()] == [second.job_id, first.job_id]


def test_later_stage_is_not_executed(tmp_path: Path) -> None:
    """The Phase 1 controller refuses requests for unimplemented stages."""
    with pytest.raises(ValueError, match="Only --stage ingest"):
        run_job(tmp_path / "show.zip", stage="render")
    assert list_jobs() == []
