"""CLI ingestion, JSON status, listing, and failure exit codes."""

from collections.abc import Callable
from pathlib import Path

import pytest

from sts.app.cli import main
from sts.pipeline.controller import list_jobs


def test_cli_run_status_and_list(
    make_zip: Callable[[list[tuple[str, bytes]]], Path],
    probe_calls: list[bytes],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The Phase 1 CLI runs ingestion and makes the resulting checkpoint inspectable."""
    archive = make_zip([("ep1.mp4", b"one")])
    assert main(["run", "--zip", str(archive), "--stage", "ingest", "--target-minutes", "2"]) == 0
    assert "ingest done" in capsys.readouterr().out
    job_id = list_jobs()[0].job_id
    assert main(["status", "--job", job_id]) == 0
    assert '"target_seconds": 120.0' in capsys.readouterr().out
    assert main(["list"]) == 0
    assert job_id in capsys.readouterr().out


def test_cli_resume_skips_completed_ingestion(
    make_zip: Callable[[list[tuple[str, bytes]]], Path], probe_calls: list[bytes]
) -> None:
    """Resuming via the CLI uses saved parameters and leaves finished episodes untouched."""
    archive = make_zip([("ep1.mp4", b"one")])
    assert main(["run", "--zip", str(archive)]) == 0
    assert main(["resume", "--job", list_jobs()[0].job_id]) == 0
    assert probe_calls == [b"one"]


def test_cli_missing_zip_returns_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """A failed run exits nonzero and reports its error."""
    assert main(["run", "--zip", "missing.zip"]) == 1
    assert "Error:" in capsys.readouterr().out


def test_cli_rejects_unimplemented_stage() -> None:
    """CLI validation cannot advance beyond the authorized build phase."""
    with pytest.raises(SystemExit) as error:
        main(["run", "--zip", "show.zip", "--stage", "render"])
    assert error.value.code == 2
