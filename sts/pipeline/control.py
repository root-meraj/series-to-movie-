"""Cooperative processing stops that leave durable stage checkpoints intact."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from time import monotonic

from sts.schemas import ProcessingControl, read_json, write_json


class ProcessingStopped(BaseException):
    """A requested stop, excluded from ordinary stage failure/error handlers."""


def stop_requested(job_dir: Path) -> bool:
    """Read the independent stop request; legacy jobs without a control file keep running."""
    try:
        return read_json(job_dir / "processing_control.json", ProcessingControl).stop_requested
    except FileNotFoundError:
        return False


def set_stop_requested(job_dir: Path, requested: bool) -> None:
    """Atomically publish a stop request without writing the worker's state.json."""
    write_json(job_dir / "processing_control.json", ProcessingControl(stop_requested=requested))


def check_stop(job_dir: Path) -> None:
    """Interrupt unfinished work when requested, preserving the latest completed checkpoint."""
    if stop_requested(job_dir):
        raise ProcessingStopped("Processing stopped; completed work is saved")


class StopCheck:
    """Limit control-file reads during frame-by-frame processing to four per second."""

    def __init__(self, job_dir: Path) -> None:
        """Create a checker local to one processing operation, without shared mutable state."""
        self.job_dir = job_dir
        self._last_check = float("-inf")

    def __call__(self) -> None:
        """Raise on a stop request at most a quarter-second after the last poll."""
        now = monotonic()
        if now - self._last_check >= 0.25:
            check_stop(self.job_dir)
            self._last_check = now


def run_media(arguments: list[str], job_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run a media tool, terminating and reaping this child when the job requests a stop.

    Existing callers without a control file retain the ordinary subprocess seam. Controlled
    jobs poll every quarter-second; output pipes are drained by communicate, including after
    termination. Only this owned process is stopped, never another project's processes.
    """
    options = dict(text=True, encoding="utf-8", errors="replace")
    if not (job_dir / "processing_control.json").is_file():
        return subprocess.run(arguments, check=True, capture_output=True, **options)
    check_stop(job_dir)
    with subprocess.Popen(
        arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options
    ) as process:
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    check_stop(job_dir)
        except BaseException:
            if process.poll() is None:
                process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise
        result = subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
        result.check_returncode()
        logging.getLogger(__name__).debug(
            "media unit-progress: %s complete", Path(arguments[0]).name
        )
        return result
