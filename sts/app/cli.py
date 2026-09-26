"""Command-line ingestion and job inspection; later stages are not implemented yet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from sts.pipeline.controller import get_status, list_jobs, resume_job, run_job


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sts", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Create a job (Phase 1: ingestion only)")
    run.add_argument("--zip", type=Path, required=True)
    run.add_argument("--target-minutes", type=float, default=None)
    run.add_argument("--name")
    run.add_argument("--stage", choices=["ingest"], default="ingest")
    resume = commands.add_parser("resume", help="Resume ingestion of a saved job")
    resume.add_argument("--job", required=True)
    status = commands.add_parser("status", help="Show state JSON or list all jobs")
    status.add_argument("--job")
    listing = commands.add_parser("list", help="List saved jobs")
    for command in (run, resume, status, listing):
        command.add_argument("--jobs-dir", type=Path, default=Path("data/jobs"))
    return parser


def _show_jobs(console: Console, jobs_dir: Path) -> None:
    table = Table("Job", "Name", "Created", "Ingest", "Episodes")
    for state in list_jobs(jobs_dir=jobs_dir):
        table.add_row(
            state.job_id,
            state.params.name or "",
            state.created_at.isoformat(),
            state.stages.ingest.status,
            str(state.stages.ingest.episode_count),
        )
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    """Dispatch the CLI and return a nonzero exit code for invalid or failed jobs."""
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), markup=False)],
    )
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            state = run_job(
                args.zip,
                args.target_minutes,
                name=args.name,
                jobs_dir=args.jobs_dir,
                stage=args.stage,
            )
        elif args.command == "resume":
            state = resume_job(args.job, jobs_dir=args.jobs_dir)
        elif args.command == "status" and args.job:
            state = get_status(args.job, jobs_dir=args.jobs_dir)
            console.print_json(state.model_dump_json(exclude_none=True))
            return 0
        else:
            _show_jobs(console, args.jobs_dir)
            return 0
        console.print(
            f"Job {state.job_id}: ingest done ({state.stages.ingest.episode_count} episodes). "
            "Stopped at Phase 1; later stages remain pending.",
            markup=False,
        )
        return 0
    except Exception as exc:
        console.print(f"Error: {exc}", style="red", markup=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
