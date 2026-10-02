"""Scene selection, explicit movie rendering, and saved-job inspection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Column, Table

from sts.pipeline.controller import get_status, list_jobs, render_saved_job, resume_job, run_job


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sts", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Create a job (through Phase 7: scene selection)")
    run.add_argument("--zip", type=Path, required=True)
    run.add_argument("--target-minutes", type=float, default=None)
    run.add_argument("--name")
    resume = commands.add_parser("resume", help="Resume a saved job through scene selection")
    resume.add_argument("--job", required=True)
    render = commands.add_parser("render", help="Render or re-render from saved selected.json")
    render.add_argument("--job", required=True)
    render.add_argument(
        "--loudnorm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize clip audio (saved for interrupted render resumes)",
    )
    for command in (run, resume):
        command.add_argument(
            "--stage",
            choices=["ingest", "transcribe", "scenes", "characters", "baseline", "llm", "select"],
            default="select",
        )
    status = commands.add_parser("status", help="Show state JSON or list all jobs")
    status.add_argument("--job")
    listing = commands.add_parser("list", help="List saved jobs")
    for command in (run, resume, render, status, listing):
        command.add_argument("--jobs-dir", type=Path, default=Path("data/jobs"))
    return parser


def _show_jobs(console: Console, jobs_dir: Path) -> None:
    table = Table(
        Column("Job", min_width=12, no_wrap=True),
        "Name",
        "Ingest",
        "Transcribe",
        "Scenes",
        "Characters",
        "Baseline",
        "LLM",
        "Select",
        "Render",
        "Episodes",
    )
    for state in list_jobs(jobs_dir=jobs_dir):
        table.add_row(
            state.job_id,
            state.params.name or "",
            state.stages.ingest.status,
            state.stages.transcribe.status,
            state.stages.scenes.status,
            state.stages.nlp.characters.status,
            state.stages.nlp.baseline.status,
            state.stages.nlp.llm.status,
            state.stages.select.status,
            state.stages.render.status,
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
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
            state = resume_job(args.job, jobs_dir=args.jobs_dir, stage=args.stage)
        elif args.command == "render":
            state = render_saved_job(args.job, jobs_dir=args.jobs_dir, loudnorm=args.loudnorm)
            console.print(
                f"Job {state.job_id}: render done. "
                f"Outputs: {args.jobs_dir / state.job_id / 'output' / 'condensed_movie.mp4'} "
                "and condensed.srt.",
                markup=False,
            )
            return 0
        elif args.command == "status" and args.job:
            state = get_status(args.job, jobs_dir=args.jobs_dir)
            console.print_json(state.model_dump_json(exclude_none=True))
            return 0
        else:
            _show_jobs(console, args.jobs_dir)
            return 0
        console.print(
            f"Job {state.job_id}: {args.stage} done "
            f"({state.stages.ingest.episode_count} episodes). "
            f"Stopped after {args.stage}; later stages remain pending.",
            markup=False,
        )
        return 0
    except Exception as exc:
        console.print(f"Error: {exc}", style="red", markup=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
