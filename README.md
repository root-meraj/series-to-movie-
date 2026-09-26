# Season to Screen

Turn a ZIP of TV episodes into a condensed movie through a resumable NLP pipeline.
The source of truth for contracts and phase boundaries is [AGENTS.md](AGENTS.md).

## Current phase

Phase 1 implements validated settings, all JSON schemas, ZIP ingestion, and a
controller/CLI skeleton with per-episode resume. Transcription and later pipeline
stages remain placeholders. Runs currently stop after ingestion.

## Development setup

Use Python 3.10+ and install FFmpeg (including FFprobe) on PATH. On Windows, use
`choco install ffmpeg` or `winget install Gyan.FFmpeg`; on macOS, use
`brew install ffmpeg`; on Debian/Ubuntu, use `sudo apt install ffmpeg`.

From this directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pytest ruff rich "pydantic>=2,<3" "pydantic-settings>=2,<3" python-dotenv
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts/make_fixture.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.
These lightweight dependencies are sufficient for Phase 1. Install the complete
pipeline dependencies with `python -m pip install -r requirements.txt` using the
virtual environment when implementing subsequent phases. `pydantic-settings`
is included for Pydantic 2 environment-backed settings.

Copy `.env.example` to `.env` to customize settings. Nonsecret settings also support
process environment overrides. `OPENAI_API_KEY` is read only from `.env`, excluded
from settings serialization, and never saved to job state. Never commit secrets.
Model setup belongs to the phases that use models; Phase 1 downloads none.

## Ingest a season

```powershell
.\.venv\Scripts\sts.exe run --zip tests/fixtures/show.zip --stage ingest --target-minutes 2 --name "Fixture Show"
.\.venv\Scripts\sts.exe status --job JOB_ID
.\.venv\Scripts\sts.exe resume --job JOB_ID
.\.venv\Scripts\sts.exe list
```

Alternatively, use `python -m sts.app.cli` with the virtual environment's Python.
Each command accepts `--jobs-dir PATH` (default: `data/jobs`). Status without
`--job` lists saved jobs. The CLI reports the generated job ID; logs go to stderr.

The fixture becomes `E01.mp4` (show.S01E01), `E02.mkv` (show_1x02), and `E03.mp4`
(ep10). Nested ZIP folders are supported, and original container extensions are
preserved. Non-video entries are ignored. A missing video stream, duration below
60 seconds, invalid ZIP, unsafe archive path, or failed probe fails the stage and
records the error in `state.json`.

Ingestion commits each episode only after validation and writes state atomically.
If interrupted, `resume` skips completed episodes whose bytes still match the ZIP.
Missing or damaged completed files are restored while resuming an unfinished stage.
The source ZIP checksum prevents mixing different archives in one job. Completed
stages are skipped even if the original ZIP is no longer present. There is no
support yet for concurrent processes writing the same job.

## Fixtures and checks

`scripts/make_fixture.py` generates `tests/fixtures/ep_test.mp4` (90 seconds,
640×360, 30 fps, test pattern and sine audio) and `tests/fixtures/show.zip`.
The ZIP contains `ep10.mp4`, `show_1x02.mkv`, and `show.S01E01.mp4`, in that order.
The three entries reuse the same synthetic content, with a full-file remux for MKV.
Use `--output-dir PATH` to change the destination. Generated media is ignored by Git.

The tests cover settings, JSON contracts, atomic state writes, episode ordering,
unsafe/corrupt input, CLI behavior, and recovery after a mid-ingestion failure.
Unit tests mock FFprobe and use no ML models or external APIs. Normal pytest exit
codes apply; the temporary Phase 0 empty-suite hook has been removed.

Jobs are created beneath `data/jobs/` with `episodes`, `audio`, `transcripts`, and
`output` directories. Only `episodes` and `state.json` are populated in Phase 1.
