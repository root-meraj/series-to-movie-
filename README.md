# Season to Screen

Turn a ZIP of TV episodes into a condensed movie through a resumable NLP pipeline.
The source of truth for contracts and phase boundaries is [AGENTS.md](AGENTS.md).

## Current phase

Phase 0 provides the package layout, configuration templates, dependency list,
lint/test configuration, and synthetic fixture generator. Pipeline modules are
placeholders. Config, schemas, ingestion, and the controller skeleton start in Phase 1.

## Development setup

Use Python 3.10+ and install FFmpeg (including FFprobe) on PATH. On Windows, use
`choco install ffmpeg` or `winget install Gyan.FFmpeg`; on macOS, use
`brew install ffmpeg`; on Debian/Ubuntu, use `sudo apt install ffmpeg`.

From this directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pytest ruff rich
.\.venv\Scripts\python.exe scripts/make_fixture.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.
These lightweight dependencies are sufficient for Phase 0. Install the complete
pipeline dependencies with `python -m pip install -r requirements.txt` using the
virtual environment when implementing subsequent phases. `pydantic-settings`
is included for Pydantic 2 environment-backed settings.

Copy `.env.example` to `.env` to customize settings in Phase 1. Never commit secrets.
Model setup belongs to the phases that use models; the scaffold downloads none.

## Fixtures and checks

`scripts/make_fixture.py` generates `tests/fixtures/ep_test.mp4` (90 seconds,
640×360, 30 fps, test pattern and sine audio) and `tests/fixtures/show.zip`.
The ZIP contains `ep10.mp4`, `show_1x02.mkv`, and `show.S01E01.mp4`, in that order.
The three entries reuse the same synthetic content, with a full-file remux for MKV.
Use `--output-dir PATH` to change the destination. Generated media is ignored by Git.

The Phase 0 test suite is intentionally empty. A temporary hook in
`tests/conftest.py` maps pytest's no-tests-collected status to success; it does not
suppress collection errors or test failures. Remove that hook when Phase 1 adds tests.

Job directories and JSON artifacts will be created at runtime beneath `data/jobs/`
when the corresponding stages are implemented.
