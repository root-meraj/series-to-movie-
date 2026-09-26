# Season to Screen

Turn a ZIP of TV episodes into a condensed movie through a resumable NLP pipeline.
The source of truth for contracts and phase boundaries is [AGENTS.md](AGENTS.md).

## Current phase

Phase 3 adds scene detection, minimum-duration merging, and dialogue/audio
alignment. Settings, schemas, ingestion, transcription, and per-episode resume are
implemented. Runs stop after producing `scenes.json`; NLP scoring, selection,
and rendering remain placeholders.

## Development setup

Use Python 3.10+ and install FFmpeg (including FFprobe) on PATH. On Windows, use
`choco install ffmpeg` or `winget install Gyan.FFmpeg`; on macOS, use
`brew install ffmpeg`; on Debian/Ubuntu, use `sudo apt install ffmpeg`.

From this directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pytest ruff rich "pydantic>=2,<3" "pydantic-settings>=2,<3" python-dotenv
.\.venv\Scripts\python.exe -m pip install faster-whisper numpy soundfile
.\.venv\Scripts\python.exe -m pip install "scenedetect[opencv]"
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts/make_fixture.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.
These dependencies are sufficient through Phase 3. Install the complete
pipeline dependencies with `python -m pip install -r requirements.txt` using the
virtual environment when implementing subsequent phases. `pydantic-settings`
is included for Pydantic 2 environment-backed settings.

Copy `.env.example` to `.env` to customize settings. Nonsecret settings also support
process environment overrides. `OPENAI_API_KEY` is read only from `.env`, excluded
from settings serialization, and never saved to job state. Never commit secrets.
Whisper is loaded lazily and cached for reuse across episodes. The configured model
(`small` by default) downloads into the Hugging Face cache the first time it is used.
For CPU execution, set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE=int8` in `.env`.

## Process a season through scene alignment

```powershell
.\.venv\Scripts\sts.exe run --zip tests/fixtures/show.zip --stage scenes --target-minutes 2 --name "Fixture Show"
.\.venv\Scripts\sts.exe status --job JOB_ID
.\.venv\Scripts\sts.exe resume --job JOB_ID
.\.venv\Scripts\sts.exe list
```

Alternatively, use `python -m sts.app.cli` with the virtual environment's Python.
Each command accepts `--jobs-dir PATH` (default: `data/jobs`). Status without
`--job` lists saved jobs. The CLI reports the generated job ID; logs go to stderr.
Both `run` and `resume` default to `--stage scenes`. To stop earlier, pass
`--stage ingest` or `--stage transcribe` to either command. The job list shows
all three implemented stage statuses.

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

Transcription extracts each episode's first audio stream to a 16 kHz mono PCM WAV,
decodes it once, and passes the waveform to Whisper with VAD enabled. It removes
low-confidence/no-speech segments and consecutive duplicate text. Silent or
music-only input may correctly produce an empty transcript. A video with no audio
stream fails with an error recorded in state.

Each episode produces `audio/E##.wav`, `audio/E##.rms.json`, and
`transcripts/E##.transcript.json`. RMS values cover one-second windows, including
the final partial second. An episode is checkpointed only after all three files
are committed. Resume validates artifacts for completed episodes in an unfinished
stage and regenerates missing or invalid outputs. Saved Whisper settings are used
even if `.env` changes; start a new job to use a different model configuration.

Scene detection retains uncut videos as a single scene. Short scenes merge into
neighbors without dropping footage. Detection threshold and minimum scene length
are saved per job and reused on resume. The stage writes `scenes.json` and records
progress after each episode, so a failure partway through a season does not force
redetection of completed episodes.

Scene alignment assigns contained dialogue by midpoint and crossing dialogue by
largest overlap. Ties divide words proportionally. Dialogue ratios count actual
speech time in each scene, merging overlapping speech intervals to avoid double
counting. Audio RMS is weighted for fractional-second boundaries. Silent scenes
are kept; first/last flags refer to the season's endpoints, and character lists
remain empty until Phase 4.

## Fixtures and checks

`scripts/make_fixture.py` generates `tests/fixtures/ep_test.mp4` (90 seconds,
640×360, 30 fps, test pattern and sine audio) and `tests/fixtures/show.zip`.
The ZIP contains `ep10.mp4`, `show_1x02.mkv`, and `show.S01E01.mp4`, in that order.
The three entries reuse the same synthetic content, with a full-file remux for MKV.
Use `--output-dir PATH` to change the destination. Generated media is ignored by Git.

The tests cover settings, JSON contracts, atomic state writes, episode ordering,
unsafe/corrupt input, CLI behavior, and recovery after a mid-ingestion failure.
Unit tests mock FFprobe and Whisper, and model construction is blocked during unit
tests. They include generator failures, filter boundaries, RMS for partial seconds,
disk-write failures, and mid-season recovery. Normal pytest exit codes apply.
Scene tests cover uncut footage, short-scene merging, midpoint/crossing/tie cases,
silent scenes, overlapping dialogue, fractional RMS windows, and crash recovery
between writing scene records and updating the checkpoint.

The real-model integration smoke test is skipped by default. Run it explicitly:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_transcribe_smoke.py --run-model-smoke -s
```

This processes the fixture ZIP through the real CLI with `small` Whisper on CPU/int8
and retains its job under `data/jobs/`. It may download the model on first use. The
synthetic sine-wave fixture does not test transcription quality on spoken dialogue.

Jobs are created beneath `data/jobs/` with `episodes`, `audio`, `transcripts`, and
`output` directories. Through Phase 3, episodes, WAV/RMS files, transcripts,
`scenes.json`, and `state.json` are populated. No condensed video or subtitles are
rendered yet.
