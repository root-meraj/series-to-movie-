# Season to Screen

Turn a ZIP of TV episodes into a condensed movie through a resumable NLP pipeline.
The source of truth for contracts and phase boundaries is [AGENTS.md](AGENTS.md).

## Current phase

Phase 4 adds text cleanup, character extraction and alias merging, and a cached
embedding wrapper. Runs stop after producing `characters.json` and adding character
mentions to `scenes.json`. NLP scoring, selection, and rendering remain placeholders.

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
.\.venv\Scripts\python.exe -m pip install spacy rapidfuzz sentence-transformers
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts/make_fixture.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.
These dependencies are sufficient through Phase 4. Install the complete
pipeline dependencies with `python -m pip install -r requirements.txt` using the
virtual environment when implementing subsequent phases. `pydantic-settings`
is included for Pydantic 2 environment-backed settings.

Copy `.env.example` to `.env` to customize settings. Nonsecret settings also support
process environment overrides. `OPENAI_API_KEY` is read only from `.env`, excluded
from settings serialization, and never saved to job state. Never commit secrets.
Whisper is loaded lazily and cached for reuse across episodes. The configured model
(`small` by default) downloads into the Hugging Face cache the first time it is used.
For CPU execution, set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE=int8` in `.env`.
spaCy NER uses the English model installed during setup. Sentence embeddings use
`all-MiniLM-L6-v2` by default, downloaded into the Hugging Face cache on first use.
Both trained NLP models are loaded lazily and reused within a process.

## Process a season through character preparation

```powershell
.\.venv\Scripts\sts.exe run --zip tests/fixtures/show.zip --target-minutes 2 --name "Fixture Show"
.\.venv\Scripts\sts.exe status --job JOB_ID
.\.venv\Scripts\sts.exe resume --job JOB_ID
.\.venv\Scripts\sts.exe list
```

Alternatively, use `python -m sts.app.cli` with the virtual environment's Python.
Each command accepts `--jobs-dir PATH` (default: `data/jobs`). Status without
`--job` lists saved jobs. The CLI reports the generated job ID; logs go to stderr.
Both `run` and `resume` default to `--stage characters`. To stop earlier, pass
`--stage ingest`, `--stage transcribe`, or `--stage scenes` to either command.
The job list shows progress through character preparation. The overall NLP status
remains `in_progress` after character preparation finishes because the scoring
phases have not been implemented yet.

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
are kept; first/last flags refer to the season's endpoints.

Character preparation removes uppercase speaker labels at line starts, collapses
whitespace, splits sentences, and counts spaCy PERSON entities. It checkpoints all
raw name counts per episode in `transcripts/E##.characters.json`. Resume reuses
completed episodes and repairs invalid caches while preparation is unfinished.
The season-wide table combines counts before merging similar names and retaining
the top `MAX_CHARACTERS` (default 12, saved per job).

Alias merging requires a case-insensitive token-sort similarity strictly above
88. The most frequent spelling becomes canonical, with longer names and then
alphabetical order breaking ties. Nicknames and first-name relationships are not
inferred. All retained characters have `is_main: true`. Scene character lists match
canonical names or observed aliases as whole words in dialogue; they represent
mentions, not a claim that those characters are visible on screen. A silent
fixture correctly produces an empty character table.

The embedding wrapper returns normalized float32 vectors and reuses its model
across calls. Blank dialogue receives a zero vector; an empty list returns an
empty `(0, 0)` array without loading a model. Scoring will use these embeddings
in later phases.

## Fixtures and checks

`scripts/make_fixture.py` generates `tests/fixtures/ep_test.mp4` (90 seconds,
640×360, 30 fps, test pattern and sine audio) and `tests/fixtures/show.zip`.
The ZIP contains `ep10.mp4`, `show_1x02.mkv`, and `show.S01E01.mp4`, in that order.
The three entries reuse the same synthetic content, with a full-file remux for MKV.
Use `--output-dir PATH` to change the destination. Generated media is ignored by Git.

The tests cover settings, JSON contracts, atomic state writes, episode ordering,
unsafe/corrupt input, CLI behavior, and recovery after a mid-ingestion failure.
Unit tests mock FFprobe, Whisper, trained spaCy NER, and sentence transformers.
Trained model construction is blocked during unit tests. They include generator
failures, filter boundaries, RMS for partial seconds,
disk-write failures, and mid-season recovery. Normal pytest exit codes apply.
Scene tests cover uncut footage, short-scene merging, midpoint/crossing/tie cases,
silent scenes, overlapping dialogue, fractional RMS windows, and crash recovery
between writing scene records and updating the checkpoint.
NLP tests cover speaker cleanup, sentence boundaries, PERSON filtering, alias
thresholds, season-wide ranking, whole-word scene mentions, checkpoint recovery,
embedding normalization, silent inputs, and singleton reuse.

Real-model integration smoke tests are skipped by default. Run them explicitly:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_transcribe_smoke.py --run-model-smoke -s
.\.venv\Scripts\python.exe -m pytest tests/test_nlp_smoke.py --run-model-smoke -s
```

The transcription smoke test processes the fixture ZIP with `small` Whisper on CPU/int8
and retains its job under `data/jobs/`. It may download the model on first use. The
synthetic sine-wave fixture does not test transcription quality on spoken dialogue.
The NLP smoke test uses synthetic dialogue to verify real spaCy character output
and 384-dimensional MiniLM embeddings, including cache reuse. It may download
MiniLM on first use; install the spaCy model during setup first.

Jobs are created beneath `data/jobs/` with `episodes`, `audio`, `transcripts`, and
`output` directories. Through Phase 4, episodes, WAV/RMS files, transcripts,
character checkpoints, `characters.json`, `scenes.json`, and `state.json` are populated.
No condensed video or subtitles are
rendered yet.
