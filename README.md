# Season to Screen

Turn a ZIP of TV episodes into a condensed movie through a resumable NLP pipeline.
The source of truth for contracts and phase boundaries is [AGENTS.md](AGENTS.md).

## Current phase

Phase 10 adds a Streamlit editing workspace with ZIP upload, background processing,
live stage progress, saved projects, movie preview, and film/subtitle downloads.
The CLI still runs the complete pipeline, with explicit stage stops available.
Evaluation scripts remain for Phase 11.

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
.\.venv\Scripts\python.exe -m pip install networkx
.\.venv\Scripts\python.exe -m pip install openai
.\.venv\Scripts\python.exe -m pip install "streamlit>=1.64,<2"
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts/make_fixture.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.
These dependencies are sufficient through Phase 10. Install the complete
pipeline dependencies with `python -m pip install -r requirements.txt` using the
virtual environment when implementing subsequent phases. `pydantic-settings`
is included for Pydantic 2 environment-backed settings.

Copy `.env.example` to `.env` to customize settings. Nonsecret settings also support
process environment overrides. `OPENAI_API_KEY` is read only from `.env`, excluded
from settings serialization, and never saved to job state. Never commit secrets.
Whisper is loaded lazily and cached for reuse across episodes. The configured model
(`small` by default) downloads into the Hugging Face cache the first time it is used.
For CPU execution, set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE=int8` in `.env`.
For NVIDIA CUDA on Windows, install the optional runtime wheels into the same
virtual environment, then set `WHISPER_DEVICE=cuda` and `WHISPER_COMPUTE=float16`:

```powershell
.\.venv\Scripts\python.exe -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12
```

The Whisper constructor discovers these wheels' DLL folders and adds them to its
process PATH. It does not change Windows settings or install libraries automatically.
An existing compatible system CUDA installation can still supply the libraries.
spaCy NER uses the English model installed during setup. Sentence embeddings use
`all-MiniLM-L6-v2` by default, downloaded into the Hugging Face cache on first use.
Both trained NLP models are loaded lazily and reused within a process.

## The editing room

Start the website from this directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run sts/app/ui.py
```

Open `http://127.0.0.1:8501`. The interface uses a dark cinematic palette, warm
gold accents, responsive layouts, and native accessible controls. Choose a season
ZIP, name the project, set a maximum runtime, choose an editorial engine, and
select **Create my film**. The finished workspace includes a video preview and
separate MP4/SRT downloads. Silent scenes can produce a valid empty subtitle file.

Preview mode uses MockProvider's sample scores and needs no external LLM service.
OpenAI story-aware editing requires `ENABLE_OPENAI=true` and an API key in `.env`;
Ollama uses the configured local endpoint and the model chosen in advanced settings.
On Windows, an optional standalone Ollama runtime at `.venv/tools/ollama/ollama.exe`
can be started with `pwsh -File scripts/start_local_ollama.ps1`. This helper starts
a hidden loopback service, keeps models and logs under `.venv/tools`, and reuses
its own healthy service when already running. It does not download the runtime or
models. Install the official standalone runtime first, then pull a local model
with `.\.venv\tools\ollama\ollama.exe pull qwen3:1.7b` while that service is running.
Configure `.env` with `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen3:1.7b`, and
`OLLAMA_BASE_URL=http://127.0.0.1:11434`. Leave OpenAI disabled for this local setup.
Cloud editing sends dialogue to the configured provider. Source videos remain in
this workspace. Credentials are never entered in the browser or saved in job state.

Advanced settings select CPU, automatic device detection, or CUDA. The UI defaults
to CPU/int8 unless CUDA was explicitly configured, avoiding a dependency on GPU
runtime libraries. This setting is saved with each new project. CLI defaults are
unchanged. Trained models load only when their stage needs them and are retained
with bounded `st.cache_resource` caches; opening the page does not load weights.

**Dialogue source** defaults to **Automatic · use subtitles first** for new jobs.
Usable embedded text subtitles in `SUBTITLE_LANGUAGE` (English by default) supply
the transcript and original cue timestamps without loading Whisper. If none are
available, Automatic falls back to audio transcription. **Embedded subtitles only**
fails clearly when no usable track exists; **Whisper · transcribe audio** forces ASR.
Image subtitles and subtitles burned into the video are not text tracks and need ASR.
Signs/songs/forced titles and sparse forced tracks are excluded. An incorrectly
forced flag can be overridden for the requested language only when the track has
enough dialogue cues, at least 25% speech coverage, and spans at least 65% of the
episode. Full non-forced dialogue remains preferred. If several full text tracks match, the
extractor chooses the one with the most usable dialogue cues and coverage.

Subtitle dialogue still feeds the existing scene alignment, character extraction,
story scoring, and selection. Selected clips retain whole scene boundaries and
season bookends, so a subtitle cue is not treated as an isolated video cut. Audio
extraction remains necessary for RMS features, and scene detection and final video
encoding still run. Existing jobs retain their saved Whisper-only behavior unless
their source policy is explicitly changed; new jobs save their selected source policy.

Uploads are accepted up to 2 GB. Streamlit buffers the upload, then the application
copies it in chunks to the job's durable `input/source.zip` before starting work.
For larger seasons, select **Use local ZIP** and paste the archive's full path.
On Windows, right-click the ZIP in File Explorer and select **Copy as path**;
surrounding quotes are accepted. Local ZIPs have no browser upload size limit and
are read directly from disk without buffering or copying the archive. Keep the
original ZIP in place and unchanged until **Import episodes** completes. Empty,
missing, or non-ZIP paths keep the create button disabled with a visible explanation. Invalid
ZIP archives do not publish partial jobs.

One background worker serves all browser sessions in this local server, preventing
overlapping model work and duplicate job launches. Stage progress is read from
`state.json` every two seconds while processing. Closing or refreshing the browser
does not stop the worker; reopening a saved project restores its status. After a
server restart, unfinished projects offer **Resume edit** from saved checkpoints.
Use **Stop processing** on a running project to pause the edit and keep completed work.
The same control appears on the new-project page while an older edit is running.
While **Stopping…** is shown, the worker is finishing or interrupting its active operation;
another project can start once it becomes idle. Scene detection checks between frames,
media conversion terminates its own child process, and Whisper checks between decoded
segments. A model loading/inference operation or an active story-model request may need
to finish first; no further unit or JSON retry starts after the request. Stop requests do
not add processing errors. **Resume edit** reuses saved episodes, scene scores, and clips;
the unfinished unit starts again. Files are retained.

Keep the Streamlit server running until processing completes. Avoid running a CLI
writer against the same active job.

Film downloads are prepared on click, so progress refreshes do not repeatedly load
the whole movie into memory. The provided configuration binds to loopback for a
local workspace. This phase does not add multi-user authentication or deployment.

## Process a season into a movie

```powershell
$env:LLM_PROVIDER = "mock"  # Offline fixture scoring; choose before creating a job.
.\.venv\Scripts\python.exe scripts/make_fixture.py --scene-cuts --output-dir tests/fixtures/end-to-end
.\.venv\Scripts\sts.exe run --zip tests/fixtures/end-to-end/show.zip --target-minutes 2 --name "Fixture Show"
.\.venv\Scripts\sts.exe status --job JOB_ID
.\.venv\Scripts\sts.exe resume --job JOB_ID
.\.venv\Scripts\sts.exe list
```

Alternatively, use `python -m sts.app.cli` with the virtual environment's Python.
Each command accepts `--jobs-dir PATH` (default: `data/jobs`). Status without
`--job` lists saved jobs. The CLI reports the generated job ID; logs go to stderr.
Both `run` and `resume` default to `--stage render`. To stop earlier, pass
`--stage ingest`, `--stage transcribe`, `--stage scenes`, `--stage characters`,
`--stage baseline`, `--stage llm`, or `--stage select`. The CLI reports absolute
movie and subtitle paths after rendering. The job list shows progress through
rendering, and overall NLP becomes `done` after LLM scoring.

Use `LLM_PROVIDER=mock` for deterministic offline development. Its scores and
summaries are canned test data. For OpenAI, configure `.env` with
`LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, `ENABLE_OPENAI=true`, and your
`OPENAI_API_KEY`. OpenAI is disabled by default; the key is read only from `.env`.
For Ollama, set `LLM_PROVIDER=ollama`, `LLM_MODEL` to an installed chat model, and
`OLLAMA_BASE_URL` to the running service (default `http://localhost:11434`).
Ollama needs no API key. An OpenAI model name such as `gpt-4o-mini` is not a local
Ollama model; choose a model installed by `ollama pull`, for example `qwen3:1.7b`.
The Qwen3 adapter disables thinking and requests a 32K context for JSON scoring
and episode summaries. On a small GPU, run the Ollama service with
`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, and `OLLAMA_NUM_PARALLEL=1`
to reduce context memory. Longer inputs and local models still require processing time.
`SEASON_NUMBER` defaults to 1 and supplies the editor prompt's season number.

New jobs save their provider, model, endpoint, and season number. Jobs created
before Phase 6 adopt the current settings on their first LLM run. Later changes
to `.env` do not switch an existing job's provider; start a new job for a different
configuration. OpenAI opt-in and credentials are checked at runtime, not persisted
as authorization. Completed stages and fully cached responses need no new API calls.

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

The controller also catches failures that occur before a stage's own error
handler. It reloads the latest state before recording the error, preserving
completed units. Errors already recorded by a stage are not duplicated; a new
failed resume attempt records a new error. A completed job's normal resume skips
every stage without rewriting artifacts or loading models.

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

During transcription, the page identifies the active episode and reports decoded
source minutes. `stages.transcribe.current` records this activity at most every five
seconds, plus activity changes. The bar advances within an episode when Whisper
yields segments; silence may not produce intermediate positions. Only fully saved
episodes enter `completed`, and an interrupted episode restarts from its beginning.
CPU transcription of long episodes can take over an hour each. A two-hour output
target still requires analyzing all source episodes first. After successful ASR,
the Whisper model is released so a local LLM can use the same GPU.

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
empty `(0, 0)` array without loading a model.

Baseline scoring checkpoints scene embeddings one episode at a time in
`transcripts/E##.embeddings.json`, using the saved `SBERT_MODEL`. Blank dialogue
does not require embedding inference. Resume checks the cache against its model,
scene IDs, cleaned text hash, and vector schema before reusing completed episodes.
If graph computation or score publication fails, the saved embeddings avoid
repeating model inference on restart.

TextRank builds one weighted graph across the season, connecting distinct scenes
when their cosine similarity exceeds 0.2. It runs NetworkX PageRank, then min-max
normalizes the results to `[0, 1]`. Silent scenes remain isolated graph nodes.
Singletons and uniform rankings receive zeros because text offers no distinction
between them. The silent fixture therefore correctly receives all-zero baselines.
These are text-centrality scores; audio and plot importance are added in later phases.

`scores.json` preserves scene order and contains `baseline_score` for every scene.
`final_score` remains zero until selection computes the weighted mix. LLM fields are
initially omitted.
`state.stages.nlp.baseline.status` becomes `done` only after scores are committed.
As with earlier stages, a completed baseline is skipped on resume.

LLM scoring first asks for a cumulative summary through the current episode,
limited to 500 words and 6000 characters. It then scores that episode's scenes
using the updated story context and main characters. The next episode receives
the same cumulative context. Scores must be integers from 1 to 10, plot-advance
flags must be booleans, and reasons must contain 1-15 words. Silent scenes are
still scored, with missing visual evidence stated in the prompt.

The parser strips Markdown fences and validates responses with Pydantic. Invalid
JSON, schema violations, and incomplete completions get at most three attempts
per summary or scene. Transport and authentication failures stop the stage with
a recorded error. OpenAI uses [Chat Completions JSON mode](https://developers.openai.com/api/docs/guides/structured-outputs);
Ollama uses the [non-streaming chat API](https://docs.ollama.com/api/chat), with a
five-minute request timeout to accommodate cold local model startup and episode summaries.

Each accepted response is saved immediately in `transcripts/E##.llm.json` before
updating `scores.json`, `rolling_summary.json`, and `state.json`. Resume reuses
valid saved responses, including a response committed just before a crash. Cache
hashes include the prompt inputs and incoming story context; a changed earlier
summary invalidates dependent later caches. A call whose response was never
successfully saved may need repeating. State reports the active episode, whether
its summary is done, and which scenes are complete.

Completed LLM scoring adds `llm_score`, `llm_plot_advance`, and `llm_reason` while
preserving baseline values and scene order. `rolling_summary.json` retains the
cumulative summary after each episode.

Selection min-max normalizes LLM, TextRank, audio RMS, and dialogue ratio across
the season, then mixes them using `SCORE_WEIGHTS` (default `0.5,0.3,0.1,0.1`).
Weights are normalized to sum to one; constant features contribute zero. Missing
LLM scores are rejected when their weight is positive. `scores.json` receives
the resulting `final_score` for every scene without changing its earlier scores.

The selector reserves the complete first and last scenes, then greedily fits
other scenes by score per second. Output follows numeric episode order and scene
start time. Optional scenes shorter than `MIN_CLIP_SECONDS` (default 15) are
excluded. After sorting, a dialogue clip with cosine similarity strictly above
0.92 to the previously kept clip is removed. Silent clips remain valid. A
closing scene removes preceding optional duplicates; forced bookends survive
both filters. Deduplication may leave some of the budget unused.

A target shorter than the combined bookends fails with the minimum required
runtime. The default synthetic fixture is uncut, so its first and last 90-second
scenes require a three-minute target. Use `--scene-cuts` when generating the
two-minute end-to-end fixture: each episode has three 30-second scenes, allowing
both season bookends to fit.
The selector uses full scene intervals and never exceeds the runtime budget.

New jobs save the weights and minimum clip length. Older jobs adopt current
settings once at their first selection attempt. Selection reuses validated
embedding checkpoints from baseline scoring and repairs missing, invalid, or
stale caches one episode at a time. Final scores and `selected.json` are published
before marking selection `done`. An interrupted publication resumes without
repeating completed model inference. Completed selection is skipped on resume.

Normal `run` and `resume` render the selected intervals automatically. Use
`sts render --job JOB_ID` to render again after editing `selected.json`. Each cut re-encodes
video to H.264 (veryfast, CRF 20) and audio to AAC (192 kbps). All clips use a
fixed 1280×720 letterbox canvas, square pixels, yuv420p, 30 fps, a common video
time base, and 48 kHz stereo audio. This preserves display aspect while making
episodes with different shapes compatible for concatenation. Cutting never uses
stream copy; only the concat demuxer joins the uniform clips with `-c copy`.
Concatenation preserves timestamps so AAC encoder priming does not shift the
movie's starting point relative to subtitles.
Both source video and audio streams are required.

Use `sts render --job JOB_ID --loudnorm` to apply
`loudnorm=I=-16:TP=-1.5:LRA=11` to each clip. The flag is saved for interrupted
renders. Omit it to preserve the saved value or use `--no-loudnorm` to disable it.
Rendering invokes no Whisper, spaCy, embedding, or LLM models.

Clip caches live in `output/clips/clip_XXXX.mp4` with paired JSON checkpoints.
Resume validates source size/mtime, clip settings, encoded-byte checksums, and
stream metadata before reusing them. It rebuilds only missing, changed, or damaged
clips. The concat manifest is `output/clips.txt`, with correctly quoted paths.
Encoding and concatenation publish only complete, validated files by atomic
replacement; failures preserve the previous complete destination.

Subtitle export trims dialogue to each retained source interval and adds its
cumulative position in the output. It uses probed clip lengths to avoid drift
from frame rounding. UTF-8 text is retained and timestamps round to milliseconds.
Silent fixture transcripts correctly produce an empty SRT file. Frame/AAC
rounding can make the encoded movie slightly longer than the selection's nominal
runtime. Each cut must match its requested length within 0.15 seconds; the movie
must match the sum of probed clip lengths within 0.15 seconds.

Rendering records clip progress and independent movie/subtitle checkpoints in
`state.json`, becoming `done` only after both outputs are committed. Repeating the
explicit `render` command reopens a completed render and reads any edits to
`selected.json`, while reusing unaffected clips. Edited clips must match known
source episodes and their bounds. The clip array controls their output order.

## Fixtures and checks

`scripts/make_fixture.py` generates `tests/fixtures/ep_test.mp4` (90 seconds,
640×360, 30 fps, test pattern and sine audio) and `tests/fixtures/show.zip`.
The ZIP contains `ep10.mp4`, `show_1x02.mkv`, and `show.S01E01.mp4`, in that order.
The three entries reuse the same synthetic content, with a full-file remux for MKV.
Use `--output-dir PATH` to change the destination. Generated media is ignored by Git.
Add `--scene-cuts` for three solid-color scenes per episode with cuts at 30 and
60 seconds. This optional mode preserves the default test-pattern fixture and
supports a two-minute selection without trimming scenes or dropping bookends.

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
Scoring tests cover weighted graph behavior, the strict similarity threshold,
cross-episode ranking, uniform/disconnected graphs, cache validation, saved model
settings, and recovery from inference, graph, and score-checkpoint interruptions.
LLM tests use MockProvider to cover strict JSON types, fenced responses, retry
limits, episode context, every model-call failure boundary, cache invalidation,
and crashes between response persistence and state publication. Provider adapter
tests mock the transports; no test makes an external LLM request.
Selection tests cover normalized weights, score density, chronological ordering,
minimum lengths, protected endpoints, infeasible budgets, adjacent-only deduplication,
the strict similarity threshold, saved settings, and cache/publication recovery.
Rendering tests verify safe encoding commands, uniform media, quoted concat paths,
subtitle trimming/offsets/Unicode/rounding, cache recovery, output publication
failures, edited selections, optional loudnorm, and a real fully decoded FFmpeg join.
The short media smoke test runs when FFmpeg and FFprobe are available; it needs
no trained models or API calls.
Controller integration tests exercise the actual stages and JSON persistence
with deterministic model/media doubles and MockProvider. They cover each explicit
stop, default full execution, saved settings, interrupted units, repeated failures
before stage error handlers, and completed resume after deleting the source ZIP.
Streamlit AppTest cases exercise native uploads, form settings, saved-project
navigation, progress refresh, failure/reconnect/resume, preview, and deferred
downloads without models or external API requests. Runner tests cover durable
uploads, atomic publication, duplicate launches, and worker recovery. Resource-cache
tests verify lazy loading, cache identity, replacement, and constructor failures.

Real-model integration smoke tests are skipped by default. Run them explicitly:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_transcribe_smoke.py --run-model-smoke -s
.\.venv\Scripts\python.exe -m pytest tests/test_nlp_smoke.py --run-model-smoke -s
.\.venv\Scripts\python.exe -m pytest tests/test_end_to_end_smoke.py --run-model-smoke -s
```

The transcription smoke test processes the fixture ZIP with `small` Whisper on CPU/int8
and retains its job under `data/jobs/`. It may download the model on first use. The
synthetic sine-wave fixture does not test transcription quality on spoken dialogue.
The NLP smoke test uses synthetic dialogue to verify real spaCy character output,
384-dimensional MiniLM embeddings, cache reuse, and baseline score artifacts. It may download
MiniLM on first use; install the spaCy model during setup first.

The end-to-end smoke test generates the cut fixture under
`tests/fixtures/end-to-end`, runs the default CLI with real local Whisper and
MockProvider, and retains its job under `data/jobs/`. It verifies complete state,
scene/score/selection contracts, a fully decoded 120-second H.264/AAC movie, SRT
consistency, and unchanged files after a completed resume. Trained NLP models
are used only when dialogue requires them; the synthetic tone normally yields
empty transcripts and a valid empty SRT. External LLM requests remain blocked.

Jobs are created beneath `data/jobs/` with `episodes`, `audio`, `transcripts`, and
`output` directories. A completed run populates episodes, WAV/RMS files, transcripts,
character/embedding/LLM checkpoints, `characters.json`, `scenes.json`, `scores.json`,
`rolling_summary.json`, `selected.json`, and `state.json`.
Rendering also populates the clip cache, concat manifest,
`output/condensed_movie.mp4`, and `output/condensed.srt`.

## Refine a film and keep important moments

New projects use subtitle story blocks by default. Embedded or supplied subtitles
provide dialogue timestamps; Whisper remains the fallback when subtitles are
unavailable. The editor groups connected footage into complete sequences, scores
several blocks per local model request, and keeps silent intervals eligible.
Confirmed end credits are excluded while uncertain footage and postcredit story
remain available. Full-video visual scene detection is still an advanced option.

On an existing project's page, open **Refine the story**. Select **Keep a specific
moment**, choose its episode, and enter the start and end in minutes. Click
**Create refined film**. This creates a separate edit using saved dialogue and
media, preserves the earlier movie, and starts fresh story scoring and rendering.
The runtime card shows the actual film length alongside the requested target.

Defaults in `.env.example`: `SCENE_METHOD=subtitles`, `SCENE_BLOCK_SECONDS=60`,
`EXCLUDE_CREDITS=true`, `SELECTION_STRATEGY=story`, `LLM_BATCH_SIZE=8`, and
`RENDER_ENCODER=auto`. GPU rendering is used only after a successful hardware
probe; otherwise the editor uses CPU encoding. Old saved jobs retain their saved
behavior; create a refined film to apply these changes.

Subtitle-only scoring cannot recognize every silent visual event. Context
preservation reduces short-shot loss, and required moments let you guarantee a
known sequence remains in a particular edit. Selection targets the requested
length using complete blocks, so the finished duration may be slightly shorter.
