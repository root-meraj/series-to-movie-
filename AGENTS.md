AGENTS.md — SEASON-TO-SCREEN PROJECT SPECIFICATION

You are building "Season to Screen": an NLP pipeline that takes a ZIP of TV-seriesepisodes, transcribes them, detects scenes, scores scene importance with NLP/LLMs,selects the essential scenes to fit a target runtime, and renders a single condensed"movie" video.



Work phase by phase (see §10). After each phase: run tests, report what passed,STOP, and wait for the next instruction. Never skip ahead or batch multiple phases.



1\. HARD RULES

Python 3.10+, full type hints, docstrings on every public function.

All module communication via the JSON schemas in §5 — these are the contract.Never rename keys without updating this file.

Pipeline MUST be resumable. Every stage writes progress to state.json.If a stage crashes, resume continues from the last completed unit.

NEVER cut video with -c copy (keyframe glitches). Re-encode cuts,then use the concat demuxer with -c copy (safe — all clips uniform).

LLM calls only inside pipeline/nlp/scorer\_llm.py behind the LLMProviderinterface. Unit tests MUST use MockProvider — never real API calls.

Secrets via .env only. Never hardcode keys.

All timestamps stored as float seconds. Convert to timecodes only at theffmpeg boundary.

Logging via rich logger — no bare print() outside the CLI.

Use pathlib.Path everywhere. Code must run on Windows, macOS, Linux.

Run ruff check + pytest before declaring any phase done.

2\. STACK

System dependency: ffmpeg + ffprobe (detect OS; install via winget/choco/apt/brew)

Python packages (requirements.txt):faster-whisper, scenedetect\[opencv], openai, sentence-transformers,spacy, rapidfuzz, networkx, numpy, soundfile, pydantic, python-dotenv,tqdm, rich, streamlit, rouge-score, bert-score, pytest, ruff

Model downloads (first run, cache them):

python -m spacy download en\_core\_web\_sm

faster-whisper "small" (downloads automatically to HF cache)

sentence-transformers all-MiniLM-L6-v2

3\. PROJECT STRUCTURE (exact)

season-to-screen/├── AGENTS.md├── requirements.txt├── .env.example├── pyproject.toml              # ruff + pytest config├── sts/                        # package│   ├── \_\_init\_\_.py│   ├── config.py               # pydantic Settings, reads .env│   ├── schemas.py              # pydantic models for ALL JSON contracts (§5)│   ├── pipeline/│   │   ├── controller.py       # state machine, resume logic│   │   ├── ingest.py│   │   ├── transcribe.py│   │   ├── scenes.py│   │   ├── align.py│   │   ├── selector.py│   │   └── render.py│   ├── nlp/│   │   ├── preprocess.py│   │   ├── ner.py│   │   ├── embeddings.py│   │   ├── scorer\_textrank.py│   │   ├── scorer\_llm.py       # LLMProvider + OpenAIProvider + OllamaProvider + MockProvider│   │   └── rolling\_summary.py│   ├── app/│   │   ├── cli.py              # entry: python -m sts.app.cli│   │   └── ui.py               # Streamlit│   └── eval/│       ├── rouge\_eval.py│       └── human\_study.py├── scripts/│   └── make\_fixture.py         # generates synthetic test episode (no real content needed)├── tests/│   ├── conftest.py│   ├── test\_ingest.py│   ├── test\_transcribe.py│   ├── test\_align.py│   ├── test\_scoring.py│   ├── test\_selector.py│   └── test\_render.py└── data/jobs/{job\_id}/    ├── episodes/     # extracted + renamed E01.mp4, E02.mp4 ...    ├── audio/        # E01.wav (16kHz mono)    ├── transcripts/  # E01.transcript.json    ├── scenes.json    ├── characters.json    ├── scores.json    ├── selected.json    ├── state.json    └── output/       # condensed\_movie.mp4, condensed.srt

4\. CONFIG DEFAULTS (.env.example)

WHISPER\_MODEL=small            # tiny|base|small|mediumWHISPER\_DEVICE=auto            # auto|cpu|cudaWHISPER\_COMPUTE=auto           # auto|int8|float16DETECTOR\_THRESHOLD=27.0MIN\_SCENE\_SECONDS=5.0TARGET\_MINUTES=120MIN\_CLIP\_SECONDS=15.0LLM\_PROVIDER=openai            # openai|ollama|mockLLM\_MODEL=gpt-4o-miniOLLAMA\_BASE\_URL=http://localhost:11434OPENAI\_API\_KEY=SBERT\_MODEL=all-MiniLM-L6-v2SCORE\_WEIGHTS=0.5,0.3,0.1,0.1  # llm, textrank, audio\_rms, dialogue\_ratioMAX\_CHARACTERS=12

5\. JSON CONTRACTS (source of truth)

state.json

{  "job\_id": "abc123",  "created\_at": "ISO-8601",  "params": {"target\_seconds": 7200, "whisper\_model": "small", "detector\_threshold": 27.0},  "stages": {    "ingest":     {"status": "done", "episode\_count": 10},    "transcribe": {"status": "in\_progress", "completed": \["E01"], "pending": \["E02"]},    "scenes":     {"status": "pending"},    "nlp":        {"status": "pending"},    "select":     {"status": "pending"},    "render":     {"status": "pending"}  },  "errors": \[{"stage": "transcribe", "episode": "E03", "message": "...", "ts": "..."}]}

transcripts/E01.transcript.json

{"episode": "E01", "language": "en", "segments": \[  {"start": 12.4, "end": 15.9, "text": "We need to talk about the money."}]}

scenes.json (list of)

{  "scene\_id": "E01\_s0042",  "episode": "E01",  "source\_path": "data/jobs/abc/episodes/E01.mp4",  "start": 734.2, "end": 795.8, "duration": 61.6,  "transcript": "We need to talk about the money. ...",  "dialogue\_ratio": 0.83,  "audio\_rms": 0.41,  "characters": \["Walter", "Jesse"],  "is\_first": false, "is\_last": false, "episode\_number": 1}

characters.json

{"characters": \[  {"name": "Walter White", "aliases": \["Walter", "Mr. White"], "mention\_count": 341, "is\_main": true}]}

scores.json (list of)

{  "scene\_id": "E01\_s0042",  "baseline\_score": 0.63,  "llm\_score": 8,  "llm\_plot\_advance": true,  "llm\_reason": "Introduces the central conflict",  "final\_score": 0.81}

selected.json

{  "target\_seconds": 7200,  "selected\_total\_seconds": 7042.3,  "clips": \[    {"scene\_id": "E01\_s0001", "episode": "E01", "source\_path": "...", "start": 0.0, "end": 45.2}  ],  "stats": {"scenes\_kept": 118, "scenes\_dropped": 340, "original\_seconds": 27000.0}}

Phase 1 contract additions and clarifications (existing keys are unchanged):

- `state.params` also supports `zip_path` (absolute string path), `zip_sha256`
  (64-character hexadecimal source checksum), and `name` (optional show name).
  These fields are optional when reading the original example above. Credentials
  are never persisted. A changed ZIP must start a new job.
- Stage status is one of `pending`, `in_progress`, `done`, or `failed`.
- `state.stages.ingest` also stores `completed`, `pending` (episode ID lists),
  and `episodes` (the validated EpisodeInfo manifest). `episode_count` counts
  successfully committed episodes, including while a job is in progress.
- Each EpisodeInfo has `episode`, `episode_number`, `source_path` (absolute path),
  `original_name` (original ZIP entry name), and `duration` (float seconds, >= 60).
  Example: {"episode":"E01","episode_number":1,"source_path":"/jobs/abc/episodes/E01.mp4",
  "original_name":"show.S01E01.mp4","duration":90.0}.
- Errors without an associated episode omit `episode`. Error history is retained
  after successful resumption. Job and error dates are written in UTC ISO-8601.
- `scores.json` remains an array. Baseline-only records omit `llm_score`,
  `llm_plot_advance`, and `llm_reason` until Phase 6 supplies them; `final_score`
  defaults to 0.0 until selection computes the weighted score.
- Episode pattern priority determines how each filename is parsed. Explicit
  season/episode pairs sort chronologically; seasonless `episode`, `ep`, and `e`
  forms use season 1. Entirely unrecognized naming schemes use full natural
  filename order. In a mixed archive, unrecognized names use season 1 and their
  first numeric token as the episode hint, with natural order as the tie-breaker.
- Phase 1 CLI supports `run --stage ingest`, `resume`, `status`, and `list`;
  `--jobs-dir` is available on each. `run` defaults to ingestion only until later
  phases implement the remaining stages. Rendering is not available yet.

Phase 2 contract additions and clarifications (existing keys are unchanged):

- Each extracted audio file also has `audio/E##.rms.json`, modeled by AudioFeaturesDoc:
  {"episode":"E01","sample_rate":16000,"duration":2.5,"rms_per_second":[0.0,0.5,1.0]}.
  The list contains one finite nonnegative RMS value per one-second sample window.
  The final partial second uses only its actual samples, so list length is
  `ceil(duration)`. Audio is decoded once per processed episode and the same float32
  NumPy waveform is passed to Whisper. The existing transcript JSON shape is unchanged.
- `state.params` also persists `whisper_device` and `whisper_compute` (both default
  to `auto` for older jobs). Resume uses the saved Whisper model, device, and compute
  settings so environment changes do not mix configurations within a season.
- A transcription episode enters `completed` only after its WAV, RMS JSON, and
  transcript JSON are committed. An unfinished stage validates completed artifacts
  and regenerates missing/invalid outputs. A stage already marked `done` is skipped.
- Transcription drops segments with `no_speech_prob > 0.6`, `avg_logprob < -1.0`,
  empty text, or non-finite timestamps/confidence values. Consecutive retained texts
  are compared with collapsed whitespace and casefolding. Timestamps are clamped
  to the audio bounds; zero-length/reversed results are discarded. Empty transcripts
  are valid. Videos with no decodable audio stream fail with a recorded error.
- Phase 2 `run` and `resume` default to `--stage transcribe`; `--stage ingest` still
  stops after ingestion. Scene detection and all later stages remain pending.
- Ordinary pytest runs prohibit real Whisper model construction. The opt-in
  integration smoke test (`pytest tests/test_transcribe_smoke.py --run-model-smoke -s`)
  runs the CLI on the synthetic fixture with real `small` Whisper on CPU/int8,
  downloading/caching it on first use. Its generated job is retained in `data/jobs`.

Phase 3 contract additions and clarifications (existing keys are unchanged):

- `state.params.min_scene_seconds` persists the scene-merging minimum (default
  5.0 for older jobs). Scene detection uses saved threshold/minimum settings on
  resume. `state.stages.scenes` adds `completed` and `pending` episode ID lists.
- `scenes.json` remains the specified array of SceneRecord objects. It is written
  atomically after each episode, before that episode's state checkpoint. Resume
  keeps valid records for checkpointed episodes and rebuilds missing/invalid
  records or records written before their checkpoint. A completed stage is skipped.
- Detection uses ContentDetector with a one-frame internal minimum, followed by
  the configured minimum in seconds. `start_in_scene=True` retains videos without
  cuts. Consecutive short scenes merge forward; a short final remainder merges
  backward. A whole video shorter than the minimum remains one valid scene.
- `align` accepts one episode's chronological SceneRecord skeletons, transcript
  segments, and per-second RMS list. It preserves metadata and returns new records.
  A contained segment belongs to its midpoint's scene; crossing segments belong
  to the greatest overlap. For tied overlaps, text is split proportionally at word
  boundaries, with half-word rounding favoring the earlier scene. The same rule
  applies if a segment crosses more than two scenes.
- `dialogue_ratio` measures the union of actual nonempty speech intervals clipped
  to each scene, independently of which scene receives the text. Overlapping
  transcript segments never double-count speech time. `audio_rms` is the mean of
  per-second RMS bins weighted by time overlap with the scene; unavailable bins
  contribute zero. No-dialogue scenes are retained with zero dialogue ratio.
- Scene IDs are sequential per episode (`E01_s0001`, etc.). `is_first` and
  `is_last` mark only the season's first and last scenes. `characters` stays empty
  until Phase 4. `run` and `resume` now default to `--stage scenes`, retaining
  explicit `--stage ingest` and `--stage transcribe` stops. NLP and later stages
  remain pending.

Phase 4 contract additions and clarifications (existing keys are unchanged):

- `state.params` adds `max_characters` (positive integer, default 12) and
  `sbert_model` (default `all-MiniLM-L6-v2`). New jobs save these settings; older
  jobs use the defaults. Resume uses the saved character cap.
- `state.stages.nlp.characters` records `status`, `completed`, and `pending` for
  per-episode character extraction. After Phase 4, this substage is `done` while
  the overall NLP stage is `in_progress`: baseline and LLM scoring are still
  future phases. Character failures mark both statuses `failed` and record an
  `nlp` error, including the episode when extraction fails.
- `transcripts/E##.characters.json` is a CharacterMentionsDoc checkpoint:
  {"episode":"E01","mentions":[{"name":"Walter White","mention_count":3}]}.
  Counts are positive integers. Every raw PERSON surface form is retained;
  capping and fuzzy merging occur only after aggregating all episodes. Empty
  mention lists are valid. Each file is committed before its episode checkpoint.
  An unfinished stage repairs missing/invalid caches; a completed substage skips.
- `preprocess(text)` strips uppercase speaker prefixes at each line start before
  collapsing whitespace. Sentence splitting uses a cached blank English spaCy
  sentencizer, requiring no trained-model download.
- `extract_characters` consumes an iterable of TranscriptDoc objects, one episode
  at a time. Trained `en_core_web_sm` is loaded lazily and cached; installation is
  an explicit setup step, never an import-time download. PERSON entities only
  contribute to the character table.
- Alias deduplication compares casefolded names using `token_sort_ratio > 88`.
  The most frequent spelling is canonical; ties prefer more words, then longer
  names, then alphabetical order. Each candidate compares against canonical names,
  avoiding transitive alias chains. No first-name/nickname relationship is inferred.
  Groups rank by total mentions, then alphabetical order. The top MAX_CHARACTERS
  groups are retained with `is_main: true`.
- Scene `characters` lists contain canonical names matched case-insensitively
  through whole-word canonical names or observed aliases in cleaned dialogue.
  All other scene fields remain unchanged. These are mentions, not proof that a
  character appears on screen.
- `embed(texts, *, model_name=None)` uses the configured SBERT_MODEL by default,
  with an optional override for saved job settings. The singleton is replaced
  when the requested model changes. Output is float32 with L2-normalized nonblank
  rows. Blank inputs have zero vectors, avoiding false similarities for silence;
  an empty list returns shape (0, 0) without loading a model.
- `run` and `resume` now default to `--stage characters`; earlier explicit stops
  remain supported. Phase 4 writes `characters.json` and enriches `scenes.json`,
  without producing `scores.json` or starting selection/rendering.
- Ordinary pytest blocks trained spaCy and SentenceTransformer construction,
  in addition to Whisper. The opt-in NLP smoke test is
  `pytest tests/test_nlp_smoke.py --run-model-smoke -s`; it exercises real PERSON
  extraction, character artifacts, normalized embeddings, and singleton reuse.

Phase 5 contract additions and clarifications (existing keys are unchanged):

- `state.stages.nlp.baseline` adds `status`, `completed`, and `pending`. Episode
  lists track embedding checkpoints. The substage becomes `done` only after the
  season-wide graph has been ranked and `scores.json` has been committed. Overall
  NLP remains `in_progress` until Phase 6 supplies LLM scoring. Baseline failures
  record an `nlp` error; graph/publication failures omit the episode.
- Each episode has `transcripts/E##.embeddings.json`, a SceneEmbeddingsDoc:
  {"episode":"E01","model_name":"all-MiniLM-L6-v2","input_sha256":"<64 hex characters>",
  "embeddings":[{"scene_id":"E01_s0001","vector":[0.6,0.8]}]}.
  The example vector is illustrative; its actual length comes from the model.
  Each vector is finite and unit length, IDs are unique and belong to the episode,
  and dimensions must agree. Blank cleaned dialogue has no embedding record.
- `input_sha256` hashes the ordered `[scene_id, cleaned_text]` pairs for every
  scene in that episode, encoded as compact UTF-8 JSON with `ensure_ascii=False`.
  An unfinished baseline validates the model name, hash, scene IDs, and vector
  contract before reusing a checkpoint. Missing, changed, or invalid episode
  caches are regenerated. A stage already marked `done` is skipped.
- Baseline uses the job's saved `sbert_model`. It cleans dialogue using Phase 4's
  preprocessing and embeds one episode at a time. The cache file is committed
  before its state checkpoint. A graph/publication failure reuses all completed
  embeddings on resume; graph ranking and score publication are one final unit.
- TextRank uses one undirected graph across the season, including silent/isolated
  scenes. Edges have cosine similarity as their weight and exist only for distinct
  scene pairs with similarity strictly above 0.2. PageRank uses alpha 0.85,
  tolerance 1e-10, and a maximum of 1000 iterations; convergence failure is recorded
  and re-raised. Similarities are computed one row at a time to avoid an extra
  full season-by-season matrix allocation.
- PageRank values are min-max normalized to [0,1]. When the range is at most
  1e-12, all baseline scores are 0.0. This covers a singleton, an all-silent season,
  and other uniform graphs. Silent scenes remain eligible for later non-textual
  scoring and selection. Empty input to `score_scenes` returns an empty mapping;
  duplicate IDs are rejected. A saved job requires scenes for every manifest episode.
- `scores.json` has one ScoreRecord for every scene, preserving `scenes.json` order.
  It includes `scene_id`, `baseline_score`, and `final_score: 0.0`. LLM fields are
  omitted until Phase 6; weighted final scoring remains Phase 7. Scene records are
  unchanged by baseline scoring.
- `run` and `resume` now default to `--stage baseline`. Explicit earlier stops,
  including `--stage characters`, remain supported. Selection and rendering remain
  pending. The opt-in NLP smoke test now verifies real MiniLM-to-TextRank output too.

Phase 6 contract additions and clarifications (existing keys are unchanged):

- Config adds `ENABLE_OPENAI=false` and `SEASON_NUMBER=1`. OpenAI client creation
  requires explicit opt-in and `OPENAI_API_KEY` loaded from `.env`. Opt-in is a
  runtime setting and is never saved as authorization in job state. Model calls
  and provider transports exist only in `sts/nlp/scorer_llm.py`.
- `state.params` adds `llm_provider`, `llm_model`, `ollama_base_url`, and
  `season_number`. New jobs persist nonsecret settings at creation. Older jobs
  with an absent/null `llm_provider` adopt current settings once, before their first
  Phase 6 call. Resume uses saved settings; create a new job to change providers.
  The Ollama endpoint must be HTTP(S), without embedded credentials or parameters.
- `state.stages.nlp.llm` stores `status`, completed/pending episode IDs,
  `active_episode` (omitted when none), `summary_done`, and `completed_scenes` for
  the active episode. Both LLM and overall NLP become `done` only after all scene
  responses are validated and published. Failures mark both failed and record an
  `nlp` error with the active episode when applicable. Error history is retained.
- Each episode is summarized first, combining previous rolling context with that
  episode's chronological scene dialogue. The response is strictly
  {"summary":"cumulative story"}, with 1-500 words and at most 6000 characters.
  This replaces the current context; the updated summary conditions scene scoring
  for that episode and the next episode's summary. Silent dialogue is valid and
  does not establish what happened visually. No transcript is silently truncated.
- Scene responses use {"score":8,"plot_advance":true,"reason":"Introduces the central conflict"}.
  Scores must be integers 1-10, booleans must be JSON booleans, and explanations
  must have 1-15 words. Unknown keys and coercible strings such as "8" are rejected.
  Markdown fences are stripped. Invalid JSON, schema failures, and incomplete
  completions allow three total attempts per unit; transport/auth/refusal errors
  are recorded immediately. Raw invalid responses and provider error bodies are
  not logged or persisted.
- `transcripts/E##.llm.json` is an EpisodeLLMDoc:
  {"episode":"E01","input_sha256":"<64 hex characters>","summary":"Story so far.",
  "scores":[{"scene_id":"E01_s0001","score":8,"plot_advance":true,"reason":"Central conflict"}]}.
  Summary is optional until its call succeeds. Scores form a chronological prefix
  of the episode's unique scene IDs and require a validated summary first.
- Each validated response is saved to its episode checkpoint before publishing
  scores, rolling summaries, and state. Resume revalidates caches even if a crash
  happened before state publication, so a durable response is reused. An API call
  completed before its response was durably saved can still need repeating.
- The checkpoint hash covers prompt version, provider/model, Ollama endpoint when
  used, show/season/runtime, main-character names, incoming summary, and ordered
  scene IDs/timestamps/dialogue. Changed inputs or invalid caches rebuild that
  episode. Changed rolling context invalidates subsequent episode caches. While
  recovering, published LLM scores contain only the revalidated episode prefix;
  stale later scores are cleared while baseline scores remain intact.
- `rolling_summary.json` is a RollingSummaryDoc:
  {"episodes":[{"episode":"E01","summary":"Story through E01."},
  {"episode":"E02","summary":"Story through E02."}]}.
  It retains each validated cumulative summary in chronological order, including
  the active episode once its summary completes.
- `scores.json` gains `llm_score`, `llm_plot_advance`, and `llm_reason` as calls
  complete. Every scene retains its original baseline score; `final_score` remains
  0.0 until Phase 7. Its array order continues to match `scenes.json`.
- `MockProvider` returns fixed canned summary/score responses and supports scripted
  responses/errors for tests. Mock output is test data, not an editorial judgment.
  OpenAI uses the configured Chat Completions model and JSON mode with a 60-second
  timeout and SDK retries disabled. Ollama uses `/api/chat`, `stream:false`, JSON
  mode, and a 120-second timeout. Both bound response generation to 2000 tokens.
- `run` and `resume` now default to `--stage llm`; all earlier stops remain valid.
  Tests use MockProvider, including mocked provider transport contracts, and block
  actual external LLM transport even in opt-in local-model smoke tests. Phase 6
  does not implement selection or rendering.

Phase 7 contract additions and clarifications (existing keys are unchanged):

- `state.params` adds `min_clip_seconds` (positive float) and `score_weights`
  (four finite, nonnegative weights with at least one positive value, ordered
  LLM, TextRank, audio RMS, dialogue ratio). New jobs save these settings. Older
  jobs adopt and persist the current settings at their first selection attempt.
  Resume uses saved settings and the saved SBERT model.
- `state.stages.select` adds `completed` and `pending` episode IDs for embedding
  validation/repair. Selection reuses the Phase 5 embedding JSON contracts,
  checking model, cleaned-text hash, IDs, and vectors before reuse. Missing or
  invalid caches are repaired one episode at a time and committed before state.
  Valid caches, including those written immediately before a crash, require no
  repeated model inference. Silent episodes need no embedding model.
- `combine_scores(scenes, scores, cfg)` returns copies in scene order, preserving
  all baseline/LLM fields. Every component is min-max normalized across the
  season; constant components contribute zero. Weights are normalized to sum to
  one. Missing LLM scores are an error when their weight is positive. Scene IDs
  must be unique and scores must cover exactly the same IDs.
- Selection reserves the complete chronological first and last scenes before
  greedily filling the remaining budget by final score divided by clip duration.
  Density ties prefer chronological order. Numeric episode order and scene start
  determine output order; intervals within an episode cannot overlap. Actual
  `end - start` bounds determine runtime, rather than rounded `duration` fields.
- Forced season bookends take priority over minimum-length and duplicate filters.
  Other clips shorter than MIN_CLIP_SECONDS are excluded. If complete bookends
  cannot fit the target, selection fails with the minimum required runtime;
  it never drops a bookend, trims a scene, or exceeds the budget. A singleton
  counts once. Empty input to the standalone `select` function is valid, while
  saved jobs require scenes matching a nonempty ingestion manifest.
- Deduplication runs after chronological sorting and drops an optional clip only
  when cosine similarity to the previously kept clip is strictly greater than
  0.92. Silent scenes are never dialogue duplicates. A forced closing scene
  removes preceding optional duplicates; identical forced bookends both survive.
  Released time remains unused rather than rerunning the greedy fill.
- `scores.json` receives `final_score` for every scene, preserving input scene
  order. `selected.json` retains its original shape; `original_seconds` sums all
  source scene intervals, and kept/dropped counts include all selection filters.
  Final scoring and publication are one retryable unit. State becomes `done`
  only after both artifacts are atomically written. Failures record a `select`
  error and retain upstream NLP outputs; an already completed stage is skipped.
- `run` and `resume` now default to `--stage select`, preserving all earlier stops,
  including `--stage llm`. Rendering remains pending. The uncut three-episode
  fixture has 90-second bookends, so Phase 7 fixture selection uses a three-minute
  target; a two-minute target correctly reports an infeasible budget.

Phase 8 contract additions and clarifications (existing keys are unchanged):

- `state.stages.render` adds `input_sha256` (optional 64-character hex digest),
  `completed` and `pending` (scene ID lists), `loudnorm` (default false),
  `video_done`, `subtitles_done` (default false), and optional `video_sha256` and
  `subtitles_sha256`. A clip is complete only after its encoded file and validated
  JSON checkpoint are committed. Output flags are checkpointed independently.
- `output/clips/clip_XXXX.json` is a RenderClipDoc paired with `clip_XXXX.mp4`:
  {"scene_id":"E01_s0001","input_sha256":"<64 hex characters>",
  "output_sha256":"<64 hex characters>","duration":15.0}.
  Duration is the probed container length in float seconds. The input digest
  includes clip bounds/identity/path, source byte size and mtime in nanoseconds,
  loudnorm, and renderer version. The output digest hashes all encoded bytes.
  Resume validates these fields and FFprobe stream metadata before reuse. Missing,
  changed, or corrupt caches are rebuilt; valid caches committed before a crash
  are reusable even if state was not yet updated.
- Rendering cuts always re-encode with `-ss` before `-i`, `-t` duration,
  libx264/veryfast/CRF 20 and AAC/192k. Uniformity also requires a fixed 1280x720
  letterbox canvas, square pixels, yuv420p, 30 fps, H.264 High/level 3.1,
  1/90000 video time base, and 48 kHz stereo audio. Display aspect is preserved
  before padding, including sources with nonsquare pixels. Both input streams
  are required. Metadata/chapters are removed and MP4 faststart is enabled.
- `output/clips.txt` is a UTF-8 concat manifest with quoted/escaped absolute paths.
  Concatenation uses `-f concat -safe 0 -c copy` only after validating uniform
  clips. It preserves demuxed timestamps with `-copyts -avoid_negative_ts disabled`
  so AAC encoder priming does not shift the movie/subtitle origin. Encoding and
  concatenation use temporary sibling files and publish
  by replacement only after FFprobe validation. Failed output leaves the previous
  complete destination intact. Each clip must match its requested source interval
  within 0.15 seconds, allowing video-frame/AAC rounding. Movie length must match
  the sum of probed clip lengths within 0.15 seconds.
- `condensed.srt` clips each nonempty transcript segment to its retained source
  interval and offsets it by cumulative probed clip duration, matching concat's
  timeline. Clip array order is authoritative for manually edited selections.
  Text is UTF-8 with whitespace collapsed. Timecodes are created only at the SRT
  output boundary, rounded half-up to milliseconds; zero-length rounded cues are
  dropped. Empty transcripts produce an empty SRT file. Processing loads one
  episode's transcript at a time.
- The stage input digest includes selected.json, clip fingerprints, and selected
  transcript byte hashes. Changes invalidate final output checkpoints. Resume
  verifies output byte hashes and repairs damaged artifacts. `done` requires both
  the movie and SRT committed; failures record a `render` error with the episode
  for clip failures. No upstream model work is performed by rendering.
- Phase 8 adds `sts render --job JOB_ID [--loudnorm | --no-loudnorm]`, also accepting
  `--jobs-dir`. The flag is saved across interrupted renders; omitting it preserves
  the saved setting. This explicit command reopens completed renders and accepts
  edited selected.json while reusing unaffected clips. Edited clips must have
  unique scene IDs, match the ingestion manifest's episode/path, and fit its bounds.
  A normal completed `render_job` call is skipped. `run`/`resume` still default to
  `--stage select`; full controller orchestration is Phase 9.
- Tests include real FFmpeg cuts and a fully decoded concat of short synthetic
  sources with different aspect ratios and audio formats, without model inference.
  The saved three-minute Phase 7 fixture is rendered and decoded separately.

Phase 9 orchestration clarifications (existing JSON keys are unchanged):

- `run_job`, `resume_job`, and their CLI commands now default to `stage="render"`.
  Stage stops are `ingest`, `transcribe`, `scenes`, `characters`, `baseline`, `llm`,
  `select`, and `render`, in that order. The three NLP stops map to their existing
  `state.stages.nlp` substage checkpoints. Completed stages are skipped before
  calling their runner, and a completed normal resume does not rewrite artifacts.
- The controller wraps every stage runner, reloads the latest state after a
  failure, preserves unit checkpoints, marks the active stage failed, records an
  error, and re-raises. NLP failures also mark aggregate NLP failed. Errors already
  recorded by a stage's own handler are not duplicated. Each failed resume attempt
  records a fresh error even if that stage was already failed.
- Successful CLI rendering reports the absolute movie and subtitle paths. The
  separate `sts render` command retains Phase 8's explicit re-render behavior for
  edited selections and optional loudness normalization.
- `scripts/make_fixture.py --scene-cuts` optionally generates three 30-second
  solid-color scenes in each 90-second episode. The default uncut fixture is
  unchanged. The two-minute end-to-end smoke uses the cut fixture in
  `tests/fixtures/end-to-end`, so complete first/last scenes fit without changing
  selector guardrails. The uncut fixture still requires at least three minutes.
- Controller integration tests use deterministic model/media doubles and
  MockProvider while exercising real stages and JSON persistence. The opt-in
  `tests/test_end_to_end_smoke.py --run-model-smoke` runs the default CLI with real
  local Whisper and FFmpeg, checks all contracts and completed stages, decodes the
  whole movie, checks subtitles, and verifies completed resume leaves artifacts
  unchanged. External LLM transports remain blocked. Silent synthetic input may
  correctly produce empty transcripts and an empty SRT.
- Phase 9 ends with controller, CLI, and smoke verification. Streamlit and
  evaluation remain Phases 10 and 11.

6\. MODULE SPECS

6.1 ingest.py

ingest\_zip(zip\_path: Path, job\_dir: Path) -> list\[EpisodeInfo]

Extract; find video files (.mp4 .mkv .avi .mov .webm); rename to E01.ext, E02.ext...

Natural order via regex (priority): (?i)s(\\d{1,2})e(\\d{1,2}) →(?i)(\\d{1,2})x(\\d{1,2}) → (?i)ep?isode\[\\s.\_-]?(\\d{1,3}) →fallback: natural sort of numbers in filename.

Probe each with ffprobe -v error -print\_format json -show\_format -show\_streams.

Reject/corrupt-flag files with no video stream or duration < 60s.

AC: fixture zip of 3 videos sorts E01,E02,E03 even when input names areshow\_2x01.mkv, ep10.mp4, etc.

6.2 transcribe.py

transcribe\_episode(audio\_path: Path, out\_path: Path, cfg) -> TranscriptDoc

Extract audio first: ffmpeg -i ep -vn -ac 1 -ar 16000 audio/E01.wav

Decode wav to numpy (soundfile) once; compute per-second RMS list for laterscene features.

from faster\_whisper import WhisperModelmodel = WhisperModel(cfg.whisper\_model, device=cfg.whisper\_device,                     compute\_type=cfg.whisper\_compute)segments, info = model.transcribe(wav, word\_timestamps=False, vad\_filter=True,                                  condition\_on\_previous\_text=False)segments = list(segments)   # generator — materialize immediately

Hallucination filter: drop segments where no\_speech\_prob > 0.6 oravg\_logprob < -1.0; drop consecutive duplicate texts (Whisper loop artifact).

Save transcript JSON. Update state per episode (resumable mid-season).

6.3 scenes.py

detect\_scenes(video\_path: Path, cfg) -> list\[tuple\[float, float]]

from scenedetect import detect, ContentDetectorscene\_list = detect(video\_path, ContentDetector(threshold=cfg.detector\_threshold))bounds = \[(t\[0].get\_seconds(), t\[1].get\_seconds()) for t in scene\_list]

Enforce min\_scene\_seconds by merging too-short scenes into neighbors.

6.4 align.py — THE TRICKY MODULE, follow exactly

align(scenes, transcript\_segments, rms\_per\_second) -> list\[SceneRecord]

Midpoint rule: transcript segment belongs to the scene containing its midpoint.

Dialogue crossing a cut: assign to scene with the larger overlapping portion;if tie, split text proportionally.

transcript = concatenated text of assigned segments;dialogue\_ratio = seconds with dialogue / scene duration.

audio\_rms = mean RMS over scene span (from 6.2's rms list).

Silent scenes are VALID (action scenes) — keep them, flag with low dialogue\_ratio.

AC: unit test with synthetic scenes/segments covering: fully-inside,straddling-cut, and no-dialogue cases.

6.5 nlp/preprocess.py

Clean text (collapse whitespace, strip speaker-label prefixes like "WALTER:" usingregex ^\\s\*\[A-Z]\[A-Z .'\\-]{1,24}:\\s), segment into sentences, return list\[str].

6.6 nlp/ner.py

extract\_characters(all\_transcripts) -> CharacterTable

spaCy en\_core\_web\_sm PERSON entities → aggregate counts → fuzzy-dedupe aliases(rapidfuzz token\_sort\_ratio > 88) → top MAX\_CHARACTERS → characters.json.

6.7 nlp/embeddings.py

Singleton wrapper around sentence-transformers;embed(texts: list\[str]) -> np.ndarray (L2-normalized).Cache model in module-level global.

6.8 nlp/scorer\_textrank.py (BASELINE — no LLM)

score\_scenes(scenes) -> dict\[scene\_id, float]

Embed scene transcripts → cosine similarity graph (edge if sim > 0.2) →networkx.pagerank → min-max normalize to \[0,1].

6.9 nlp/scorer\_llm.py (PROPOSED METHOD)

Abstract LLMProvider.complete(system, user, json\_mode=True) -> str withOpenAIProvider (default), OllamaProvider, MockProvider (deterministiccanned responses for tests).

score\_scenes(scenes, characters, rolling\_summary, cfg) -> (list\[ScoreRecord], str\_new\_rolling\_summary)

Per-episode flow: (1) episode-summary call → update rolling summary;(2) score that episode's scenes with rolling context.Retry JSON parse up to 3 times; validate with pydantic.

System prompt:

You are a TV editor condensing Season {n} of "{show}" into a {target} movie.Main characters: {characters}Story so far: {rolling\_summary}

User prompt per scene:

Scene {i} of {N} (episode {ep}, {start}-{end}s):"{transcript}"Respond ONLY with JSON:{"score": <1-10>, "plot\_advance": <bool>, "reason": "<max 15 words>"}

6.10 selector.py

select(scenes, scores, target\_seconds, cfg) -> SelectionDoc

final\_score = weighted mix per SCORE\_WEIGHTS (normalize each component to\[0,1] first).

Greedy by score density (score/second) subject to total ≤ target, then sortclips chronologically (episode\_number, start).

Guardrails: force-keep first and last scene of the season; drop selected clips< MIN\_CLIP\_SECONDS; drop a clip if its embedding similarity to the previouslykept clip > 0.92 (dedup).

6.11 render.py

Per clip (uniform params mandatory):

ffmpeg -ss {start} -i {src} -t {dur} -vf scale=-2:720,fps=30 \\  -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 192k -ar 48000 clip\_XXXX.mp4

Concat: ffmpeg -f concat -safe 0 -i clips.txt -c copy output/condensed\_movie.mp4

Export condensed.srt: transcript segments of kept clips, timestamps offset bycumulative output position (pure Python).

Optional loudnorm flag: apply loudnorm=I=-16:TP=-1.5:LRA=11 per clip's audio.

6.12 controller.py

run\_job(zip\_path, target\_minutes, ...), resume\_job(job\_id), get\_status(job\_id)

Stages in order: ingest → transcribe → scenes+align → nlp → select → render.

Wrap each stage in try/except → record error in state.json → re-raise.

Idempotent stages: skip if state says done.

6.13 CLI (python -m sts.app.cli)

sts run --zip PATH --target-minutes 120 \[--name "My Show"]sts resume --job JOB\_IDsts status \[--job JOB\_ID]sts render --job JOB\_ID        # re-render from edited selected.jsonsts list

6.14 Streamlit UI (app/ui.py)

Upload ZIP → create job → auto-refresh loop reading state.json → progress perstage (st.progress) → final: st.video preview + st.download\_button for mp4 andsrt + job list sidebar.

Cache models with st.cache\_resource.

7\. TESTING

scripts/make\_fixture.py: generate tests/fixtures/ —

ffmpeg -f lavfi -i testsrc2=duration=90:size=640x360:rate=30 \\  -f lavfi -i sine=frequency=440:duration=90 -shortest tests/fixtures/ep\_test.mp4

plus a zip containing 3 such files with tricky names.



Unit tests: mock Whisper + LLM (MockProvider). Only integration smoke test mayhit real models:python -m sts.app.cli run --zip tests/fixtures/show.zip --target-minutes 2must produce a playable mp4 + srt + complete state.json.

Every module spec above has a matching test file with ≥3 cases including edgecases.

8\. KNOWN PITFALLS — READ BEFORE CODING

faster-whisper transcribe() returns a GENERATOR — list() it immediately.

scenedetect returns FrameTimecode pairs — use .get\_seconds().

Whisper repeats/hallucinates on music/silence — apply §6.2 filters.

Naive sort breaks episode order (E10 < E2) — use §6.1 regexes.

\-c copy cuts produce frozen frames at joins — always re-encode clips.

Put -ss BEFORE -i (fast seek); use -t duration, not -to.

LLM JSON output must be parsed defensively (strip markdown fences, retry).

spaCy model must be downloaded in setup, not at import time.

Don't hold all episodes in memory — process episode by episode.

Streamlit re-runs the whole script per interaction — cache everything heavy.

9\. CODING CONVENTIONS

ruff (line length 100) + full type hints.

Commit per phase: phase(N): <summary>.

Every stage logs: started / unit-progress / done / failed.

No global mutable state except model singletons.

10\. BUILD PHASES (execute in order, stop after each)

Phase	Deliverable	Definition of Done

0	Scaffold	structure per §3, requirements.txt, .env.example, pyproject, make\_fixture.py works, pytest runs green (empty)

1	config + schemas + ingest + controller skeleton	ingest tests pass; sts run --zip fixture --stage ingest renames E01–E03 correctly

2	transcribe	fixture transcribes (may be near-empty text — valid); resumable mid-season

3	scenes + align	align unit tests cover midpoint/straddle/silent cases; scenes.json generated for fixture

4	preprocess + ner + embeddings	characters.json built; embedding wrapper cached

5	scorer\_textrank	scores.json has baseline\_score for every scene

6	scorer\_llm + rolling summary	MockProvider tests pass; OpenAIProvider behind env flag

7	selector	selected.json ≤ target; first/last scenes present; dedup rule works

8	render + srt	fixture renders a playable condensed mp4 + matching srt

9	end-to-end controller + CLI + smoke test	full sts run on fixture zip: ZIP in → condensed movie out

10	Streamlit UI	upload→progress→download flow works

11	eval scripts	rouge\_eval.py compares condensed SRT vs. reference recap txt; prints ROUGE-1/2/L

After each phase: run ruff check and pytest, print results, and STOP.

