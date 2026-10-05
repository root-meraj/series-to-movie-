"""A cinematic Streamlit workspace for resumable season-to-movie editing."""

from __future__ import annotations

import html
import logging
from functools import partial
from pathlib import Path
from time import time
from types import MappingProxyType
from zipfile import BadZipFile

import streamlit as st
from rich.logging import RichHandler

from sts.app.model_cache import install_model_cache
from sts.app.runner import JobRunner
from sts.config import Settings
from sts.pipeline.controller import get_status
from sts.schemas import JobState, SelectionDoc, StageState, read_json

_STYLES = """
<style>
:root { --gold:#d8af72; --muted:#95a0b2; --line:#283140; }
.stApp { background:#0b0f16; color:#f2f0eb; }
.stMainBlockContainer { max-width:1440px; padding:2.3rem 3.4rem 3rem; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stAppDeployButton"] { display:none; }
[data-testid="stSidebar"] { background:#101620; border-right:1px solid #26303e; }
[data-testid="stSidebarContent"] { padding:1.4rem 1.15rem; }
[data-testid="stSidebar"] .stButton button { text-align:left; justify-content:flex-start; }
[data-testid="stSidebar"] .stButton button p { font-size:.84rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.55rem; }
.brand { display:flex; gap:12px; align-items:center; padding:8px 0 36px; }
.brand-mark { border:1px solid #8e744d; color:#e3c28e; width:40px; height:44px;
 display:grid; place-items:center; font-family:Georgia,serif; font-size:25px;
 border-radius:9px; background:linear-gradient(135deg,#2d291f,#121923); }
.brand-name { font-size:15px; letter-spacing:-.3px; font-weight:650; }
.brand-sub { font-size:9px; color:#a8b0bf; letter-spacing:2.5px; margin-top:4px; }
.eyebrow { color:#c5a878; font-size:10px; font-weight:600; letter-spacing:2px;
 text-transform:uppercase; margin-bottom:12px; }
.sidebar-label { color:#8190a5; font-size:10px; letter-spacing:1.8px; margin:28px 0 10px; }
.sidebar-foot { margin-top:32px; border-top:1px solid #283140; padding-top:20px;
 font-size:12px; color:#99a4b5; line-height:1.8; }
.sidebar-foot b { color:#d3d9e2; font-weight:500; }
.topline { display:flex; justify-content:space-between; gap:15px; align-items:center;
 margin-bottom:34px; font-size:11px; color:#a3adbd; letter-spacing:1.2px; }
.topline span { color:#e3e6eb; }
.local-pill { border:1px solid #2c3c37; border-radius:30px; padding:7px 12px;
 color:#acc5b7; font-size:9px; letter-spacing:1.4px; white-space:nowrap; }
.local-pill::before { content:'●'; color:#9fbba4; margin-right:7px; }
.hero { display:grid; grid-template-columns:1.15fr 1fr; gap:40px; align-items:center;
 padding:0 0 32px; }
.hero h1 { font-family:Georgia,'Times New Roman',serif; font-size:clamp(38px,4vw,60px);
 font-weight:400; line-height:1.12; letter-spacing:-1.9px; margin:0 0 19px; padding:0; }
.hero h1 em { color:#ddba87; font-weight:400; }
.hero p { color:#a7b0be; font-size:14px; line-height:1.9; max-width:455px; margin:0; }
.film-art { height:220px; position:relative; border:1px solid #2d3543; border-radius:14px;
 overflow:hidden; background:radial-gradient(ellipse at 70% 0%,#5b47324d,transparent 65%),#111822; }
.art-label { position:absolute; top:17px; left:21px; color:#aa987e; font-size:8px;
 letter-spacing:2.2px; }
.film-stack { position:absolute; top:52px; left:26px; right:26px; display:flex; gap:9px;
 transform:rotate(-5deg); }
.film-frame { height:93px; flex:1; border:1px solid #806f5d; border-radius:4px;
 background:linear-gradient(135deg,#544b3d,#182633 55%,#0a121b); position:relative;
 box-shadow:0 16px 22px #0007; }
.film-frame:nth-child(2) { background:linear-gradient(145deg,#232a39,#8b6843 65%,#171a22); }
.film-frame:nth-child(3) { background:linear-gradient(135deg,#63716a,#253442 55%,#131820); }
.film-frame::before { content:''; position:absolute; inset:9px 6px;
 border-top:1px solid #bfa78370; border-bottom:1px solid #bfa78370; }
.film-frame i { position:absolute; bottom:19px; left:12px; color:#d8cbb9;
 font-family:Georgia,serif; font-size:22px; font-style:normal; }
.art-timeline { position:absolute; left:27px; right:27px; bottom:27px;
 display:flex; align-items:center; gap:8px; }
.art-timeline span { height:4px; flex:1; border-radius:3px; background:#b3946570; }
.art-timeline span:nth-child(2) { background:#dec08c; flex:2; }
.art-timeline b { color:#d1c2ab; font-size:9px; font-weight:400; letter-spacing:1px;
 margin-left:6px; white-space:nowrap; }
.divider { height:1px; background:linear-gradient(90deg,#434033,#26313e); margin:4px 0 26px; }
.section-head { display:flex; align-items:center; justify-content:space-between; gap:16px;
 margin:0 0 18px; }
.section-head h2 { color:#eeeae3; font-size:20px; font-weight:500; letter-spacing:-.4px;
 margin:0; padding:0; }
.section-head small { color:#8b98ab; font-size:11px; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#293343!important;
 background:#121923; border-radius:14px!important; }
.st-key-upload_panel, .st-key-edit_panel, .st-key-stage_panel, .st-key-export_panel {
 padding:8px 10px; }
.panel-title { font-size:15px; font-weight:550; color:#e9e7e1; margin:2px 0 7px; }
.panel-note { color:#99a5b6; font-size:12px; line-height:1.7; margin-bottom:19px; }
[data-testid="stFileUploaderDropzone"] { background:#0e141e; border:1px dashed #5b5549;
 border-radius:10px; min-height:146px; padding:24px 16px; }
[data-testid="stFileUploaderDropzone"] svg { color:#c5a878; }
[data-testid="stFileUploaderDropzone"] small { color:#9eabbb; }
[data-testid="stWidgetLabel"] p { font-size:12px; color:#d4d8df; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { background:#0d141e; }
.stButton button, .stDownloadButton button { min-height:43px; border-radius:8px;
 font-size:12px; font-weight:500; transition:border-color .15s,background .15s; }
.stButton button:focus-visible, .stDownloadButton button:focus-visible {
 outline:2px solid #e3c28e; outline-offset:3px; }
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
 background:#d8af72; color:#15181b; border:1px solid #d8af72; }
.stButton button[kind="primary"]:hover:not(:disabled),
.stDownloadButton button[kind="primary"]:hover:not(:disabled) {
 background:#e8c58f; border-color:#e8c58f; color:#10151c; }
.stButton button[kind="primary"]:disabled {
 background:#33343a; border-color:#42444b; color:#9b9ca2; cursor:not-allowed; }
.quiet-note { color:#91a0b4; font-size:11px; line-height:1.8; }
.promise-row { display:flex; gap:26px; flex-wrap:wrap; padding:19px 0 0;
 color:#aeb7c3; font-size:11px; }
.promise-row span::before { content:'✓'; color:#d8af72; margin-right:7px; }
.workspace-title { display:flex; justify-content:space-between; align-items:flex-start;
 gap:20px; margin:2px 0 24px; }
.workspace-title h1 { font-family:Georgia,serif; font-weight:400; font-size:39px;
 line-height:1.2; letter-spacing:-1px; margin:0 0 10px; padding:0; overflow-wrap:anywhere; }
.workspace-title p { color:#93a1b5; font-size:12px; margin:0; }
.status-pill { font-size:10px; border-radius:30px; padding:7px 12px; letter-spacing:.5px;
 border:1px solid #414354; color:#b9c1cf; white-space:nowrap; }
.status-pill.ready { border-color:#45634f; color:#b3ceba; background:#253e2a40; }
.status-pill.processing { border-color:#6c5a3e; color:#ddba87; background:#4b3d2740; }
.status-pill.failed { border-color:#83564f; color:#e1b0a5; background:#693b3340; }
.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
.stat { border:1px solid #293344; padding:17px 20px; border-radius:11px; background:#111923; }
.stat label { display:block; font-size:9px; color:#97a5b7; letter-spacing:1.2px;
 text-transform:uppercase; margin-bottom:8px; }
.stat strong { font-size:23px; font-weight:450; color:#eeeae2; }
.stage-row { display:flex; gap:12px; align-items:center; margin:12px 0 5px; }
.stage-number { border:1px solid #364154; width:30px; height:30px; border-radius:8px;
 display:grid; place-items:center; color:#8b9aad; font-size:11px; }
.stage-number.done { border-color:#5c6450; background:#39402e60; color:#c8c6a2; }
.stage-number.in_progress { border-color:#a28455; color:#dfbf8e; background:#60492b45; }
.stage-number.failed { border-color:#965c52; color:#ecb1a0; }
.stage-name { flex:1; font-size:12px; color:#e2e3e5; }
.stage-detail { font-size:10px; color:#99a5b7; }
[data-testid="stProgress"] { margin-bottom:11px; }
[data-testid="stProgress"] > div > div { background:#26303d; }
[data-testid="stProgress"] > div > div > div > div { background:#c3a06b; }
.empty-preview { display:flex; min-height:280px; flex-direction:column; justify-content:center;
 align-items:center; text-align:center; padding:30px; border:1px solid #293344;
 border-radius:12px; background:radial-gradient(ellipse at center,#2c35463d,transparent),#0c131d; }
.preview-icon { font-family:Georgia,serif; font-size:33px; color:#c5a878; margin-bottom:15px; }
.empty-preview h3 { font-family:Georgia,serif; font-size:25px; font-weight:400; margin:0 0 9px; }
.empty-preview p { color:#8f9eb2; font-size:12px; line-height:1.8; max-width:320px; }
.export-caption { font-size:11px; color:#a0acbd; margin:13px 0 18px; }
.footer { display:flex; justify-content:space-between; gap:16px; border-top:1px solid #273141;
 color:#7f8da1; font-size:10px; padding-top:18px; margin-top:31px; }
@media(max-width:1000px) { .stMainBlockContainer { padding:2rem 1.5rem; }
 .hero { gap:20px; } .hero h1 { font-size:42px; } .film-art { height:200px; } }
@media(max-width:700px) { .hero { grid-template-columns:1fr; } .film-art { display:none; }
 .hero h1 { font-size:34px; } .stats { grid-template-columns:repeat(2,1fr); }
 .stat { padding:13px 15px; } .workspace-title h1 { font-size:29px; }
 .topline { font-size:9px; margin-bottom:24px; } .stMainBlockContainer { padding:1.5rem 1rem; }
 .stage-detail { max-width:130px; text-align:right; } .promise-row { gap:13px; } }
@media(prefers-reduced-motion:reduce) { * { transition:none!important; animation:none!important; } }
</style>
"""
_ENGINES = MappingProxyType(
    {
        "Preview · sample scoring": "mock",
        "OpenAI · story-aware": "openai",
        "Ollama · local model": "ollama",
    }
)
_DIALOGUE_SOURCES = MappingProxyType(
    {
        "Automatic · use subtitles first": "auto",
        "Embedded subtitles only": "subtitles",
        "Whisper · transcribe audio": "whisper",
    }
)
_STAGES = (
    ("ingest", "Import episodes"),
    ("transcribe", "Capture dialogue"),
    ("scenes", "Find the scenes"),
    ("nlp", "Understand the story"),
    ("select", "Shape your edit"),
    ("render", "Render the film"),
)


def _jobs_dir() -> Path:
    return Path("data/jobs").resolve()


@st.cache_resource(show_spinner=False)
def _get_runner(jobs_dir: str) -> JobRunner:
    install_model_cache()
    return JobRunner(Path(jobs_dir))


def _html(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


def _saved_jobs(jobs_dir: Path) -> list[JobState]:
    states: list[JobState] = []
    for path in jobs_dir.glob("*/state.json"):
        try:
            states.append(get_status(path.parent.name, jobs_dir=jobs_dir))
        except (OSError, ValueError):
            logging.getLogger(__name__).warning("Unable to read saved project %s", path.parent.name)
    return sorted(states, key=lambda state: state.created_at, reverse=True)


def _job_status(state: JobState, running: bool) -> tuple[str, str]:
    if state.stages.render.status == "done":
        return "Ready", "ready"
    if running:
        return "Processing", "processing"
    if any(getattr(state.stages, key).status == "failed" for key, _ in _STAGES):
        return "Needs attention", "failed"
    return "Paused", "paused"


def _fraction(stage: StageState) -> float:
    if stage.status == "done":
        return 1.0
    if stage.status == "pending":
        return 0.0
    completed = getattr(stage, "completed", [])
    pending = getattr(stage, "pending", [])
    total = len(completed) + len(pending)
    return len(completed) / total if total else 0.0


def _stage_progress(state: JobState, name: str) -> float:
    stage = getattr(state.stages, name)
    if name == "transcribe" and stage.status != "done":
        current = getattr(stage, "current", None)
        total = len(stage.completed) + len(stage.pending)
        if current and current.episode in stage.pending and current.duration > 0 and total:
            fraction = min(0.99, current.processed_seconds / current.duration)
            return (len(stage.completed) + fraction) / total
    if name == "nlp" and stage.status != "done":
        return sum(_fraction(getattr(stage, key)) for key in ("characters", "baseline", "llm")) / 3
    if name == "render" and stage.status != "done":
        return (_fraction(stage) * 0.8) + 0.1 * stage.video_done + 0.1 * stage.subtitles_done
    return _fraction(stage)


def _stage_detail(state: JobState, name: str) -> str:
    stage = getattr(state.stages, name)
    if stage.status == "done":
        return "Complete"
    if stage.status == "failed":
        return "Saved · ready to retry"
    if name == "nlp":
        names = {
            "characters": "Characters",
            "baseline": "Scene connections",
            "llm": "Story scoring",
        }
        for key, label in names.items():
            if getattr(stage, key).status != "done":
                return label if stage.status == "in_progress" else "Waiting"
    completed, pending = getattr(stage, "completed", []), getattr(stage, "pending", [])
    if completed or pending:
        unit = "clips" if name == "render" else "episodes"
        return f"{len(completed)} of {len(completed) + len(pending)} {unit}"
    return "Working" if stage.status == "in_progress" else "Waiting"


def _transcription_activity(state: JobState, directory: Path, *, running: bool) -> str | None:
    stage = state.stages.transcribe
    if not running or stage.status != "in_progress" or not stage.pending:
        return None
    episode = stage.pending[0]
    current = getattr(stage, "current", None)
    if current is not None:
        episode = current.episode
        if current.activity == "extracting_subtitles":
            return f"Extracting timed subtitles from {episode}."
        if current.activity == "extracting_audio":
            return f"Preparing audio for {episode}."
        if current.activity == "loading_model":
            return f"Loading Whisper {state.params.whisper_model} for {episode}."
    source = next(
        (item for item in state.stages.ingest.episodes if item.episode == episode), None
    )
    audio_path = directory / "audio" / f"{episode}.wav"
    try:
        metadata = audio_path.stat()
    except OSError:
        return f"Preparing audio for {episode}. Counters advance after each episode finishes."
    if metadata.st_size <= 0:
        return f"Preparing audio for {episode}. Counters advance after each episode finishes."
    minutes = f" · {source.duration / 60:.1f} min of source audio" if source else ""
    age_minutes = max(0, int((time() - metadata.st_mtime) / 60))
    device = "automatic device selection" if state.params.whisper_device == "auto" else (
        state.params.whisper_device.upper()
    )
    decoded = (
        f"Decoded {current.processed_seconds / 60:.1f} of {current.duration / 60:.1f} min. "
        if current is not None
        else ""
    )
    return (
        f"Transcribing {episode}{minutes} · Whisper {state.params.whisper_model} on {device}. "
        f"Audio prepared {age_minutes} min ago. "
        f"{decoded}"
        "Episode counters update after the whole episode finishes. "
        + ("Long episodes can take over an hour on CPU." if device == "CPU" else "")
    )


def _sidebar(runner: JobRunner, states: list[JobState]) -> None:
    with st.sidebar:
        _html(
            '<div class="brand"><div class="brand-mark">S</div><div>'
            '<div class="brand-name">Season to Screen</div>'
            '<div class="brand-sub">THE EDITING ROOM</div></div></div>'
        )
        if st.button("New project", icon=":material/add:", key="new_project", width="stretch"):
            st.session_state.active_job = None
            st.rerun()
        _html(f'<div class="sidebar-label">YOUR PROJECTS &nbsp; / &nbsp; {len(states):02d}</div>')
        if not states:
            st.caption("Your edits will appear here. Start with a season ZIP.")
        for state in states:
            label, _ = _job_status(state, runner.is_running(state.job_id))
            name = state.params.name or f"Untitled · {state.job_id[:6]}"
            icon = ":material/check_circle:" if label == "Ready" else ":material/movie:"
            if st.button(
                name,
                key=f"job_{state.job_id}",
                help=f"{label} · {state.params.target_seconds / 60:g} minute target",
                icon=icon,
                type="primary" if st.session_state.active_job == state.job_id else "tertiary",
                width="stretch",
            ):
                st.session_state.active_job = state.job_id
                st.rerun()
        if st.button(
            "Refresh projects", icon=":material/refresh:", type="tertiary", width="stretch"
        ):
            st.rerun()
        _html(
            '<div class="sidebar-foot"><b>A season, distilled.</b><br>'
            "Keep the story. Make every scene count.<br><br>"
            "Your files stay in this workspace.<br>Story-aware cloud editing sends dialogue "
            "to your configured provider.</div>"
        )


def _hero() -> None:
    _html(
        '<section class="hero"><div><div class="eyebrow">Every scene has a purpose</div>'
        "<h1>The whole story.<br><em>One unforgettable cut.</em></h1>"
        "<p>Turn a season into a single, compelling film. Bring your episodes. "
        "Set your runtime. Keep the moments that move the story.</p></div>"
        '<div class="film-art" role="img" aria-label="Three episode frames condensed into a film">'
        '<div class="art-label">FROM EPISODES TO ESSENTIALS</div><div class="film-stack">'
        '<div class="film-frame"><i>01</i></div><div class="film-frame"><i>02</i></div>'
        '<div class="film-frame"><i>03</i></div></div><div class="art-timeline">'
        "<span></span><span></span><span></span><b>ONE STORY</b></div></div></section>"
        '<div class="divider"></div>'
    )


def _local_zip_source(value: str) -> tuple[Path | None, str | None]:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].strip()
    if not value:
        return None, "Paste the full path to your ZIP file to enable Create my film."
    try:
        path = Path(value).expanduser()
        if path.suffix.lower() != ".zip":
            return None, "Choose a ZIP archive; the file path must end in .zip."
        if not path.is_file():
            return None, "ZIP file not found. Check the full path and try again."
        return path, None
    except (OSError, ValueError, RuntimeError):
        return None, "That file path cannot be opened. Check the full path and try again."


def _new_project(runner: JobRunner, settings: Settings) -> None:
    _hero()
    _html(
        '<div class="section-head"><h2>Create your next cut</h2>'
        "<small>01 / NEW PROJECT</small></div>"
    )
    left, right = st.columns([1.15, 1], gap="large")
    with left, st.container(border=True, key="upload_panel"):
        _html(
            '<div class="panel-title">Bring the season</div>'
            '<div class="panel-note">Choose a ZIP of your episodes. We’ll put them in order.</div>'
        )
        source = st.radio(
            "Episode source",
            ["Upload ZIP", "Use local ZIP"],
            horizontal=True,
            key="archive_source",
        )
        upload = None
        local_path = None
        if source == "Use local ZIP":
            st.caption("Recommended for archives over 2 GB. Reads directly from this computer.")
            local_value = st.text_input(
                "ZIP file path",
                placeholder=r"C:\Videos\season.zip",
                key="local_zip_path",
                help="Paste the full path. On Windows, right-click the ZIP → Copy as path.",
            )
            local_path, source_issue = _local_zip_source(local_value)
            st.caption(
                "Keep the original ZIP in place and unchanged until Import episodes completes."
            )
        else:
            upload = st.file_uploader(
                "Season archive",
                type=["zip"],
                key="upload_zip",
                label_visibility="collapsed",
                max_upload_size=2048,
                help="MP4, MKV, AVI, MOV, and WebM episodes in one ZIP. Up to 2 GB.",
            )
            st.caption("Upload up to 2 GB. For a larger archive, choose Use local ZIP above.")
            source_issue = (
                "Choose a ZIP up to 2 GB, or switch to Use local ZIP for a larger archive."
                if upload is None
                else None
            )
        name = st.text_input(
            "Project name",
            placeholder="e.g. The Last Horizon · Season 1",
            key="show_name",
            max_chars=100,
        )
    with right, st.container(border=True, key="edit_panel"):
        _html(
            '<div class="panel-title">Make it your cut</div>'
            '<div class="panel-note">Choose the length and the way your story is scored.</div>'
        )
        target = st.number_input(
            "Target runtime · minutes",
            min_value=0.5,
            max_value=1440.0,
            value=min(1440.0, max(0.5, settings.target_minutes)),
            step=5.0,
            key="target_minutes",
            help="A maximum runtime. Complete opening and closing scenes are preserved.",
        )
        provider = settings.llm_provider
        if provider == "openai" and not settings.enable_openai:
            provider = "mock"
        options = list(_ENGINES)
        chosen = st.selectbox(
            "Editorial engine",
            options,
            index=list(_ENGINES.values()).index(provider),
            key="editorial_engine",
            help="Choose story-aware cloud editing, a local model, or sample scoring.",
        )
        engine = _ENGINES[chosen]
        model = settings.llm_model
        if engine == "mock":
            st.caption(
                "Preview mode uses sample scores. Choose a configured model for story-aware edits."
            )
        with st.expander("Advanced settings"):
            dialogue_source = st.selectbox(
                "Dialogue source",
                list(_DIALOGUE_SOURCES),
                index=list(_DIALOGUE_SOURCES.values()).index(settings.transcript_source),
                key="transcript_source",
                help=(
                    "Automatic uses timestamped embedded text subtitles first and falls back "
                    "to Whisper audio transcription. Embedded subtitles only requires a usable "
                    "text subtitle track. Whisper transcribes the episode audio."
                ),
            )
            device = st.selectbox(
                "Processing device",
                ["CPU", "Automatic", "GPU · CUDA"],
                index=2 if settings.whisper_device == "cuda" else 0,
                key="processing_device",
                help="CPU works without GPU libraries. CUDA requires a configured NVIDIA runtime.",
            )
            st.caption(
                "CUDA is configured for this workspace."
                if settings.whisper_device == "cuda"
                else "CPU is the default for reliable local processing."
            )
            if engine != "mock":
                model = st.text_input("Model", value=settings.llm_model, key="editorial_model")
                st.caption(
                    "Ollama runs on this computer and needs no API key. Use an installed model."
                    if engine == "ollama"
                    else "Provider access and credentials come from your local .env configuration."
                )
        blocked_reason = (
            "An edit is processing. You can create another when it finishes."
            if runner.busy
            else source_issue
        )
        create = st.button(
            "Create my film",
            icon=":material/arrow_forward:",
            key="create_project",
            type="primary",
            width="stretch",
            disabled=blocked_reason is not None,
            help=blocked_reason,
        )
        if blocked_reason:
            st.caption(blocked_reason)
    _html(
        '<div class="promise-row"><span>Opening & closing scenes preserved</span>'
        "<span>Progress saved automatically</span><span>Film + subtitles included</span></div>"
    )
    if create:
        try:
            if engine == "openai" and (
                not settings.enable_openai or not settings.openai_api_key.get_secret_value()
            ):
                st.error(
                    "Enable OpenAI and add its API key in .env "
                    "before using story-aware cloud editing."
                )
                return
            if not model.strip():
                st.error("Enter a model name before creating the film.")
                return
            compute_device = {"CPU": "cpu", "Automatic": "auto", "GPU · CUDA": "cuda"}[device]
            cfg = settings.model_copy(
                update={
                    "llm_provider": engine,
                    "llm_model": model.strip(),
                    "whisper_device": compute_device,
                    "whisper_compute": "int8" if compute_device == "cpu" else "auto",
                    "transcript_source": _DIALOGUE_SOURCES[dialogue_source],
                }
            )
            if local_path is not None:
                state = runner.create_from_path(
                    local_path, float(target), name=name.strip() or None, cfg=cfg
                )
            elif upload is not None:
                state = runner.create(upload, float(target), name=name.strip() or None, cfg=cfg)
            else:
                return
            st.session_state.active_job = state.job_id
            runner.start(state.job_id)
            st.rerun()
        except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
            st.error(f"Couldn't start your edit: {exc}")


def _selection(directory: Path) -> SelectionDoc | None:
    try:
        return read_json(directory / "selected.json", SelectionDoc)
    except (OSError, ValueError):
        return None


def _read_download(path: Path) -> bytes:
    return path.read_bytes()


def _watch_worker(runner: JobRunner) -> None:
    job_id = runner.active_job_id
    if job_id is None:
        st.rerun()
    st.caption("Your current edit is processing. This form will be ready when it finishes.")
    if st.button("View current edit", key="view_current_edit", type="tertiary"):
        st.session_state.active_job = job_id
        st.rerun()


def _exports(state: JobState, directory: Path, selection: SelectionDoc | None) -> None:
    movie, subtitles = directory / "output/condensed_movie.mp4", directory / "output/condensed.srt"
    ready = state.stages.render.status == "done"
    _html('<div class="section-head"><h2>Your film</h2><small>PREVIEW & EXPORT</small></div>')
    if ready and movie.is_file() and subtitles.is_file():
        st.video(str(movie))
        duration = (
            selection.selected_total_seconds / 60 if selection else state.params.target_seconds / 60
        )
        _html(
            f'<div class="export-caption">{duration:g} min · 720p · MP4 &nbsp; / &nbsp; '
            "Ready for your next screening.</div>"
        )
        movie_column, subtitle_column = st.columns([1.1, 1])
        with movie_column:
            st.download_button(
                "Download film",
                data=partial(_read_download, movie),
                file_name="condensed_movie.mp4",
                mime="video/mp4",
                key=f"movie_{state.job_id}",
                type="primary",
                icon=":material/download:",
                width="stretch",
                on_click="ignore",
            )
        with subtitle_column:
            st.download_button(
                "Subtitles · SRT",
                data=partial(_read_download, subtitles),
                file_name="condensed.srt",
                mime="application/x-subrip",
                key=f"subtitles_{state.job_id}",
                icon=":material/subtitles:",
                width="stretch",
                on_click="ignore",
            )
        if subtitles.stat().st_size == 0:
            st.caption(
                "No dialogue was detected in the retained scenes. Your subtitle file is empty."
            )
    elif ready:
        st.warning(
            "An output file is missing. Re-render this project with the CLI to restore its exports."
        )
    else:
        _html(
            '<div class="empty-preview"><div class="preview-icon">▷</div>'
            "<h3>A story worth waiting for.</h3><p>Your finished film will appear here, "
            "with downloads for the movie and its subtitles.</p></div>"
        )
        st.caption("You can leave this page while editing continues. Your progress is saved.")


def _project_view(runner: JobRunner, job_id: str) -> None:
    try:
        state = runner.get_state(job_id)
    except (OSError, ValueError) as exc:
        st.error(f"This project couldn't be opened: {exc}")
        return
    running = runner.is_running(job_id)
    label, style = _job_status(state, running)
    directory = runner.jobs_dir / job_id
    selection = _selection(directory)
    name = html.escape(state.params.name or "Untitled season")
    _html(
        f'<div class="workspace-title"><div><div class="eyebrow">Your editing workspace</div>'
        f"<h1>{name}</h1><p>Project {html.escape(job_id)} · "
        f"{state.created_at.strftime('%d %b %Y')}</p></div>"
        f'<span class="status-pill {style}">{label}</span></div>'
    )
    episodes = state.stages.ingest.episode_count
    kept = str(selection.stats.scenes_kept) if selection else "—"
    original_seconds = (
        selection.stats.original_seconds
        if selection
        else sum(item.duration for item in state.stages.ingest.episodes)
    )
    original = (
        f"{original_seconds / 3600:.1f} h"
        if original_seconds >= 3600
        else f"{original_seconds / 60:g} min" if original_seconds else "—"
    )
    _html(
        '<div class="stats">'
        f'<div class="stat"><label>Episodes</label><strong>{episodes:02d}</strong></div>'
        '<div class="stat"><label>Target runtime</label>'
        f"<strong>{state.params.target_seconds / 60:g} min</strong></div>"
        f'<div class="stat"><label>Scenes kept</label><strong>{kept}</strong></div>'
        f'<div class="stat"><label>Original runtime</label><strong>{original}</strong></div></div>'
    )
    error = runner.error(job_id)
    if not running and state.stages.render.status != "done":
        if label == "Needs attention" or error:
            st.error("This edit needs attention. Your completed work is saved.")
            st.text(
                error
                or (state.errors[-1].message if state.errors else "The last stage did not finish.")
            )
        else:
            st.info("This edit is paused. Resume from the latest saved checkpoint.")
        if st.button(
            "Resume edit",
            icon=":material/play_arrow:",
            key=f"resume_{job_id}",
            type="primary",
            disabled=runner.busy,
        ):
            try:
                runner.start(job_id)
                st.rerun()
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Couldn't resume your edit: {exc}")
    progress_column, export_column = st.columns([0.85, 1.5], gap="large")
    with progress_column, st.container(border=True, key="stage_panel"):
        _html('<div class="section-head"><h2>The journey</h2><small>6 STAGES</small></div>')
        overall = sum(_stage_progress(state, name) for name, _ in _STAGES) / len(_STAGES)
        if state.stages.render.status != "done":
            overall = min(overall, 0.99)
        st.progress(overall, text=f"{int(overall * 100)}% complete")
        for index, (key, title) in enumerate(_STAGES, 1):
            stage = getattr(state.stages, key)
            symbol = "✓" if stage.status == "done" else f"{index:02d}"
            _html(
                f'<div class="stage-row"><span class="stage-number {stage.status}">{symbol}</span>'
                f'<span class="stage-name">{title}</span>'
                f'<span class="stage-detail">{_stage_detail(state, key)}</span></div>'
            )
            st.progress(_stage_progress(state, key))
            if key == "transcribe":
                activity = _transcription_activity(state, directory, running=running)
                if activity:
                    st.info(activity, icon=":material/graphic_eq:")
        st.caption(
            "Updated automatically while processing."
            if running
            else "Every completed stage is saved."
        )
    with export_column, st.container(border=True, key="export_panel"):
        _exports(state, directory, selection)
    if state.params.llm_provider == "mock":
        st.caption("Preview edit · This project uses sample story scores.")
    if state.errors:
        with st.expander(f"Processing history · {len(state.errors)} recorded issue(s)"):
            for entry in state.errors:
                st.text(f"{entry.stage} · {entry.episode or 'project'} · {entry.message}")
    if state.stages.render.status == "done" and st.session_state.get("was_processing", False):
        st.session_state.was_processing = False
        st.rerun()


def main() -> None:
    """Render the local editing workspace and poll jobs without rerunning models."""
    st.set_page_config(
        page_title="Season to Screen · The Editing Room",
        page_icon="🎞️",
        layout="wide",
        initial_sidebar_state="auto",
    )
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)]
    )
    _html(_STYLES)
    try:
        settings = Settings()
        runner = _get_runner(str(_jobs_dir()))
    except (OSError, ValueError) as exc:
        st.error(f"The workspace configuration needs attention: {exc}")
        return
    st.session_state.setdefault("active_job", None)
    _sidebar(runner, _saved_jobs(runner.jobs_dir))
    _html(
        '<div class="topline"><div>WORKSPACE &nbsp; / &nbsp; <span>THE EDITING ROOM</span></div>'
        '<div class="local-pill">LOCAL WORKSPACE</div></div>'
    )
    job_id = st.session_state.active_job
    if job_id is None:
        _new_project(runner, settings)
        if runner.busy:
            st.fragment(run_every="2s")(_watch_worker)(runner)
    else:
        try:
            unfinished = runner.get_state(job_id).stages.render.status != "done"
        except (OSError, ValueError):
            unfinished = False
        st.session_state.was_processing = unfinished
        view = st.fragment(run_every="2s" if unfinished else None)(_project_view)
        view(runner, job_id)
    _html(
        '<div class="footer"><span>SEASON TO SCREEN</span>'
        "<span>Made for the story. Built for the screen.</span></div>"
    )


if __name__ == "__main__":
    main()
