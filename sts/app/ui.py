"""A cinematic Streamlit workspace for resumable season-to-movie editing."""

from __future__ import annotations

import html
import logging
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from time import time
from types import MappingProxyType
from zipfile import BadZipFile

import streamlit as st
from rich.logging import RichHandler

from sts.app.media_server import LocalMediaServer, start_media_server
from sts.app.model_cache import install_model_cache
from sts.app.runner import JobRunner
from sts.config import Settings
from sts.pipeline.controller import get_status
from sts.schemas import JobState, RequiredRange, SelectionDoc, StageState, read_json

_STYLES = """
<style>
:root { --gold:#dfb575; --cream:#f3eee5; --muted:#a6b2c1; --line:#2b3644;
 --ink:#0a1018; --panel:#111b27; --field:#0d1520; }
.stApp { background:var(--ink); color:var(--cream); font-size:16px; }
.stMainBlockContainer { max-width:1360px; padding:2.4rem 3.25rem 3rem; }
[data-testid="stMain"] { min-width:0; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stAppDeployButton"] { display:none; }
[data-testid="stMarkdownContainer"] p { line-height:1.65; }
[data-testid="stCaptionContainer"] { color:var(--muted); font-size:14px; line-height:1.6; }
[data-testid="stSidebar"] { background:#0f1722; border-right:1px solid var(--line); }
[data-testid="stSidebarContent"] { padding:1.4rem 1.1rem; }
[data-testid="stSidebar"] .stButton button { text-align:left; justify-content:flex-start; }
[data-testid="stSidebar"] .stButton button p { font-size:14px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.6rem; }
.brand { display:flex; gap:12px; align-items:center; padding:8px 0 28px; }
.brand-mark { border:1px solid #89734e; color:#e8c58f; width:42px; height:46px;
 display:grid; place-items:center; font-family:Georgia,serif; font-size:27px;
 border-radius:11px; flex-shrink:0; background:#1d2329; }
.brand-name { font-size:16px; letter-spacing:-.3px; font-weight:650; color:var(--cream); }
.brand-sub { font-size:10px; color:#b2bbca; letter-spacing:2px; margin-top:5px; }
.sidebar-label { color:#a1aec0; font-size:11px; letter-spacing:1.5px; margin:24px 0 10px; }
.sidebar-foot { margin-top:28px; border-top:1px solid var(--line); padding-top:20px;
 font-size:13px; color:var(--muted); line-height:1.8; }
.sidebar-foot b { color:var(--cream); font-weight:500; }
.workspace-top { display:flex; justify-content:space-between; align-items:center;
 gap:20px; padding:0 0 20px; margin-bottom:4px; border-bottom:1px solid var(--line); }
.workspace-brand { font-family:Georgia,'Times New Roman',serif; color:var(--cream);
 font-size:23px; letter-spacing:-.6px; line-height:1.25; }
.workspace-tagline { color:var(--muted); font-size:11px; letter-spacing:1.6px;
 text-transform:uppercase; margin-top:6px; }
.st-key-workspace_nav { padding:16px 0 20px; border-bottom:1px solid var(--line);
 margin-bottom:14px; }
.st-key-workspace_nav [data-testid="stHorizontalBlock"] { align-items:flex-end; }
.st-key-workspace_nav [data-testid="stColumn"] { min-width:0; }
.st-key-phone_access { margin:0 0 10px; }
.st-key-phone_access [data-testid="stExpander"] details { background:#0f1924; }
.phone-banner { display:flex; align-items:flex-start; gap:12px; padding:13px 15px;
 border:1px solid #405440; border-radius:11px; background:#19291f55;
 font-size:14px; line-height:1.6; color:#d3e4d5; }
.phone-banner p { margin:0; }
.phone-banner a { color:#e9c58e; overflow-wrap:anywhere; }
.topline { display:flex; justify-content:space-between; gap:16px; align-items:center;
 margin-bottom:26px; font-size:12px; color:var(--muted); letter-spacing:1px; }
.topline span { color:var(--cream); }
.local-pill { border:1px solid #344739; border-radius:30px; padding:8px 12px;
 color:#c0d5c6; font-size:10px; letter-spacing:1.2px; white-space:nowrap; }
.local-pill::before { content:'●'; color:#b3cfb3; margin-right:7px; }
.eyebrow, .step-kicker { color:#e1bc83; font-size:11px; font-weight:600; letter-spacing:1.8px;
 text-transform:uppercase; margin-bottom:12px; line-height:1.5; }
.hero { display:grid; grid-template-columns:1.25fr .9fr; gap:44px; align-items:center;
 padding:12px 0 30px; }
.hero > div { min-width:0; }
.hero h1 { font-family:Georgia,'Times New Roman',serif; font-size:clamp(40px,4.1vw,56px);
 font-weight:400; line-height:1.1; letter-spacing:-1.8px; margin:0 0 19px; padding:0;
 color:var(--cream); }
.hero h1 em { color:#e0bb82; font-weight:400; }
.hero p { color:#b2bdcc; font-size:16px; line-height:1.75; max-width:500px; margin:0; }
.film-art { height:234px; position:relative; border:1px solid #394248; border-radius:18px;
 overflow:hidden; background:radial-gradient(ellipse at 75% 0%,#54433055,transparent 75%),
 linear-gradient(145deg,#17212b,#0d1520); }
.art-label { position:absolute; top:20px; left:24px; color:#c8b18d; font-size:10px;
 letter-spacing:1.7px; }
.film-stack { position:absolute; top:62px; left:25px; right:25px; display:flex; gap:10px;
 transform:rotate(-5deg); }
.film-frame { height:100px; flex:1; min-width:0; border:1px solid #84735e; border-radius:5px;
 background:linear-gradient(135deg,#665844,#243440 58%,#131b23); position:relative;
 box-shadow:0 12px 18px #0005; }
.film-frame:nth-child(2) { background:linear-gradient(145deg,#29313d,#9b794d 65%,#24252b); }
.film-frame:nth-child(3) { background:linear-gradient(135deg,#718078,#344552 55%,#18212a); }
.film-frame::before { content:''; position:absolute; inset:9px 7px;
 border-top:1px solid #ddc69d80; border-bottom:1px solid #ddc69d80; }
.film-frame i { position:absolute; bottom:19px; left:12px; color:#eee2cc;
 font-family:Georgia,serif; font-size:25px; font-style:normal; }
.art-timeline { position:absolute; left:25px; right:25px; bottom:27px;
 display:flex; align-items:center; gap:8px; }
.art-timeline span { height:5px; flex:1; border-radius:3px; background:#b3946570; }
.art-timeline span:nth-child(2) { background:#e4c58e; flex:2; }
.art-timeline b { color:#d8c7a8; font-size:10px; font-weight:500; letter-spacing:1px;
 margin-left:6px; white-space:nowrap; }
.step-strip { display:flex; gap:24px; flex-wrap:wrap; padding:18px 0 25px; }
.step-item { display:flex; gap:10px; align-items:center; color:#c0c9d5; font-size:14px; }
.step-item b { display:grid; place-items:center; flex-shrink:0; width:28px; height:28px;
 border:1px solid #6d5c41; border-radius:50%; color:#e2bf89; font-size:12px; font-weight:500; }
.divider { height:1px; background:var(--line); margin:4px 0 24px; }
.section-head { display:flex; align-items:center; justify-content:space-between; gap:16px;
 margin:0 0 18px; }
.section-head h2 { color:var(--cream); font-size:22px; font-weight:550; letter-spacing:-.6px;
 margin:0; padding:0; line-height:1.3; }
.section-head small { color:var(--muted); font-size:11px; letter-spacing:.5px; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:var(--line)!important;
 background:var(--panel); border-radius:16px!important; }
.st-key-upload_panel, .st-key-edit_panel, .st-key-stage_panel, .st-key-export_panel {
 padding:10px 12px; }
.panel-title { font-size:20px; font-weight:550; color:var(--cream); letter-spacing:-.4px;
 margin:2px 0 8px; line-height:1.4; }
.panel-note { color:#b2bdcc; font-size:15px; line-height:1.7; margin-bottom:22px; }
[data-testid="stFileUploaderDropzone"] { background:var(--field); border:1px dashed #786749;
 border-radius:12px; min-height:160px; padding:24px 18px; }
[data-testid="stFileUploaderDropzone"] svg { color:#e1bc83; }
[data-testid="stFileUploaderDropzone"] small { color:#adbac9; font-size:13px; }
[data-testid="stWidgetLabel"] p { font-size:14px; color:#dee3e9; font-weight:500; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea { background:var(--field); font-size:16px;
 min-height:48px; line-height:1.5; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { min-height:48px; font-size:16px; }
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
 min-height:48px; align-items:center; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { font-size:15px; }
[data-testid="stExpander"] details { border-color:var(--line); border-radius:12px; }
[data-testid="stExpander"] summary { min-height:48px; }
[data-testid="stExpander"] summary p { font-size:15px; }
.stButton button, .stDownloadButton button, .stLinkButton a {
 min-height:48px; border-radius:10px; font-size:15px; font-weight:550; padding:10px 16px;
 transition:border-color .15s,background .15s; overflow-wrap:anywhere; }
.stButton button p, .stDownloadButton button p, .stLinkButton a p {
 font-size:15px; line-height:1.35; }
.stButton button:focus-visible, .stDownloadButton button:focus-visible,
.stLinkButton a:focus-visible, [data-testid="stFileUploaderDropzone"] button:focus-visible {
 outline:2px solid #f0ce98; outline-offset:3px; }
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"],
.stLinkButton a[kind="primary"] { background:var(--gold); color:#181b20;
 border:1px solid var(--gold); }
.stButton button[kind="primary"]:hover:not(:disabled),
.stDownloadButton button[kind="primary"]:hover:not(:disabled),
.stLinkButton a[kind="primary"]:hover { background:#ebc793; border-color:#ebc793; color:#141922; }
.stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"],
.stLinkButton a[kind="secondary"] { background:#172332; border-color:#3c4b5e; color:#e5eaf0; }
.stButton button[kind="primary"]:disabled { background:#313740; border-color:#444d59;
 color:#aab1bd; cursor:not-allowed; }
.quiet-note { color:var(--muted); font-size:13px; line-height:1.7; }
.promise-row { display:flex; gap:24px; flex-wrap:wrap; padding:20px 0 0;
 color:#b4c0cf; font-size:13px; }
.promise-row span::before { content:'✓'; color:#e3bd83; margin-right:8px; }
.workspace-title { display:flex; justify-content:space-between; align-items:flex-start;
 gap:20px; margin:8px 0 26px; }
.workspace-title > div { min-width:0; }
.workspace-title h1 { font-family:Georgia,'Times New Roman',serif; font-weight:400;
 font-size:46px; line-height:1.15; letter-spacing:-1.3px; margin:0 0 12px; padding:0;
 color:var(--cream); overflow-wrap:anywhere; }
.workspace-title p { color:var(--muted); font-size:13px; line-height:1.6; margin:0; }
.status-pill { font-size:12px; border-radius:30px; padding:8px 13px; letter-spacing:.3px;
 border:1px solid #485568; color:#cbd3df; white-space:nowrap; }
.status-pill.ready { border-color:#4a6856; color:#c0dfc6; background:#243a2c60; }
.status-pill.processing { border-color:#80663f; color:#e9c58e; background:#4b3d2740; }
.status-pill.failed { border-color:#956257; color:#f0c0b5; background:#693b3340; }
.stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px;
 margin-bottom:24px; }
.stat { border:1px solid var(--line); padding:20px 22px; border-radius:13px;
 background:linear-gradient(135deg,#15202d,#111a26); min-width:0; }
.stat label { display:block; font-size:10px; color:#adbbcc; letter-spacing:1.1px;
 text-transform:uppercase; margin-bottom:10px; line-height:1.4; }
.stat strong { font-size:27px; font-weight:500; color:var(--cream); line-height:1.25; }
.stage-row { display:flex; gap:10px; align-items:center; margin:16px 0 8px; }
.stage-number { border:1px solid #415066; width:34px; height:34px; border-radius:9px;
 display:grid; place-items:center; flex-shrink:0; color:#b0bed0; font-size:12px; }
.stage-number.done { border-color:#62705d; background:#35413265; color:#d5dcc1; }
.stage-number.in_progress { border-color:#a28455; color:#edcd9b; background:#60492b45; }
.stage-number.failed { border-color:#965c52; color:#f2beaf; }
.stage-name { flex:1; min-width:0; font-size:14px; color:#e5e9ee; line-height:1.4; }
.stage-detail { font-size:12px; color:#b1bed0; line-height:1.5; text-align:right; }
[data-testid="stProgress"] { margin-bottom:10px; }
[data-testid="stProgress"] > div > div { background:#2b3544; }
[data-testid="stProgress"] > div > div > div > div { background:#d9b375; }
.empty-preview { display:flex; min-height:280px; flex-direction:column; justify-content:center;
 align-items:center; text-align:center; padding:32px; border:1px solid #354152;
 border-radius:13px; background:radial-gradient(ellipse at center,#3942543d,transparent),#0e1722; }
.preview-icon { font-family:Georgia,serif; font-size:35px; color:#dfb575; margin-bottom:17px; }
.empty-preview h3 { font-family:Georgia,serif; font-size:30px; font-weight:400;
 color:var(--cream); line-height:1.3; margin:0 0 12px; }
.empty-preview p { color:#afbdce; font-size:15px; line-height:1.75; max-width:360px; }
.export-caption { font-size:13px; color:#b7c2d0; line-height:1.6; margin:14px 0 18px; }
.st-key-export_panel video { width:100%; max-width:100%; border-radius:12px; display:block;
 background:#05090e; }
.footer { display:flex; justify-content:space-between; gap:16px; border-top:1px solid var(--line);
 color:#a1afc2; font-size:12px; line-height:1.6; padding-top:20px; margin-top:32px; }
@media(max-width:1100px) {
 .stMainBlockContainer { padding:2.1rem 1.75rem 2.5rem; }
 .hero { gap:28px; } .hero h1 { font-size:44px; } .film-art { height:218px; }
 .stat { padding:18px 16px; } .stat strong { font-size:25px; }
}
@media(max-width:850px) {
 .st-key-create_grid [data-testid="stHorizontalBlock"],
 .st-key-project_grid [data-testid="stHorizontalBlock"],
 .st-key-export_actions [data-testid="stHorizontalBlock"] {
  flex-wrap:wrap!important; gap:1rem!important; }
 .st-key-create_grid [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
 .st-key-project_grid [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
 .st-key-export_actions [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
  flex:1 1 100%!important; width:100%!important; min-width:0!important; }
 .hero { grid-template-columns:1.2fr .85fr; gap:22px; }
 .hero h1 { font-size:40px; letter-spacing:-1.2px; } .hero p { font-size:15px; }
 .film-art { height:190px; } .film-stack { top:55px; left:18px; right:18px; gap:7px; }
 .film-frame { height:78px; } .art-label { left:18px; font-size:9px; }
 .art-timeline { left:18px; right:18px; bottom:23px; }
 .stats { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media(max-width:600px) {
 .stMainBlockContainer { padding:2.1rem 16px 2rem; }
 .workspace-top { gap:12px; padding-bottom:17px; }
 .workspace-brand { font-size:21px; }
 .workspace-tagline { font-size:10px; letter-spacing:1.15px; }
 .workspace-top .local-pill { font-size:9px; padding:7px 9px; letter-spacing:.4px; }
 .st-key-workspace_nav { padding:12px 0 16px; margin-bottom:6px; }
 .st-key-workspace_nav [data-testid="stHorizontalBlock"] { gap:10px; }
 .st-key-workspace_nav [data-testid="stColumn"]:first-child {
  flex:1 1 0!important; min-width:0!important; }
 .st-key-workspace_nav [data-testid="stColumn"]:last-child {
  flex:0 0 110px!important; width:110px!important; min-width:110px!important; }
 .st-key-workspace_nav .stButton button { padding:10px 11px; }
 .topline { font-size:10px; margin-bottom:20px; gap:10px; letter-spacing:.6px; }
 .hero { grid-template-columns:minmax(0,1fr); gap:22px; padding:10px 0 20px; }
 .hero h1 { font-size:39px; line-height:1.12; letter-spacing:-1.2px; margin-bottom:16px; }
 .hero p { font-size:16px; line-height:1.65; max-width:100%; }
 .film-art { display:none; }
 .art-label { top:15px; left:20px; font-size:9px; }
 .film-stack { top:41px; left:23px; right:23px; gap:9px; transform:rotate(-4deg); }
 .film-frame { height:65px; } .film-frame i { bottom:14px; font-size:20px; }
 .art-timeline { left:21px; right:21px; bottom:18px; }
 .step-strip { gap:13px 20px; padding:8px 0 20px; }
 .step-item { font-size:13px; gap:8px; } .step-item b { width:25px; height:25px; font-size:11px; }
 .eyebrow, .step-kicker { font-size:10px; letter-spacing:1.35px; margin-bottom:10px; }
 .section-head { gap:10px; margin-bottom:16px; }
 .section-head h2 { font-size:21px; } .section-head small { font-size:10px; }
 .st-key-upload_panel, .st-key-edit_panel, .st-key-stage_panel, .st-key-export_panel {
  padding:3px 1px; }
 .panel-title { font-size:19px; } .panel-note { font-size:15px; margin-bottom:18px; }
 [data-testid="stFileUploaderDropzone"] { padding:18px 12px; min-height:142px;
  flex-wrap:wrap; gap:12px; }
 [data-testid="stFileUploaderDropzone"] > div { min-width:0; }
 [data-testid="stFileUploaderDropzoneInstructions"] { min-width:0; }
 [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
  overflow-wrap:anywhere; }
 [data-testid="stFileUploaderDropzone"] button { min-height:48px; padding:9px 11px; }
 .workspace-title { gap:12px; flex-wrap:wrap; margin:4px 0 20px; }
 .workspace-title h1 { font-size:35px; letter-spacing:-.8px; margin-bottom:9px; }
 .workspace-title p { font-size:12px; }
 .status-pill { font-size:11px; padding:7px 11px; }
 .stats { gap:10px; margin-bottom:20px; }
 .stat { padding:15px 14px; border-radius:11px; }
 .stat label { font-size:9px; letter-spacing:.8px; margin-bottom:9px; }
 .stat strong { font-size:24px; }
 .stage-row { gap:9px; flex-wrap:wrap; }
 .stage-name { font-size:14px; }
 .stage-detail { font-size:12px; max-width:110px; overflow-wrap:anywhere; }
 .promise-row { gap:12px 18px; padding-top:17px; font-size:12px; }
 .empty-preview { min-height:236px; padding:24px 16px; }
 .empty-preview h3 { font-size:26px; } .empty-preview p { font-size:14px; }
 .export-caption { font-size:12px; margin:12px 0 16px; }
 .footer { flex-direction:column; gap:8px; margin-top:26px; padding-top:18px; font-size:11px; }
}
@media(max-width:360px) {
 .workspace-brand { font-size:19px; } .workspace-tagline { letter-spacing:.65px; }
 .workspace-top .local-pill { padding:6px 7px; font-size:8px; }
 .hero h1 { font-size:34px; } .hero p { font-size:15px; }
 .section-head { align-items:flex-start; } .section-head small { max-width:76px; text-align:right; }
 .stage-detail { flex-basis:100%; max-width:none; padding-left:43px; text-align:left; }
 .workspace-title h1 { font-size:32px; }
}
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


@st.cache_resource(show_spinner=False)
def _get_media_server(jobs_dir: str) -> LocalMediaServer:
    """Share a disk-streaming server without caching movie contents in Streamlit."""
    return start_media_server(Path(jobs_dir))


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
        story = stage.llm
        fraction = _fraction(story)
        total = len(story.completed) + len(story.pending)
        if story.active_episode and story.scene_count and total and story.status != "done":
            active = (
                0.1 * story.summary_done + 0.9 * len(story.completed_scenes) / story.scene_count
            )
            fraction = (len(story.completed) + min(0.99, active)) / total
        return (_fraction(stage.characters) + _fraction(stage.baseline) + fraction) / 3
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
                if key == "llm" and stage.llm.active_episode:
                    story = stage.llm
                    if not story.summary_done:
                        return f"{story.active_episode} · Reading the story"
                    return (
                        f"{story.active_episode} · "
                        f"{len(story.completed_scenes)} of {story.scene_count} blocks"
                    )
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
    source = next((item for item in state.stages.ingest.episodes if item.episode == episode), None)
    audio_path = directory / "audio" / f"{episode}.wav"
    try:
        metadata = audio_path.stat()
    except OSError:
        return f"Preparing audio for {episode}. Counters advance after each episode finishes."
    if metadata.st_size <= 0:
        return f"Preparing audio for {episode}. Counters advance after each episode finishes."
    minutes = f" · {source.duration / 60:.1f} min of source audio" if source else ""
    age_minutes = max(0, int((time() - metadata.st_mtime) / 60))
    device = (
        "automatic device selection"
        if state.params.whisper_device == "auto"
        else (state.params.whisper_device.upper())
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
        '<div class="step-strip" aria-label="Three steps to your film">'
        '<div class="step-item"><b>01</b><span>Bring your episodes</span></div>'
        '<div class="step-item"><b>02</b><span>Shape your edit</span></div>'
        '<div class="step-item"><b>03</b><span>Watch & download</span></div></div>'
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
    with st.container(key="create_grid"):
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
            scene_method = st.selectbox(
                "Editing approach",
                ["Fast · subtitle story blocks", "Visual scene boundaries"],
                index=0 if settings.scene_method == "subtitles" else 1,
                key="scene_method",
                help="Timed story blocks keep dialogue and nearby action together. "
                "Visual boundaries scan the episode for camera cuts and take longer.",
            )
            remove_credits = st.checkbox(
                "Remove end credits", value=settings.exclude_credits, key="exclude_credits"
            )
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
                    "scene_method": "subtitles" if scene_method.startswith("Fast") else "visual",
                    "exclude_credits": remove_credits,
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


def _watch_worker(runner: JobRunner) -> None:
    job_id = runner.active_job_id
    if job_id is None:
        st.rerun()
    st.caption(
        "Your current edit is processing. Stop it or wait for it to finish to start another."
    )
    if st.button("View current edit", key="view_current_edit", type="tertiary"):
        st.session_state.active_job = job_id
        st.rerun()
    _stop_processing(runner, job_id, key="stop_previous_edit")


def _stop_processing(runner: JobRunner, job_id: str, *, key: str) -> None:
    stopping = runner.is_stopping(job_id)
    if stopping:
        st.info("Stopping processing… Completed work is saved. Waiting for the active operation.")
    if st.button(
        "Stopping…" if stopping else "Stop processing",
        icon=":material/stop_circle:",
        key=key,
        disabled=stopping,
    ):
        try:
            runner.request_stop(job_id)
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(f"Couldn't stop processing: {exc}")
    if not stopping:
        st.caption("Completed work is saved. You can resume this project later.")


def _exports(state: JobState, directory: Path, selection: SelectionDoc | None) -> None:
    movie, subtitles = directory / "output/condensed_movie.mp4", directory / "output/condensed.srt"
    ready = state.stages.render.status == "done"
    _html('<div class="section-head"><h2>Your film</h2><small>PREVIEW & EXPORT</small></div>')
    if ready and movie.is_file() and subtitles.is_file():
        try:
            media = _get_media_server(str(directory.parent))
            preview_url = media.url(state.job_id, movie.name)
            movie_url = media.url(state.job_id, movie.name, download=True)
            subtitle_url = media.url(state.job_id, subtitles.name, download=True)
        except (OSError, ValueError) as exc:
            st.error(f"Couldn't open the film player: {exc}")
            return
        st.video(preview_url)
        duration = (
            selection.selected_total_seconds / 60 if selection else state.params.target_seconds / 60
        )
        _html(
            f'<div class="export-caption">{duration:.1f} min · 720p · MP4 &nbsp; / &nbsp; '
            "Ready for your next screening.</div>"
        )
        with st.container(key="export_actions"):
            movie_column, subtitle_column = st.columns([1.1, 1])
        with movie_column:
            st.link_button(
                "Download film",
                url=movie_url,
                key=f"movie_{state.job_id}",
                type="primary",
                icon=":material/download:",
                width="stretch",
                on_click="ignore",
            )
        with subtitle_column:
            st.link_button(
                "Subtitles · SRT",
                url=subtitle_url,
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


def _moment_time(seconds: float) -> str:
    """Format saved source seconds as a clock timestamp, rounded to tenths."""
    tenths = int((Decimal(str(seconds)) * 10).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    hours, remainder = divmod(tenths, 36000)
    minutes, remainder = divmod(remainder, 600)
    whole_seconds, fraction = divmod(remainder, 10)
    ending = f"{whole_seconds:02d}" + (f".{fraction}" if fraction else "")
    return f"{hours}:{minutes:02d}:{ending}" if hours else f"{minutes}:{ending}"


def _project_view(runner: JobRunner, job_id: str) -> None:
    try:
        state = runner.get_state(job_id)
    except (OSError, ValueError) as exc:
        st.error(f"This project couldn't be opened: {exc}")
        return
    running = runner.is_running(job_id)
    label, style = _job_status(state, running)
    if running and runner.is_stopping(job_id):
        label = "Stopping…"
    directory = runner.jobs_dir / job_id
    selection = _selection(directory)
    name = html.escape(state.params.name or "Untitled season")
    _html(
        f'<div class="workspace-title"><div><div class="eyebrow">Your editing workspace</div>'
        f"<h1>{name}</h1><p>Your season edit · "
        f"{state.created_at.strftime('%d %b %Y')}</p></div>"
        f'<span class="status-pill {style}">{label}</span></div>'
    )
    episodes = state.stages.ingest.episode_count
    kept = str(selection.stats.scenes_kept) if selection else "—"
    film_runtime = f"{selection.selected_total_seconds / 60:.1f} min" if selection else "—"
    original_seconds = (
        selection.stats.original_seconds
        if selection
        else sum(item.duration for item in state.stages.ingest.episodes)
    )
    original = (
        f"{original_seconds / 3600:.1f} h"
        if original_seconds >= 3600
        else f"{original_seconds / 60:g} min"
        if original_seconds
        else "—"
    )
    _html(
        '<div class="stats">'
        f'<div class="stat"><label>Episodes</label><strong>{episodes:02d}</strong></div>'
        '<div class="stat"><label>Target runtime</label>'
        f"<strong>{state.params.target_seconds / 60:g} min</strong></div>"
        f'<div class="stat"><label>Film runtime</label><strong>{film_runtime}</strong></div>'
        f'<div class="stat"><label>Original runtime</label><strong>{original}</strong></div></div>'
    )
    if state.params.must_keep_ranges:
        with st.expander(f"Protected moments · {len(state.params.must_keep_ranges)} saved"):
            for required in state.params.must_keep_ranges:
                st.caption(
                    f"Protected moment · {required.episode} · "
                    f"{_moment_time(required.start)}–{_moment_time(required.end)} · Saved request"
                )
    if selection:
        st.caption(f"{kept} story clips selected in episode order.")
        if selection.selected_total_seconds < state.params.target_seconds * 0.98:
            st.warning(
                "This edit is shorter than your target. Refine it to keep connected story "
                "sequences and use more of the available runtime."
            )
    error = runner.error(job_id)
    if running:
        _stop_processing(runner, job_id, key=f"stop_{job_id}")
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
    with st.container(key="project_grid"):
        export_column, progress_column = st.columns([1.5, 0.85], gap="large")
    with progress_column, st.container(border=True, key="stage_panel"):
        _html('<div class="section-head"><h2>The journey</h2><small>6 STAGES</small></div>')
        overall = sum(_stage_progress(state, name) for name, _ in _STAGES) / len(_STAGES)
        if state.stages.render.status != "done":
            overall = min(overall, 0.99)
        st.progress(overall, text=f"{int(overall * 100)}% complete")
        with st.expander("Editing progress", expanded=state.stages.render.status != "done"):
            for index, (key, title) in enumerate(_STAGES, 1):
                stage = getattr(state.stages, key)
                symbol = "✓" if stage.status == "done" else f"{index:02d}"
                _html(
                    f'<div class="stage-row">'
                    f'<span class="stage-number {stage.status}">{symbol}</span>'
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
    if not running and state.stages.transcribe.status == "done":
        _refine_edit(runner, state)
    if state.params.llm_provider == "mock":
        st.caption("Preview edit · This project uses sample story scores.")
    if state.errors:
        with st.expander(f"Processing history · {len(state.errors)} recorded issue(s)"):
            for entry in state.errors:
                st.text(f"{entry.stage} · {entry.episode or 'project'} · {entry.message}")
    if state.stages.render.status == "done" and st.session_state.get("was_processing", False):
        st.session_state.was_processing = False
        st.rerun()


def _refine_edit(runner: JobRunner, state: JobState) -> None:
    with st.expander("Refine the story"):
        st.caption(
            "Create a new edit with connected story sequences and confirmed end credits "
            "removed. Your earlier movie is kept. You can protect a moment the edit missed."
        )
        protect = st.checkbox("Keep a specific moment", key=f"protect_{state.job_id}")
        required = list(state.params.must_keep_ranges)
        if protect:
            episode = st.selectbox(
                "Episode",
                [item.episode for item in state.stages.ingest.episodes],
                key=f"required_episode_{state.job_id}",
            )
            start = st.number_input(
                "Moment starts · minutes",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f"required_start_{state.job_id}",
            )
            end = st.number_input(
                "Moment ends · minutes",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f"required_end_{state.job_id}",
            )
            st.caption(
                "The new edit keeps the whole surrounding story blocks, including nearby action."
            )
        if st.button(
            "Create refined film",
            type="primary",
            disabled=runner.busy,
            key=f"refine_{state.job_id}",
        ):
            try:
                if protect:
                    required.append(RequiredRange(episode=episode, start=start * 60, end=end * 60))
                refined = runner.refine(state.job_id, must_keep_ranges=required)
                runner.start(refined.job_id)
                st.session_state.active_job = refined.job_id
                st.rerun()
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Couldn't refine this edit: {exc}")


def _pick_project() -> None:
    st.session_state.active_job = st.session_state.project_picker


def _workspace_navigation(runner: JobRunner, states: list[JobState]) -> None:
    names = {
        state.job_id: (
            f"{state.params.name or 'Untitled season'} · "
            f"{_job_status(state, runner.is_running(state.job_id))[0]}"
        )
        for state in states
    }
    options: list[str | None] = [None, *names]
    current = st.session_state.active_job
    if current not in options:
        current = None
        st.session_state.active_job = None
    if st.session_state.get("project_picker") != current:
        st.session_state.project_picker = current
    with st.container(key="workspace_nav"):
        picker, action = st.columns([3, 1], gap="small")
        with picker:
            st.selectbox(
                "Your projects",
                options,
                format_func=lambda value: names.get(value, "Create a new film"),
                key="project_picker",
                on_change=_pick_project,
            )
        with action:
            if st.button(
                "New film", icon=":material/add:", key="new_film", width="stretch"
            ):
                st.session_state.active_job = None
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
    states = _saved_jobs(runner.jobs_dir)
    _sidebar(runner, states)
    _html(
        '<div class="workspace-top"><div><div class="workspace-brand">Season to Screen</div>'
        '<div class="workspace-tagline">The editing room</div></div>'
        '<div class="local-pill">LOCAL WORKSPACE</div></div>'
    )
    _workspace_navigation(runner, states)
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
