"""Exercise real Streamlit widgets with durable jobs and deterministic background work."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO
from zipfile import ZipFile, ZipInfo

import pytest
from streamlit.testing.v1 import AppTest

from sts.app import ui
from sts.app.runner import JobRunner
from sts.config import Settings
from sts.schemas import (
    EpisodeInfo,
    ErrorRecord,
    JobState,
    SelectedClip,
    SelectionDoc,
    SelectionStats,
    TranscriptionProgress,
    read_json,
    write_json,
)


class ControlledRunner(JobRunner):
    """Use real upload persistence while manually controlling the background worker."""

    def __init__(self, jobs_dir: Path) -> None:
        """Record create/start requests and leave all model work disabled."""
        super().__init__(jobs_dir)
        self.created: list[JobState] = []
        self.local_paths: list[Path] = []
        self.upload_count = 0
        self.started: list[str] = []
        self.running_id: str | None = None
        self.worker_errors: dict[str, str] = {}

    @property
    def busy(self) -> bool:
        """Expose whether the manually controlled worker is occupied."""
        return self.running_id is not None

    @property
    def active_job_id(self) -> str | None:
        """Expose the manually controlled worker's current job."""
        return self.running_id

    def create(
        self,
        upload: BinaryIO,
        target_minutes: float,
        *,
        name: str | None = None,
        cfg: Settings | None = None,
    ) -> JobState:
        """Publish an uploaded ZIP through the real runner without starting models."""
        self.upload_count += 1
        state = super().create(upload, target_minutes, name=name, cfg=cfg)
        self.created.append(state)
        return state

    def create_from_path(
        self,
        zip_path: Path,
        target_minutes: float,
        *,
        name: str | None = None,
        cfg: Settings | None = None,
    ) -> JobState:
        """Retain a local ZIP reference through the real runner without starting models."""
        state = super().create_from_path(zip_path, target_minutes, name=name, cfg=cfg)
        self.created.append(state)
        self.local_paths.append(zip_path.resolve())
        return state

    def start(self, job_id: str) -> bool:
        """Record one requested launch and defer execution until the test writes state."""
        if self.running_id == job_id:
            return False
        if self.busy:
            raise RuntimeError("Another job is already running")
        self.started.append(job_id)
        self.running_id = job_id
        self.worker_errors.pop(job_id, None)
        return True

    def is_running(self, job_id: str) -> bool:
        """Return whether this job occupies the deterministic background worker."""
        return self.running_id == job_id

    def error(self, job_id: str) -> str | None:
        """Return an explicitly injected background failure."""
        return self.worker_errors.get(job_id)


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ControlledRunner]:
    """Replace only the background runner, retaining real UI widgets and JSON contracts."""
    controlled = ControlledRunner(tmp_path / "jobs")
    monkeypatch.setattr(ui, "_jobs_dir", lambda: controlled.jobs_dir)
    monkeypatch.setattr(ui, "_get_runner", lambda jobs_dir: controlled)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("TRANSCRIPT_SOURCE", "auto")
    yield controlled
    controlled.shutdown()


def _zip_bytes() -> bytes:
    archive = BytesIO()
    with ZipFile(archive, "w") as contents:
        contents.writestr(ZipInfo("show.S01E01.mp4"), b"synthetic video stand-in")
    return archive.getvalue()


def _app() -> AppTest:
    return AppTest.from_string("from sts.app.ui import main\nmain()", default_timeout=10).run()


def _create(
    app: AppTest,
    *,
    name: str = "Test season",
    minutes: float = 2.0,
    device: str | None = None,
    dialogue_source: str | None = None,
) -> AppTest:
    app.file_uploader(key="upload_zip").set_value(("show.zip", _zip_bytes(), "application/zip"))
    app.text_input(key="show_name").set_value(name)
    app.number_input(key="target_minutes").set_value(minutes)
    app.selectbox(key="editorial_engine").select("Preview · sample scoring")
    if device is not None:
        app.selectbox(key="processing_device").select(device)
    if dialogue_source is not None:
        app.selectbox(key="transcript_source").select(dialogue_source)
    app.run()
    return app.button(key="create_project").click().run()


def _save_finished(runner: ControlledRunner, state: JobState) -> Path:
    directory = runner.jobs_dir / state.job_id
    for name in ("ingest", "transcribe", "scenes", "nlp", "select", "render"):
        getattr(state.stages, name).status = "done"
    for name in ("characters", "baseline", "llm"):
        getattr(state.stages.nlp, name).status = "done"
    state.stages.render.video_done = True
    state.stages.render.subtitles_done = True
    write_json(directory / "state.json", state)
    write_json(
        directory / "selected.json",
        SelectionDoc(
            target_seconds=120,
            selected_total_seconds=30,
            clips=[
                SelectedClip(
                    scene_id="E01_s0001",
                    episode="E01",
                    source_path=directory / "episodes/E01.mp4",
                    start=0,
                    end=30,
                )
            ],
            stats=SelectionStats(scenes_kept=1, scenes_dropped=2, original_seconds=90),
        ),
    )
    (directory / "output/condensed_movie.mp4").write_bytes(b"deterministic movie payload")
    (directory / "output/condensed.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello.\n\n", encoding="utf-8"
    )
    runner.running_id = None
    return directory


def test_empty_app_requires_an_upload_and_shows_creation_widgets(runner: ControlledRunner) -> None:
    """A new session exposes the ZIP workflow without starting any model or job."""
    app = _app()
    assert not app.exception
    assert app.radio(key="archive_source").value == "Upload ZIP"
    assert app.file_uploader(key="upload_zip").allowed_type == [".zip"]
    assert app.file_uploader(key="upload_zip").proto.max_upload_size_mb == 2048
    assert app.button(key="create_project").disabled
    assert app.number_input(key="target_minutes").value > 0
    assert app.selectbox(key="processing_device").value == "CPU"
    assert app.selectbox(key="transcript_source").label == "Dialogue source"
    assert app.selectbox(key="transcript_source").value == "Automatic · use subtitles first"
    assert "falls back" in app.selectbox(key="transcript_source").proto.help
    assert runner.created == runner.started == []
    explanations = " ".join(item.value for item in app.caption)
    assert "2 GB" in explanations
    assert "local" in explanations.casefold()


def test_local_zip_path_with_copy_as_path_quotes_creates_one_job_without_upload_or_copy(
    runner: ControlledRunner, tmp_path: Path
) -> None:
    """A quoted local path bypasses browser transfer and retains the original archive reference."""
    source = tmp_path / "My season archive.zip"
    source.write_bytes(_zip_bytes())
    before = source.stat().st_mtime_ns
    app = _app()
    app.radio(key="archive_source").set_value("Use local ZIP").run()
    assert not app.get("file_uploader")
    app.text_input(key="local_zip_path").set_value(f'  "{source}"  ')
    app.text_input(key="show_name").set_value("Local season")
    app.number_input(key="target_minutes").set_value(3.0)
    app.selectbox(key="editorial_engine").select("Preview · sample scoring")
    app.run()
    assert not app.button(key="create_project").disabled
    app.button(key="create_project").click().run()
    assert not app.exception
    assert runner.upload_count == 0
    assert runner.local_paths == [source.resolve()]
    assert len(runner.created) == 1
    state = runner.created[0]
    assert state.params.zip_path == source.resolve()
    assert state.params.target_seconds == 180.0
    assert state.params.name == "Local season"
    assert state.params.llm_provider == "mock"
    assert state.params.whisper_device == "cpu"
    assert state.params.whisper_compute == "int8"
    assert state.params.transcript_source == "auto"
    assert state.params.subtitle_language == "en"
    directory = runner.jobs_dir / state.job_id
    saved = read_json(directory / "state.json", JobState)
    assert saved.params.zip_path == source.resolve()
    assert saved.params.transcript_source == "auto"
    assert not list(directory.rglob("*.zip"))
    assert source.read_bytes() == _zip_bytes()
    assert source.stat().st_mtime_ns == before
    assert app.session_state["active_job"] == state.job_id
    assert runner.started == [state.job_id]
    app.run()
    assert runner.started == [state.job_id]


@pytest.mark.parametrize(
    "kind, expected_reason",
    [
        ("empty", "Paste the full path"),
        ("missing", "ZIP file not found"),
        ("wrong_extension", "file path must end in .zip"),
    ],
)
def test_local_zip_requires_an_existing_zip_path_before_enabling_create(
    runner: ControlledRunner, tmp_path: Path, kind: str, expected_reason: str
) -> None:
    """Missing or unsupported source paths keep creation disabled with a visible explanation."""
    app = _app()
    app.radio(key="archive_source").set_value("Use local ZIP").run()
    source = tmp_path / ("missing.zip" if kind == "missing" else "season.mp4")
    if kind == "wrong_extension":
        source.write_bytes(_zip_bytes())
    if kind != "empty":
        app.text_input(key="local_zip_path").set_value(str(source)).run()
    assert not app.exception
    assert app.button(key="create_project").disabled
    assert runner.created == runner.started == []
    reason = " ".join(
        item.value for elements in (app.caption, app.warning, app.error) for item in elements
    )
    assert expected_reason in reason


def test_invalid_local_zip_displays_error_without_a_saved_job(
    runner: ControlledRunner, tmp_path: Path
) -> None:
    """An existing .zip with corrupt contents fails before publishing any project."""
    source = tmp_path / "broken.zip"
    source.write_bytes(b"not an archive")
    app = _app()
    app.radio(key="archive_source").set_value("Use local ZIP").run()
    app.text_input(key="local_zip_path").set_value(str(source)).run()
    assert not app.button(key="create_project").disabled
    app.button(key="create_project").click().run()
    assert not app.exception
    assert app.error
    assert runner.created == runner.started == []
    assert runner.upload_count == 0
    assert not list(runner.jobs_dir.glob("*/state.json"))


def test_upload_creates_one_durable_mock_job_and_widget_reruns_do_not_restart_it(
    runner: ControlledRunner,
) -> None:
    """Native upload/submission persists ZIP bytes and starts exactly one background job."""
    app = _create(_app(), name="My show", minutes=3)
    assert not app.exception
    assert len(runner.created) == 1
    state = runner.created[0]
    assert state.params.target_seconds == 180
    assert state.params.name == "My show"
    assert state.params.llm_provider == "mock"
    assert state.params.whisper_device == "cpu"
    assert state.params.whisper_compute == "int8"
    assert state.params.zip_path is not None
    assert state.params.zip_path.read_bytes() == _zip_bytes()
    assert app.session_state["active_job"] == state.job_id
    assert runner.started == [state.job_id]
    app.run()
    app.run()
    assert runner.started == [state.job_id]
    assert not app.exception


@pytest.mark.parametrize("device, expected", [("Automatic", "auto"), ("GPU · CUDA", "cuda")])
def test_explicit_processing_device_is_saved_without_loading_models(
    runner: ControlledRunner, device: str, expected: str
) -> None:
    """Advanced device choices persist while the UI leaves model construction to the worker."""
    app = _create(_app(), device=device)
    assert not app.exception
    assert len(runner.created) == 1
    state = runner.created[0]
    assert state.params.whisper_device == expected
    assert state.params.whisper_compute == "auto"
    saved = read_json(runner.jobs_dir / state.job_id / "state.json", JobState)
    assert saved.params.whisper_device == expected
    assert saved.params.whisper_compute == "auto"


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Automatic · use subtitles first", "auto"),
        ("Embedded subtitles only", "subtitles"),
        ("Whisper · transcribe audio", "whisper"),
    ],
)
def test_dialogue_source_is_saved_without_transcription_or_additional_launches(
    runner: ControlledRunner, label: str, expected: str
) -> None:
    """Each dialogue mode persists on the job while model work stays with the worker."""
    app = _create(_app(), dialogue_source=label)
    assert not app.exception
    assert len(runner.created) == 1
    state = runner.created[0]
    saved = read_json(runner.jobs_dir / state.job_id / "state.json", JobState)
    assert state.params.transcript_source == saved.params.transcript_source == expected
    assert state.params.subtitle_language == saved.params.subtitle_language == "en"
    assert state.stages.transcribe.status == "pending"
    assert state.stages.transcribe.current is None
    app.run()
    assert not app.exception
    assert runner.started == [state.job_id]


def test_invalid_zip_submission_displays_error_without_publishing_a_job(
    runner: ControlledRunner,
) -> None:
    """Corrupt uploaded bytes fail visibly and leave no half-created saved project."""
    app = _app()
    app.file_uploader(key="upload_zip").set_value(("broken.zip", b"invalid", "application/zip"))
    app.run()
    app.button(key="create_project").click().run()
    assert not app.exception
    assert app.error
    assert runner.created == runner.started == []
    assert not list(runner.jobs_dir.glob("*/state.json"))


def test_unconfigured_openai_engine_is_rejected_before_job_creation(
    runner: ControlledRunner,
) -> None:
    """Cloud editing requires the explicit .env opt-in before any work can be launched."""
    app = _app()
    app.file_uploader(key="upload_zip").set_value(("show.zip", _zip_bytes(), "application/zip"))
    app.selectbox(key="editorial_engine").select("OpenAI · story-aware").run()
    app.button(key="create_project").click().run()
    assert not app.exception
    assert any("OpenAI" in error.value and ".env" in error.value for error in app.error)
    assert runner.created == runner.started == []


def test_busy_worker_disables_creation_of_another_project(runner: ControlledRunner) -> None:
    """Multiple browser projects cannot queue competing model work through the form."""
    app = _create(_app())
    app.button(key="new_project").click().run()
    app.file_uploader(key="upload_zip").set_value(("other.zip", _zip_bytes(), "application/zip"))
    app.run()
    assert not app.exception
    assert app.button(key="create_project").disabled
    assert len(runner.created) == len(runner.started) == 1
    _save_finished(runner, runner.created[0])
    app.run()
    assert not app.button(key="create_project").disabled


def test_progress_refresh_reads_latest_episode_checkpoint(runner: ControlledRunner) -> None:
    """A rerun reflects the worker's saved unit progress while leaving inference untouched."""
    app = _create(_app())
    state = runner.created[0]
    directory = runner.jobs_dir / state.job_id
    state.stages.ingest.status = "done"
    state.stages.ingest.episode_count = 3
    state.stages.transcribe.status = "in_progress"
    state.stages.transcribe.completed = ["E01"]
    state.stages.transcribe.pending = ["E02", "E03"]
    write_json(directory / "state.json", state)
    app.run()
    assert not app.exception
    progress = app.get("progress")
    assert progress
    assert any(0 < item.proto.value < 100 for item in progress)
    state.stages.transcribe.completed = ["E01", "E02"]
    state.stages.transcribe.pending = ["E03"]
    write_json(directory / "state.json", state)
    before = [item.proto.value for item in progress]
    app.run()
    assert [item.proto.value for item in app.get("progress")] != before
    assert runner.started == [state.job_id]


@pytest.mark.parametrize("audio_state", ["ready", "missing", "empty"])
def test_transcription_activity_explains_long_active_episode_without_running_inference(
    runner: ControlledRunner, monkeypatch: pytest.MonkeyPatch, audio_state: str
) -> None:
    """Active-episode context uses saved metadata and stays absent when work pauses or finishes."""
    app = _create(_app())
    state = runner.created[0]
    directory = runner.jobs_dir / state.job_id
    state.stages.ingest.status = "done"
    state.stages.ingest.episode_count = 3
    state.stages.ingest.completed = ["E01", "E02", "E03"]
    state.stages.ingest.episodes = [
        EpisodeInfo(
            episode=f"E{number:02d}",
            episode_number=number,
            source_path=directory / "episodes" / f"E{number:02d}.mp4",
            original_name=f"show.S01E{number:02d}.mp4",
            duration=duration,
        )
        for number, duration in enumerate((2760.0, 2940.0, 3000.0), 1)
    ]
    state.stages.transcribe.status = "in_progress"
    state.stages.transcribe.completed = ["E01"]
    state.stages.transcribe.pending = ["E02", "E03"]
    write_json(directory / "state.json", state)
    if audio_state == "ready":
        # Deliberately not a WAV: UI inspection must only stat the prepared audio,
        # never decode it or ask a model to recover an intra-episode percentage.
        (directory / "audio/E02.wav").write_bytes(b"metadata-only stand-in")
        prepared_time = (directory / "audio/E02.wav").stat().st_mtime
        monkeypatch.setattr(ui, "time", lambda: prepared_time + 180)
    elif audio_state == "empty":
        (directory / "audio/E02.wav").write_bytes(b"")
    checkpoint = (directory / "state.json").read_bytes()
    app.run()
    assert not app.exception
    notices = " ".join(info.value for info in app.info)
    if audio_state == "ready":
        assert "Transcribing E02" in notices
        assert "49.0 min" in notices
        assert "whole episode" in notices.casefold()
        assert "Audio prepared 3 min ago" in notices
    else:
        assert "Preparing audio for E02" in notices
        assert "Transcribing E02" not in notices
    source_runtime = " ".join(markdown.value for markdown in app.markdown)
    assert "Original runtime" in source_runtime and "2.4 h" in source_runtime
    assert runner.started == [state.job_id]
    assert (directory / "state.json").read_bytes() == checkpoint

    runner.running_id = None
    app.run()
    assert not app.exception
    notices = " ".join(info.value for info in app.info)
    assert "Transcribing E02" not in notices and "Preparing audio for E02" not in notices
    state.stages.transcribe.status = "done"
    state.stages.transcribe.completed = ["E01", "E02", "E03"]
    state.stages.transcribe.pending = []
    write_json(directory / "state.json", state)
    runner.running_id = state.job_id
    app.run()
    assert not app.exception
    notices = " ".join(info.value for info in app.info)
    assert "Transcribing E02" not in notices and "Preparing audio for E02" not in notices
    _save_finished(runner, state)
    app.run()
    assert not app.exception
    notices = " ".join(info.value for info in app.info)
    assert "Transcribing E02" not in notices and "Preparing audio for E02" not in notices
    assert runner.started == [state.job_id]


@pytest.mark.parametrize(
    "activity, processed_seconds, expected_fraction, expected_notice",
    [
        ("extracting_subtitles", 0.0, 1 / 3, "Extracting timed subtitles from E02"),
        ("extracting_audio", 0.0, 1 / 3, "Preparing audio for E02"),
        ("loading_model", 0.0, 1 / 3, "Loading Whisper small for E02"),
        ("decoding", 600.0, 0.5, "Decoded 10.0 of 20.0 min"),
    ],
)
def test_live_transcription_position_updates_activity_and_progress_without_committing_episode(
    runner: ControlledRunner,
    activity: str,
    processed_seconds: float,
    expected_fraction: float,
    expected_notice: str,
) -> None:
    """Stored decoder activity advances the UI without turning partial work into a checkpoint."""
    app = _create(_app())
    state = runner.created[0]
    directory = runner.jobs_dir / state.job_id
    state.stages.ingest.status = "done"
    state.stages.ingest.episode_count = 3
    state.stages.ingest.episodes = [
        EpisodeInfo(
            episode=f"E{number:02d}",
            episode_number=number,
            source_path=directory / "episodes" / f"E{number:02d}.mp4",
            original_name=f"episode{number}.mp4",
            duration=1200.0,
        )
        for number in range(1, 4)
    ]
    stage = state.stages.transcribe
    stage.status = "in_progress"
    stage.completed = ["E01"]
    stage.pending = ["E02", "E03"]
    stage.current = TranscriptionProgress(
        episode="E02",
        activity=activity,
        duration=1200.0,
        processed_seconds=processed_seconds,
    )
    # An invalid WAV with no decodable content ensures UI refreshes do not
    # inspect samples, rerun transcription, or eagerly load cached models.
    if activity != "extracting_subtitles":
        (directory / "audio/E02.wav").write_bytes(b"metadata only")
    write_json(directory / "state.json", state)
    checkpoint = (directory / "state.json").read_bytes()
    app.run()
    assert not app.exception
    notices = " ".join(info.value for info in app.info)
    assert expected_notice in notices
    if activity == "extracting_subtitles":
        assert "Whisper" not in notices and "GPU" not in notices
        assert "Preparing audio" not in notices
    assert ui._stage_progress(state, "transcribe") == pytest.approx(expected_fraction)
    assert app.get("progress")[2].proto.value == int(expected_fraction * 100)
    assert stage.completed == ["E01"] and stage.pending == ["E02", "E03"]
    assert runner.started == [state.job_id]
    assert (directory / "state.json").read_bytes() == checkpoint


@pytest.mark.parametrize(
    "episode, duration, processed_seconds, expected_fraction",
    [
        ("E02", 1200.0, 1200.0, (1 + 0.99) / 3),
        ("E01", 1200.0, 600.0, 1 / 3),
        ("E02", 0.0, 0.0, 1 / 3),
    ],
)
def test_transcription_progress_caps_uncommitted_work_and_ignores_stale_or_unknown_duration(
    episode: str, duration: float, processed_seconds: float, expected_fraction: float
) -> None:
    """Partial progress cannot count an episode twice or divide by an unknown duration."""
    state = JobState(job_id="progress-test")
    stage = state.stages.transcribe
    stage.status = "in_progress"
    stage.completed = ["E01"]
    stage.pending = ["E02", "E03"]
    stage.current = TranscriptionProgress(
        episode=episode,
        activity="decoding",
        duration=duration,
        processed_seconds=processed_seconds,
    )
    assert ui._stage_progress(state, "transcribe") == pytest.approx(expected_fraction)
    assert ui._stage_progress(state, "transcribe") < 2 / 3
    stage.status = "done"
    assert ui._stage_progress(state, "transcribe") == 1.0


def test_failure_is_visible_and_resume_keeps_same_job(runner: ControlledRunner) -> None:
    """A saved failure offers resumption, retaining the upload and completed checkpoints."""
    app = _create(_app())
    state = runner.created[0]
    state.stages.transcribe.status = "failed"
    state.errors.append(ErrorRecord(stage="transcribe", episode="E02", message="decode failure"))
    write_json(runner.jobs_dir / state.job_id / "state.json", state)
    runner.running_id = None
    runner.worker_errors[state.job_id] = "decode failure"
    app.run()
    assert not app.exception
    assert app.error
    assert any("decode failure" in text.value for text in app.text)
    assert not app.button(key=f"resume_{state.job_id}").disabled
    app.button(key=f"resume_{state.job_id}").click().run()
    assert runner.started == [state.job_id, state.job_id]
    assert len(runner.created) == 1
    assert app.session_state["active_job"] == state.job_id


def test_completed_project_has_preview_and_both_downloads_without_restarting_work(
    runner: ControlledRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finished artifacts populate preview/download widgets and stay idle when clicked."""
    registered: dict[str, tuple[Callable[[], bytes], str, str]] = {}
    original = ui.st.download_button

    def download(
        label: str,
        *,
        data: Callable[[], bytes],
        file_name: str,
        mime: str,
        key: str,
        **kwargs: Any,
    ) -> bool:
        registered[key] = (data, file_name, mime)
        return original(label, data=data, file_name=file_name, mime=mime, key=key, **kwargs)

    monkeypatch.setattr(ui.st, "download_button", download)
    app = _create(_app())
    state = runner.created[0]
    directory = _save_finished(runner, state)
    app.run()
    assert not app.exception
    assert app.get("video")
    movie = app.download_button(key=f"movie_{state.job_id}")
    subtitles = app.download_button(key=f"subtitles_{state.job_id}")
    assert not movie.disabled and not subtitles.disabled
    assert movie.proto.deferred_file_id and subtitles.proto.deferred_file_id
    assert movie.proto.ignore_rerun and subtitles.proto.ignore_rerun
    film_callback, film_name, film_mime = registered[f"movie_{state.job_id}"]
    subtitle_callback, subtitle_name, subtitle_mime = registered[f"subtitles_{state.job_id}"]
    assert film_callback() == (directory / "output/condensed_movie.mp4").read_bytes()
    assert subtitle_callback() == (directory / "output/condensed.srt").read_bytes()
    assert (film_name, film_mime) == ("condensed_movie.mp4", "video/mp4")
    assert (subtitle_name, subtitle_mime) == ("condensed.srt", "application/x-subrip")
    before = (directory / "state.json").read_bytes()
    movie.click().run()
    assert not app.exception
    app.download_button(key=f"subtitles_{state.job_id}").click().run()
    assert not app.exception
    assert runner.started == [state.job_id]
    assert (directory / "state.json").read_bytes() == before


def test_sidebar_can_return_to_new_project_and_reopen_saved_job(runner: ControlledRunner) -> None:
    """Sidebar navigation changes the visible project while preserving saved jobs."""
    app = _create(_app())
    state = runner.created[0]
    _save_finished(runner, state)
    app.run()
    app.button(key="new_project").click().run()
    assert app.session_state["active_job"] is None
    assert app.file_uploader(key="upload_zip")
    app.button(key=f"job_{state.job_id}").click().run()
    assert not app.exception
    assert app.session_state["active_job"] == state.job_id
    assert app.get("video")
    assert runner.started == [state.job_id]


def test_fresh_browser_session_reopens_and_resumes_a_saved_project(
    runner: ControlledRunner,
) -> None:
    """A browser reconnect can recover a persisted upload without creating another job."""
    state = runner.create(BytesIO(_zip_bytes()), 2, name="Saved project", cfg=Settings())
    state.stages.ingest.status = "done"
    state.stages.transcribe.status = "failed"
    write_json(runner.jobs_dir / state.job_id / "state.json", state)
    app = _app()
    app.button(key=f"job_{state.job_id}").click().run()
    assert not app.exception
    assert app.session_state["active_job"] == state.job_id
    app.button(key=f"resume_{state.job_id}").click().run()
    assert runner.started == [state.job_id]
    assert len(runner.created) == 1


@pytest.fixture
def fragment_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    """Record timer configuration while retaining actual Streamlit fragment rendering."""
    calls: list[tuple[str, str | None]] = []
    original = ui.st.fragment

    def fragment(
        function: Callable[..., Any] | None = None, *, run_every: str | None = None
    ) -> Any:
        def decorate(target: Callable[..., Any]) -> Any:
            calls.append((target.__name__, run_every))
            return original(target, run_every=run_every)

        return decorate(function) if function is not None else decorate

    monkeypatch.setattr(ui.st, "fragment", fragment)
    return calls


def test_completed_selected_project_does_not_poll_while_another_worker_runs(
    runner: ControlledRunner, fragment_calls: list[tuple[str, str | None]]
) -> None:
    """An unrelated background edit cannot cause a completed film's player to refresh."""
    app = _create(_app())
    selected = runner.created[0]
    _save_finished(runner, selected)
    other = runner.create(BytesIO(_zip_bytes()), 2, name="Other edit", cfg=Settings())
    runner.running_id = other.job_id
    fragment_calls.clear()
    app.run()
    assert not app.exception
    assert app.session_state["active_job"] == selected.job_id
    assert app.get("video")
    assert fragment_calls == [("_project_view", None)]
    assert runner.started == [selected.job_id]


def test_paused_unfinished_project_polls_until_completed(
    runner: ControlledRunner, fragment_calls: list[tuple[str, str | None]]
) -> None:
    """A paused project observes changes from another session and stops polling on completion."""
    state = runner.create(BytesIO(_zip_bytes()), 2, name="Paused edit", cfg=Settings())
    app = _app()
    fragment_calls.clear()
    app.button(key=f"job_{state.job_id}").click().run()
    assert not app.exception
    assert not runner.busy
    assert fragment_calls and all(interval == "2s" for _, interval in fragment_calls)
    assert not app.button(key=f"resume_{state.job_id}").disabled
    runner.start(state.job_id)
    app.run()
    assert fragment_calls[-1] == ("_project_view", "2s")
    _save_finished(runner, state)
    fragment_calls.clear()
    app.run()
    assert not app.exception
    assert app.get("video")
    assert fragment_calls == [("_project_view", None)]


def test_finished_state_with_missing_output_displays_error(runner: ControlledRunner) -> None:
    """A missing movie is explained to the user instead of crashing Streamlit."""
    app = _create(_app())
    state = runner.created[0]
    directory = _save_finished(runner, state)
    (directory / "output/condensed_movie.mp4").unlink()
    app.run()
    assert not app.exception
    assert app.error or app.warning
    assert not app.get("video")
    assert read_json(directory / "state.json", JobState).stages.render.status == "done"
