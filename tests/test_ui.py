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
    ErrorRecord,
    JobState,
    SelectedClip,
    SelectionDoc,
    SelectionStats,
    read_json,
    write_json,
)


class ControlledRunner(JobRunner):
    """Use real upload persistence while manually controlling the background worker."""

    def __init__(self, jobs_dir: Path) -> None:
        """Record create/start requests and leave all model work disabled."""
        super().__init__(jobs_dir)
        self.created: list[JobState] = []
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
        state = super().create(upload, target_minutes, name=name, cfg=cfg)
        self.created.append(state)
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
) -> AppTest:
    app.file_uploader(key="upload_zip").set_value(("show.zip", _zip_bytes(), "application/zip"))
    app.text_input(key="show_name").set_value(name)
    app.number_input(key="target_minutes").set_value(minutes)
    app.selectbox(key="editorial_engine").select("Preview · sample scoring")
    if device is not None:
        app.selectbox(key="processing_device").select(device)
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
    assert app.file_uploader(key="upload_zip").allowed_type == [".zip"]
    assert app.button(key="create_project").disabled
    assert app.number_input(key="target_minutes").value > 0
    assert app.selectbox(key="processing_device").value == "CPU"
    assert runner.created == runner.started == []


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
