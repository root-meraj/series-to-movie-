"""Exercise the persistent HTTP contract with durable files and no trained models."""

from __future__ import annotations

import hashlib
import hmac
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from sts.app import web_api
from sts.app.runner import JobRunner
from sts.schemas import (
    ErrorRecord,
    JobState,
    RequiredRange,
    SelectionDoc,
    SelectionStats,
    WebHealthDoc,
    WebJobDetailDoc,
    WebMediaTicketDoc,
    read_json,
    write_json,
)

_TOKEN = "a7ad9fde60634945966798d74f7489ac-test-token"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}
_ORIGIN = "https://season.example"


class _Runner(JobRunner):
    """Record processing controls while retaining the real disk-spooled job creation."""

    def __init__(self, jobs_dir: Path) -> None:
        """Create an idle deterministic runner without launching processing."""
        super().__init__(jobs_dir)
        self.active: str | None = None
        self.stopping: str | None = None
        self.worker_error: str | None = None
        self.refinements: list[RequiredRange] | None = None
        self.closed = False

    @property
    def busy(self) -> bool:
        """Return the synthetic worker's activity state."""
        return self.active is not None

    @property
    def active_job_id(self) -> str | None:
        """Expose the synthetic worker's active project for lifespan handling."""
        return self.active

    def start(self, job_id: str) -> bool:
        """Start a saved project without running its pipeline or model transports."""
        if self.active is not None and self.active != job_id:
            raise RuntimeError("busy")
        if self.get_state(job_id).stages.render.status == "done" or self.active == job_id:
            return False
        self.active = job_id
        self.stopping = None
        return True

    def request_stop(self, job_id: str) -> bool:
        """Record a safe stop request for the synthetic active worker."""
        if self.active != job_id:
            return False
        self.stopping = job_id
        return True

    def is_running(self, job_id: str) -> bool:
        """Return whether this project owns the synthetic worker."""
        return self.active == job_id

    def is_stopping(self, job_id: str) -> bool:
        """Return whether a matching stop request is waiting to be acknowledged."""
        return self.stopping == job_id

    def error(self, job_id: str) -> str | None:
        """Return an injected worker error without creating an actual failed task."""
        return self.worker_error

    def refine(
        self,
        job_id: str,
        *,
        must_keep_ranges: list[RequiredRange] | None = None,
        target_minutes: float | None = None,
    ) -> JobState:
        """Clone a test checkpoint and record the validated protected moments."""
        if self.busy:
            raise RuntimeError("busy")
        self.refinements = must_keep_ranges
        if any(interval.end > 90 for interval in must_keep_ranges or []):
            raise ValueError("outside episode")
        state = self.get_state(job_id).model_copy(deep=True)
        state.job_id = "refined"
        state.params.must_keep_ranges = must_keep_ranges or []
        state.params.parent_job_id = job_id
        state.stages.render.status = "pending"
        write_json(self.jobs_dir / state.job_id / "state.json", state)
        return state

    def shutdown(self, *, wait: bool = True) -> None:
        """Release the real idle executor and record orderly application shutdown."""
        self.closed = True
        super().shutdown(wait=wait)


@pytest.fixture
def settings(tmp_path: Path) -> web_api.WebSettings:
    """Keep tokens, workspace paths, and frontend origins independent of developer settings."""
    return web_api.WebSettings(
        _env_file=None,
        access_token=_TOKEN,
        allowed_origins=(_ORIGIN,),
        jobs_dir=tmp_path / "jobs",
    )


@pytest.fixture
def runner(settings: web_api.WebSettings) -> _Runner:
    """Return a deterministic injected runner using the API's real workspace directory."""
    return _Runner(settings.jobs_dir)


@pytest.fixture
def client(settings: web_api.WebSettings, runner: _Runner) -> Iterator[TestClient]:
    """Enter the real app lifespan and dispatch requests through Starlette's ASGI transport."""
    with TestClient(web_api.create_web_app(settings, runner)) as test_client:
        yield test_client


def _archive() -> bytes:
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("episode-one.mp4", b"synthetic video")
    return stream.getvalue()


def _state(runner: _Runner, *, job_id: str = "project", done: bool = False) -> JobState:
    state = JobState(job_id=job_id)
    state.params.name = "A saved show"
    if done:
        state.stages.render.status = "done"
        state.stages.render.video_done = True
        state.stages.render.subtitles_done = True
        output = runner.jobs_dir / job_id / "output"
        output.mkdir(parents=True)
        (output / "condensed_movie.mp4").write_bytes(b"0123456789movie")
        (output / "condensed.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8"
        )
    write_json(runner.jobs_dir / job_id / "state.json", state)
    return state


def test_settings_fail_closed_without_a_token(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        web_api.WebSettings(_env_file=None, allowed_origins=(_ORIGIN,))


@pytest.mark.parametrize("token", ["short", " " * 32, "a" * 31 + "\n"])
def test_settings_reject_short_or_blank_tokens(token: str) -> None:
    with pytest.raises(ValidationError):
        web_api.WebSettings(_env_file=None, access_token=token, allowed_origins=(_ORIGIN,))


def test_settings_secrets_come_from_dotenv_not_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / ".env"
    path.write_text(
        f"STS_WEB_ACCESS_TOKEN={_TOKEN}\nSTS_WEB_ALLOWED_ORIGINS={_ORIGIN}\n", encoding="utf-8"
    )
    monkeypatch.setenv("STS_WEB_ACCESS_TOKEN", "process-secret" * 4)
    configured = web_api.WebSettings(_env_file=path)
    assert configured.access_token.get_secret_value() == _TOKEN
    assert _TOKEN not in repr(configured)
    assert "access_token" not in configured.model_dump()


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "http://season.example",
        "https://user:secret@season.example",
        "https://season.example/path",
        "https://season.example?token=secret",
        "",
    ],
)
def test_settings_reject_insecure_or_ambiguous_origins(origin: str) -> None:
    with pytest.raises(ValidationError):
        web_api.WebSettings(_env_file=None, access_token=_TOKEN, allowed_origins=(origin,))


def test_settings_accept_explicit_origins_and_loopback_development() -> None:
    configured = web_api.WebSettings(
        _env_file=None,
        access_token=_TOKEN,
        allowed_origins=f"{_ORIGIN}/, http://localhost:5173, http://127.0.0.1:5173",
        public_url="https://processor.example/",
    )
    assert configured.allowed_origins == (
        _ORIGIN,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    assert configured.public_url == "https://processor.example"


@pytest.mark.parametrize("url", ["http://processor.example", "http://localhost:8765"])
def test_settings_require_https_for_published_backend_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        web_api.WebSettings(
            _env_file=None,
            access_token=_TOKEN,
            allowed_origins=(_ORIGIN,),
            public_url=url,
        )


def test_health_is_public_and_contains_no_secrets(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    health = WebHealthDoc.model_validate(response.json())
    assert health.status == "ok"
    assert health.max_upload_bytes == 2 * 1024**3
    assert health.default_target_minutes == 120
    assert _TOKEN not in response.text
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/jobs"),
        ("POST", "/api/jobs"),
        ("GET", "/api/jobs/project"),
        ("POST", "/api/jobs/project/stop"),
        ("POST", "/api/jobs/project/resume"),
        ("POST", "/api/jobs/project/refine"),
        ("POST", "/api/jobs/project/media-ticket"),
    ],
)
def test_all_workspace_routes_require_bearer_auth(
    client: TestClient,
    method: str,
    path: str,
) -> None:
    response = client.request(method, path, content=b"not a valid request body")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_unauthorized_upload_is_not_parsed(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_parse(self: object) -> None:
        raise AssertionError("An unauthorized upload must never be parsed")

    monkeypatch.setattr(web_api.MultiPartParser, "parse", forbidden_parse)
    assert client.post("/api/jobs", files={"archive": ("show.zip", _archive())}).status_code == 401


def test_wrong_bearer_token_and_other_auth_schemes_are_denied(client: TestClient) -> None:
    for value in ("Bearer wrong", f"Basic {_TOKEN}", "Bearer", ""):
        assert client.get("/api/jobs", headers={"Authorization": value}).status_code == 401


def test_cors_allows_only_the_explicit_frontend(client: TestClient) -> None:
    good = client.options(
        "/api/jobs",
        headers={
            "Origin": _ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    assert good.status_code == 200
    assert good.headers["access-control-allow-origin"] == _ORIGIN
    assert "access-control-allow-credentials" not in good.headers
    bad = client.options(
        "/api/jobs",
        headers={
            "Origin": "https://elsewhere.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert bad.status_code == 400
    assert "access-control-allow-origin" not in bad.headers
    authenticated = client.get("/api/jobs", headers={**_AUTH, "Origin": _ORIGIN})
    assert authenticated.headers["access-control-allow-origin"] == _ORIGIN


def test_empty_workspace_lists_no_jobs(client: TestClient) -> None:
    assert client.get("/api/jobs", headers=_AUTH).json() == []


def test_saved_jobs_are_listed_and_detail_uses_durable_contract(
    client: TestClient,
    runner: _Runner,
) -> None:
    _state(runner)
    selection = SelectionDoc(
        target_seconds=7200,
        selected_total_seconds=0,
        stats=SelectionStats(scenes_kept=0, scenes_dropped=0, original_seconds=90),
    )
    write_json(runner.jobs_dir / "project/selected.json", selection)
    listing = client.get("/api/jobs", headers=_AUTH).json()
    assert len(listing) == 1
    assert JobState.model_validate(listing[0]).job_id == "project"
    detail = WebJobDetailDoc.model_validate(client.get("/api/jobs/project", headers=_AUTH).json())
    assert detail.selection == selection
    assert detail.state.params.name == "A saved show"
    assert detail.running is False and detail.error is None


def test_missing_project_returns_a_clear_not_found(client: TestClient) -> None:
    assert client.get("/api/jobs/missing", headers=_AUTH).status_code == 404
    assert client.post("/api/jobs/missing/resume", headers=_AUTH).status_code == 404
    assert client.post("/api/jobs/missing/stop", headers=_AUTH).status_code == 404
    assert client.post("/api/jobs/missing/refine", headers=_AUTH, json={}).status_code == 404


def test_zip_upload_streams_to_a_durable_job_and_starts_processing(
    client: TestClient,
    runner: _Runner,
) -> None:
    contents = _archive()
    response = client.post(
        "/api/jobs",
        headers=_AUTH,
        files={"archive": ("../../untrusted.zip", contents, "application/zip")},
        data={"target_minutes": "60", "name": " My Show "},
    )
    assert response.status_code == 201
    detail = WebJobDetailDoc.model_validate(response.json())
    assert detail.running is True
    assert runner.active == detail.state.job_id
    assert detail.state.params.target_seconds == 3600
    assert detail.state.params.name == "My Show"
    assert (runner.jobs_dir / detail.state.job_id / "input/source.zip").read_bytes() == contents
    assert not (runner.jobs_dir.parent / "untrusted.zip").exists()


def test_upload_uses_backend_defaults_when_optional_fields_are_absent(client: TestClient) -> None:
    response = client.post("/api/jobs", headers=_AUTH, files={"archive": ("show.zip", _archive())})
    assert response.status_code == 201
    assert response.json()["state"]["params"]["target_seconds"] == 7200


@pytest.mark.parametrize(
    "filename,contents",
    [
        ("show.mp4", b"video"),
        ("show.zip", b"not a zip"),
        ("show.zip", b""),
    ],
)
def test_invalid_archives_never_publish_a_job(
    client: TestClient,
    runner: _Runner,
    filename: str,
    contents: bytes,
) -> None:
    response = client.post("/api/jobs", headers=_AUTH, files={"archive": (filename, contents)})
    assert response.status_code == 400
    assert list(runner.jobs_dir.glob("*/state.json")) == []


@pytest.mark.parametrize("minutes", ["nan", "inf", "0", "-1", "0.4", "1441", "garbage"])
def test_invalid_target_minutes_are_rejected_before_copying(
    client: TestClient,
    runner: _Runner,
    minutes: str,
) -> None:
    response = client.post(
        "/api/jobs",
        headers=_AUTH,
        files={"archive": ("show.zip", _archive())},
        data={"target_minutes": minutes},
    )
    assert response.status_code == 422
    assert list(runner.jobs_dir.glob("*/state.json")) == []


def test_remote_requests_cannot_override_provider_or_local_paths(client: TestClient) -> None:
    response = client.post(
        "/api/jobs",
        headers=_AUTH,
        files={"archive": ("show.zip", _archive())},
        data={"llm_provider": "mock"},
    )
    assert response.status_code == 422


def test_multipart_limits_reject_multiple_archives(client: TestClient) -> None:
    response = client.post(
        "/api/jobs",
        headers=_AUTH,
        files=[
            ("archive", ("show.zip", _archive())),
            ("archive", ("second.zip", _archive())),
        ],
    )
    assert response.status_code == 400


def test_actual_archive_length_is_checked_even_without_trusted_content_length(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_api, "MAX_UPLOAD_BYTES", 16)
    response = client.post("/api/jobs", headers=_AUTH, files={"archive": ("show.zip", _archive())})
    assert response.status_code == 413


def test_content_length_limit_is_checked_before_multipart_parsing(client: TestClient) -> None:
    response = client.post(
        "/api/jobs",
        headers={**_AUTH, "Content-Length": str(3 * 1024**3)},
        content=b"",
    )
    assert response.status_code == 413


def test_streamed_request_limit_rejects_unbounded_multipart_chunks(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_api, "MAX_UPLOAD_BYTES", 1)
    monkeypatch.setattr(web_api, "_MULTIPART_OVERHEAD", 1)
    boundary = "test-boundary"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="archive"; '
            f'filename="show.zip"\r\n\r\n'
        ).encode()
        + _archive()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    response = client.post(
        "/api/jobs",
        headers={**_AUTH, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        content=iter([body[:10], body[10:]]),
    )
    assert response.status_code == 413


def test_stop_and_resume_keep_completed_checkpoints(client: TestClient, runner: _Runner) -> None:
    state = _state(runner)
    state.stages.ingest.status = "done"
    write_json(runner.jobs_dir / "project/state.json", state)
    resumed = client.post("/api/jobs/project/resume", headers=_AUTH)
    assert resumed.status_code == 200
    assert resumed.json()["running"] is True
    stopped = client.post("/api/jobs/project/stop", headers=_AUTH)
    assert stopped.status_code == 200
    assert stopped.json()["stopping"] is True
    assert (
        read_json(runner.jobs_dir / "project/state.json", JobState).stages.ingest.status == "done"
    )


def test_busy_worker_returns_conflict_without_queuing_jobs(
    client: TestClient, runner: _Runner
) -> None:
    _state(runner)
    _state(runner, job_id="second")
    runner.active = "project"
    assert client.post("/api/jobs/second/resume", headers=_AUTH).status_code == 409
    assert client.post("/api/jobs/second/refine", headers=_AUTH, json={}).status_code == 409
    assert (
        client.post(
            "/api/jobs",
            headers=_AUTH,
            files={"archive": ("show.zip", _archive())},
        ).status_code
        == 409
    )


def test_repeated_resume_of_active_or_completed_job_is_idempotent(
    client: TestClient,
    runner: _Runner,
) -> None:
    _state(runner)
    runner.active = "project"
    assert client.post("/api/jobs/project/resume", headers=_AUTH).status_code == 200
    runner.active = None
    _state(runner, job_id="complete", done=True)
    response = client.post("/api/jobs/complete/resume", headers=_AUTH)
    assert response.status_code == 200 and response.json()["running"] is False


def test_refine_passes_validated_ranges_and_preserves_parent(
    client: TestClient, runner: _Runner
) -> None:
    _state(runner, done=True)
    before = (runner.jobs_dir / "project/state.json").read_bytes()
    response = client.post(
        "/api/jobs/project/refine",
        headers=_AUTH,
        json={
            "must_keep_ranges": [{"episode": "E01", "start": 12.0, "end": 25.0}],
        },
    )
    assert response.status_code == 201
    detail = WebJobDetailDoc.model_validate(response.json())
    assert detail.state.params.parent_job_id == "project"
    assert detail.running is True
    assert runner.refinements == [RequiredRange(episode="E01", start=12, end=25)]
    assert (runner.jobs_dir / "project/state.json").read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        {"must_keep_ranges": [{"episode": "E01", "start": 25, "end": 12}]},
        {"must_keep_ranges": [{"episode": "../E01", "start": 12, "end": 25}]},
        {"must_keep_ranges": [{"episode": "E01", "start": -1, "end": 25}]},
        {"llm_provider": "mock"},
    ],
)
def test_refine_rejects_invalid_ranges_and_configuration_overrides(
    client: TestClient,
    runner: _Runner,
    payload: dict[str, Any],
) -> None:
    _state(runner)
    assert client.post("/api/jobs/project/refine", headers=_AUTH, json=payload).status_code == 422
    assert runner.refinements is None


def test_refine_handles_out_of_episode_ranges(client: TestClient, runner: _Runner) -> None:
    _state(runner)
    response = client.post(
        "/api/jobs/project/refine",
        headers=_AUTH,
        json={
            "must_keep_ranges": [{"episode": "E01", "start": 80, "end": 100}],
        },
    )
    assert response.status_code == 422


def test_refine_request_size_is_bounded(client: TestClient, runner: _Runner) -> None:
    _state(runner)
    response = client.post("/api/jobs/project/refine", headers=_AUTH, content=b" " * (65536 + 1))
    assert response.status_code == 413


def test_errors_are_redacted_before_becoming_json(client: TestClient, runner: _Runner) -> None:
    state = _state(runner)
    state.errors.append(ErrorRecord(stage="nlp", message=f"provider failed with token {_TOKEN}"))
    write_json(runner.jobs_dir / "project/state.json", state)
    runner.worker_error = f"failed {_TOKEN}"
    response = client.get("/api/jobs/project", headers=_AUTH)
    assert _TOKEN not in response.text
    assert response.json()["error"] == "failed [redacted]"
    assert "[redacted]" in response.json()["state"]["errors"][0]["message"]


def test_ticket_requires_finished_movie_and_subtitles(client: TestClient, runner: _Runner) -> None:
    _state(runner)
    assert client.post("/api/jobs/project/media-ticket", headers=_AUTH).status_code == 409
    _state(runner, job_id="done", done=True)
    (runner.jobs_dir / "done/output/condensed.srt").unlink()
    assert client.post("/api/jobs/done/media-ticket", headers=_AUTH).status_code == 404


def test_signed_urls_support_video_seek_and_subtitle_downloads(
    client: TestClient,
    runner: _Runner,
) -> None:
    _state(runner, done=True)
    response = client.post("/api/jobs/project/media-ticket", headers=_AUTH)
    assert response.status_code == 200
    ticket = WebMediaTicketDoc.model_validate(response.json())
    assert ticket.expires_in == 8 * 3600
    assert _TOKEN not in response.text
    movie = client.get(ticket.movie_url, headers={"Range": "bytes=2-5"})
    assert movie.status_code == 206
    assert movie.content == b"2345"
    assert movie.headers["content-range"] == "bytes 2-5/15"
    assert movie.headers["accept-ranges"] == "bytes"
    assert movie.headers["content-type"] == "video/mp4"
    assert movie.headers["content-disposition"].startswith("inline;")
    download = client.get(ticket.subtitles_url + "&download=1")
    assert download.status_code == 200 and "Hello" in download.text
    assert download.headers["content-disposition"].startswith("attachment;")


def test_media_bearer_auth_and_head_do_not_require_a_ticket(
    client: TestClient, runner: _Runner
) -> None:
    _state(runner, done=True)
    head = client.head("/api/jobs/project/movie", headers=_AUTH)
    assert head.status_code == 200 and head.content == b""
    assert head.headers["content-length"] == "15"
    assert client.get("/api/jobs/project/movie", headers=_AUTH).content == b"0123456789movie"
    assert client.get("/api/jobs/project/movie").status_code == 401


def test_media_invalid_ranges_are_not_silently_served(client: TestClient, runner: _Runner) -> None:
    _state(runner, done=True)
    response = client.get("/api/jobs/project/movie", headers={**_AUTH, "Range": "bytes=100-200"})
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */15"


def test_media_ticket_is_scoped_to_exact_job_and_file(client: TestClient, runner: _Runner) -> None:
    _state(runner, done=True)
    _state(runner, job_id="second", done=True)
    issued = client.post("/api/jobs/project/media-ticket", headers=_AUTH).json()
    query = urlsplit(issued["movie_url"]).query
    assert client.get(f"/api/jobs/second/movie?{query}").status_code == 401
    assert client.get(f"/api/jobs/project/subtitles?{query}").status_code == 401
    token = parse_qs(query)["ticket"][0]
    assert client.get(f"/api/jobs/project/movie?ticket={token[:-1]}X").status_code == 401


def test_media_ticket_expiry_is_enforced(
    client: TestClient,
    runner: _Runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _state(runner, done=True)
    monkeypatch.setattr(web_api.time, "time", lambda: 1000.0)
    issued = client.post("/api/jobs/project/media-ticket", headers=_AUTH).json()
    monkeypatch.setattr(web_api.time, "time", lambda: 1000.0 + 8 * 3600)
    assert client.get(issued["movie_url"]).status_code == 401


def test_media_ticket_bad_payload_is_denied_even_with_a_valid_signature(
    client: TestClient,
    runner: _Runner,
) -> None:
    _state(runner, done=True)
    encoded = "bm90LWpzb24"
    signature = hmac.new(_TOKEN.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    assert client.get(f"/api/jobs/project/movie?ticket={encoded}.{signature}").status_code == 401


def test_media_urls_use_explicit_https_backend_origin(
    settings: web_api.WebSettings,
    runner: _Runner,
) -> None:
    settings.public_url = "https://processor.example"
    _state(runner, done=True)
    with TestClient(web_api.create_web_app(settings, runner)) as local:
        response = local.post("/api/jobs/project/media-ticket", headers=_AUTH)
    assert response.json()["movie_url"].startswith(
        "https://processor.example/api/jobs/project/movie?"
    )


def test_unknown_export_paths_are_not_served(client: TestClient, runner: _Runner) -> None:
    _state(runner, done=True)
    assert client.get("/api/jobs/project/state.json", headers=_AUTH).status_code == 404
    assert client.get("/api/jobs/project/other-file", headers=_AUTH).status_code == 404


def test_external_output_symlinks_are_rejected(
    client: TestClient,
    runner: _Runner,
    tmp_path: Path,
) -> None:
    _state(runner, done=True)
    exported = runner.jobs_dir / "project/output/condensed_movie.mp4"
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"private external video")
    exported.unlink()
    try:
        exported.symlink_to(outside)
    except OSError:
        pytest.skip("This platform does not permit test symlinks")
    assert client.get("/api/jobs/project/movie", headers=_AUTH).status_code == 404
    assert client.post("/api/jobs/project/media-ticket", headers=_AUTH).status_code == 404


def test_external_state_symlinks_are_not_listed_or_exposed(
    client: TestClient,
    runner: _Runner,
    tmp_path: Path,
) -> None:
    state = _state(runner)
    outside = tmp_path / "outside.json"
    write_json(outside, state)
    saved = runner.jobs_dir / "project/state.json"
    saved.unlink()
    try:
        saved.symlink_to(outside)
    except OSError:
        pytest.skip("This platform does not permit test symlinks")
    assert client.get("/api/jobs/project", headers=_AUTH).status_code == 404
    assert client.get("/api/jobs", headers=_AUTH).json() == []


def test_export_symlinks_cannot_redirect_to_other_files_inside_the_job(
    client: TestClient,
    runner: _Runner,
) -> None:
    _state(runner, done=True)
    exported = runner.jobs_dir / "project/output/condensed_movie.mp4"
    exported.unlink()
    try:
        exported.symlink_to(runner.jobs_dir / "project/state.json")
    except OSError:
        pytest.skip("This platform does not permit test symlinks")
    assert client.get("/api/jobs/project/movie", headers=_AUTH).status_code == 404


def test_state_identity_must_match_its_workspace_directory(
    client: TestClient, runner: _Runner
) -> None:
    write_json(runner.jobs_dir / "project/state.json", JobState(job_id="other"))
    assert client.get("/api/jobs/project", headers=_AUTH).status_code == 404


def test_lifespan_requests_safe_stop_then_shuts_down(
    settings: web_api.WebSettings,
    runner: _Runner,
) -> None:
    _state(runner)
    runner.active = "project"
    with TestClient(web_api.create_web_app(settings, runner)):
        assert not runner.closed
    assert runner.stopping == "project"
    assert runner.closed


def test_factory_rejects_runner_with_a_different_workspace(
    settings: web_api.WebSettings,
    tmp_path: Path,
) -> None:
    other = _Runner(tmp_path / "elsewhere")
    try:
        with pytest.raises(ValueError, match="same jobs directory"):
            web_api.create_web_app(settings, other)
    finally:
        other.shutdown()


def test_startup_without_credentials_exits_cleanly(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STS_WEB_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("STS_WEB_ALLOWED_ORIGINS", raising=False)
    with pytest.raises(SystemExit) as exc:
        web_api.main()
    assert exc.value.code == 2
    assert "Set STS_WEB_ACCESS_TOKEN" in caplog.text
    assert _TOKEN not in caplog.text

def test_configured_upload_limit_is_advertised_and_enforced(
    settings: web_api.WebSettings, runner: _Runner,
) -> None:
    """Use the same operator ceiling for health and actual multipart bytes."""
    settings.max_upload_bytes = 16
    with TestClient(web_api.create_web_app(settings, runner)) as test_client:
        assert test_client.get('/api/health').json()['max_upload_bytes'] == 16
        response = test_client.post('/api/jobs', headers=_AUTH,
                                    files={'archive': ('show.zip', _archive())})
        assert response.status_code == 413
        assert runner.active is None


def test_large_season_limit_accepts_declared_length_above_default(
    settings: web_api.WebSettings, runner: _Runner,
) -> None:
    """A larger deployment ceiling does not reject a valid upload at the old ceiling."""
    settings.max_upload_bytes = 16 * 1024**3
    with TestClient(web_api.create_web_app(settings, runner)) as test_client:
        assert test_client.get('/api/health').json()['max_upload_bytes'] == 16 * 1024**3
        response = test_client.post('/api/jobs',
                                    headers={**_AUTH, 'Content-Length': str(10 * 1024**3)},
                                    files={'archive': ('show.zip', _archive())})
        assert response.status_code == 201


def test_archive_storage_shortage_does_not_start_worker(
    client: TestClient, runner: _Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an archive if extraction and rendering would exhaust persistent disk."""
    from types import SimpleNamespace

    monkeypatch.setattr(web_api.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    response = client.post('/api/jobs', headers=_AUTH,
                           files={'archive': ('show.zip', _archive())})
    assert response.status_code == 507
    assert runner.active is None
