"""Authenticated persistent API for the mobile frontend and its streamed exports.

Run this service on the machine hosting the pipeline, independently of the static
frontend. Personal accounts authorize owned projects; the private operator token
retains access to existing jobs for administration.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import re
import secrets
import shutil
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Any, BinaryIO
from urllib.parse import urlencode, urlsplit
from zipfile import BadZipFile, ZipFile, is_zipfile

import anyio
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from rich.console import Console
from rich.logging import RichHandler
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from sts.app.accounts import SESSION_SECONDS, AccountStore
from sts.app.runner import JobRunner
from sts.config import Settings
from sts.schemas import (
    JobState,
    SelectionDoc,
    WebCredentialsDoc,
    WebHealthDoc,
    WebJobDetailDoc,
    WebMediaTicketDoc,
    WebRefineRequestDoc,
    WebSessionDoc,
    read_json,
)

MAX_UPLOAD_BYTES = 2 * 1024**3
MEDIA_TICKET_SECONDS = 8 * 60 * 60
_MULTIPART_OVERHEAD = 64 * 1024
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_MOVIE = "condensed_movie.mp4"
_SUBTITLES = "condensed.srt"


def _is_loopback(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_url(value: str, *, allow_loopback: bool) -> str:
    parsed = urlsplit(value)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or (
            parsed.scheme != "https"
            and not (allow_loopback and parsed.scheme == "http" and _is_loopback(parsed.hostname))
        )
    ):
        raise ValueError("Use an HTTPS origin, or HTTP on loopback for local development")
    _ = parsed.port  # Validate the port before CORS or generated media URLs use it.
    return value.rstrip("/")


class WebSettings(BaseSettings):
    """Explicit deployment settings; an absent access token prevents startup."""

    model_config = SettingsConfigDict(
        env_prefix="STS_WEB_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    access_token: SecretStr = Field(exclude=True, repr=False)
    allowed_origins: Annotated[tuple[str, ...], NoDecode]
    jobs_dir: Path = Path("data/jobs")
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    public_url: str | None = None
    trusted_proxy_ips: str = "127.0.0.1"
    accounts_enabled: bool = False
    max_accounts: int = Field(default=100, ge=1, le=1000)
    max_upload_bytes: int | None = Field(default=None, gt=0, le=16 * 1024**3)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource | Callable[[], dict[str, Any]], ...]:
        """Keep deployed secrets in .env; permit explicit injection for isolated factories."""

        def nonsecret_environment() -> dict[str, Any]:
            """Read nonsecret process overrides while reserving the token for .env."""
            values = env_settings()
            values.pop("access_token", None)
            return values

        return init_settings, nonsecret_environment, dotenv_settings

    @field_validator("access_token")
    @classmethod
    def validate_token(cls, value: SecretStr) -> SecretStr:
        """Require a long nonblank token supplied by the backend operator."""
        token = value.get_secret_value()
        if len(token) < 32 or any(character.isspace() for character in token):
            raise ValueError("STS_WEB_ACCESS_TOKEN needs at least 32 non-whitespace characters")
        return value

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        """Accept the deployment's comma-separated list of frontend origins."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject wildcard, credential-bearing, and insecure remote origins."""
        if not value:
            raise ValueError("STS_WEB_ALLOWED_ORIGINS must name at least one frontend origin")
        return tuple(dict.fromkeys(_validate_url(origin, allow_loopback=True) for origin in value))

    @field_validator("public_url")
    @classmethod
    def validate_public_url(cls, value: str | None) -> str | None:
        """Keep published media URLs on an explicit HTTPS backend origin."""
        return _validate_url(value, allow_loopback=False) if value else None


class _WebAPI:
    def __init__(self, settings: WebSettings, runner: JobRunner) -> None:
        self.settings = settings
        self.runner = runner
        self.jobs_dir = settings.jobs_dir.resolve()
        self.token = settings.access_token.get_secret_value()
        self.pipeline_settings = Settings()
        self._secrets = tuple(
            value
            for value in (self.token, self.pipeline_settings.openai_api_key.get_secret_value())
            if value
        )
        self._mutation_lock = anyio.Lock()
        self._auth_lock = anyio.Lock()
        self.accounts = (
            AccountStore(self.jobs_dir.parent / "accounts/accounts.sqlite3", settings.max_accounts)
            if settings.accounts_enabled
            else None
        )

    def _bearer(self, request: Request) -> str:
        scheme, _, value = request.headers.get("authorization", "").partition(" ")
        return value if scheme.casefold() == "bearer" else ""

    def _authorized(self, request: Request) -> bool:
        value = self._bearer(request)
        if secrets.compare_digest(value.encode(), self.token.encode()):
            request.state.user_id = None  # Private operator credential; never sent to visitors.
            return True
        user = self.accounts.resolve(value) if self.accounts and value else None
        if user:
            request.state.user_id = user.id
            return True
        return False

    def _require_auth(self, request: Request) -> None:
        if not self._authorized(request):
            raise HTTPException(401, "Sign in to continue", {"WWW-Authenticate": "Bearer"})
        job_id = request.path_params.get("job_id")
        if job_id and request.state.user_id is not None:
            if not self.accounts or not self.accounts.owns(job_id, request.state.user_id):
                raise HTTPException(404, "Project not found")

    async def account_auth(self, request: Request) -> Response:
        """Register or sign in using bounded inputs and serialized password hashing."""
        if not self.accounts:
            raise HTTPException(404, "Accounts are not enabled on this workspace")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > 4096:
                raise HTTPException(413, "Account details are too large")
            body.extend(chunk)
        try:
            credentials = WebCredentialsDoc.model_validate_json(bytes(body))
        except ValidationError:
            raise HTTPException(
                422, "Use a valid email and a password of 8 to 128 characters"
            ) from None
        source = request.client.host if request.client else "unknown"
        await anyio.to_thread.run_sync(self.accounts.throttle, source, credentials.email)
        async with self._auth_lock:
            operation = (
                self.accounts.register
                if request.url.path.endswith("register")
                else self.accounts.login
            )
            user = await anyio.to_thread.run_sync(
                operation, credentials.email, credentials.password
            )
            token = await anyio.to_thread.run_sync(self.accounts.issue_session, user.id)
        return self._json(
            WebSessionDoc(user=user, session_token=token, expires_in=SESSION_SECONDS).model_dump(
                mode="json"
            ),
            status_code=201 if request.url.path.endswith("register") else 200,
        )

    async def account_me(self, request: Request) -> Response:
        """Return the current personal account, without accepting the operator key."""
        user = self.accounts.resolve(self._bearer(request)) if self.accounts else None
        if user is None:
            raise HTTPException(401, "Sign in to continue")
        return self._json(user.model_dump(mode="json"))

    async def account_logout(self, request: Request) -> Response:
        """Revoke the browser session and all of its signed download links."""
        if self.accounts:
            await anyio.to_thread.run_sync(self.accounts.logout, self._bearer(request))
        return self._json({"status": "signed_out"})

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[redacted]")
            return value
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        return value

    def _json(self, value: Any, *, status_code: int = 200) -> JSONResponse:
        return JSONResponse(
            self._redact(value),
            status_code=status_code,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    def _job_directory(self, job_id: str) -> Path:
        if _JOB_ID.fullmatch(job_id) is None:
            raise HTTPException(404, "Project not found")
        try:
            directory = (self.jobs_dir / job_id).resolve(strict=True)
            if directory.parent != self.jobs_dir or not directory.is_dir():
                raise HTTPException(404, "Project not found")
            return directory
        except (OSError, RuntimeError):
            raise HTTPException(404, "Project not found") from None

    def _artifact(self, job_id: str, relative: str) -> Path:
        directory = self._job_directory(job_id)
        try:
            candidate = directory / relative
            path = candidate.resolve(strict=True)
            if path != candidate or not path.is_relative_to(directory) or not path.is_file():
                raise HTTPException(404, "Project file not found")
            return path
        except (OSError, RuntimeError):
            raise HTTPException(404, "Project file not found") from None

    def _state(self, job_id: str) -> JobState:
        state = read_json(self._artifact(job_id, "state.json"), JobState)
        if state.job_id != job_id:
            raise HTTPException(404, "Project not found")
        return state

    def _detail(self, job_id: str) -> dict[str, Any]:
        state = self._state(job_id)
        selection: SelectionDoc | None = None
        try:
            selection = read_json(self._artifact(job_id, "selected.json"), SelectionDoc)
        except HTTPException:
            pass
        return WebJobDetailDoc(
            state=state,
            selection=selection,
            running=self.runner.is_running(job_id),
            stopping=self.runner.is_stopping(job_id),
            error=self.runner.error(job_id),
        ).model_dump(mode="json")

    async def health(self, request: Request) -> Response:
        """Report public readiness and upload/runtime limits without backend credentials."""
        return self._json(
            WebHealthDoc(
                max_upload_bytes=self._upload_limit,
                accounts_enabled=self.settings.accounts_enabled,
                default_target_minutes=min(1440.0, max(0.5, self.pipeline_settings.target_minutes)),
            ).model_dump(mode="json")
        )

    async def jobs(self, request: Request) -> Response:
        """List the shared workspace's valid checkpoints after authentication."""
        self._require_auth(request)

        def read_states() -> list[dict[str, Any]]:
            """Read valid checkpoints off the event loop and skip unavailable entries."""
            states: list[JobState] = []
            for path in self.jobs_dir.glob("*/state.json"):
                if request.state.user_id is not None and (
                    not self.accounts
                    or not self.accounts.owns(path.parent.name, request.state.user_id)
                ):
                    continue
                try:
                    states.append(self._state(path.parent.name))
                except (HTTPException, ValidationError, OSError):
                    logging.getLogger(__name__).warning("Skipped an unreadable project checkpoint")
            return [
                state.model_dump(mode="json")
                for state in sorted(states, key=lambda state: state.created_at, reverse=True)
            ]

        return self._json(await anyio.to_thread.run_sync(read_states))

    async def detail(self, request: Request) -> Response:
        """Return the latest durable state, selection, and worker status for one project."""
        self._require_auth(request)
        result = await anyio.to_thread.run_sync(self._detail, request.path_params["job_id"])
        return self._json(result)

    @property
    def _upload_limit(self) -> int:
        return self.settings.max_upload_bytes or MAX_UPLOAD_BYTES

    @property
    def _upload_limit_message(self) -> str:
        gib = self._upload_limit / 1024**3
        return f"Upload exceeds the {gib:g} GB limit"

    async def _upload_stream(self, request: Request) -> AsyncIterator[bytes]:
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > self._upload_limit + _MULTIPART_OVERHEAD:
                # The multipart parser closes its temporary files for this exception.
                raise MultiPartException(self._upload_limit_message)
            yield chunk

    def _check_upload(self, upload: UploadFile) -> None:
        file: BinaryIO = upload.file
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        if size > self._upload_limit:
            raise HTTPException(413, self._upload_limit_message)
        if size == 0 or not is_zipfile(file):
            raise HTTPException(400, "Choose a valid ZIP archive")
        with ZipFile(file) as archive:
            expanded_bytes = sum(member.file_size for member in archive.infolist())
        # The upload is already spooled; leave room for its durable copy,
        # extracted episodes, rendered clips and the final film.
        required_bytes = size + expanded_bytes + 10 * 1024**3
        storage_path = next(
            path for path in (self.jobs_dir, *self.jobs_dir.parents) if path.exists()
        )
        if required_bytes > shutil.disk_usage(storage_path).free:
            raise HTTPException(507, "Not enough server storage for this archive and its film")
        file.seek(0)

    async def create(self, request: Request) -> Response:
        """Spool a bounded ZIP upload to disk and launch its resumable background job."""
        self._require_auth(request)  # Authenticate before reading any multipart content.
        length = request.headers.get("content-length")
        if length is not None:
            try:
                if int(length) < 0:
                    raise ValueError
                if int(length) > self._upload_limit + _MULTIPART_OVERHEAD:
                    raise HTTPException(413, self._upload_limit_message)
            except ValueError:
                raise HTTPException(400, "Invalid upload length") from None
        async with self._mutation_lock:
            if self.runner.busy:
                raise HTTPException(
                    409, "Another edit is running; stop it before creating a project"
                )
            parser = MultiPartParser(
                request.headers,
                self._upload_stream(request),
                max_files=1,
                max_fields=2,
                max_part_size=16 * 1024,
            )
            try:
                form = await parser.parse()
            except MultiPartException as exc:
                status = 413 if str(exc) == self._upload_limit_message else 400
                raise HTTPException(status, str(exc)) from None
            try:
                fields = form.multi_items()
                if any(
                    key not in {"archive", "target_minutes", "name"} for key, _ in fields
                ) or len({key for key, _ in fields}) != len(fields):
                    raise HTTPException(422, "Use archive, target_minutes, and name only")
                upload = form.get("archive")
                if not isinstance(upload, UploadFile) or not (
                    upload.filename or ""
                ).lower().endswith(".zip"):
                    raise HTTPException(400, "Choose a ZIP archive")
                try:
                    minutes = float(
                        str(
                            form.get(
                                "target_minutes",
                                min(1440.0, max(0.5, self.pipeline_settings.target_minutes)),
                            )
                        )
                    )
                except ValueError:
                    raise HTTPException(422, "Choose a valid target runtime") from None
                if not math.isfinite(minutes) or not 0.5 <= minutes <= 1440:
                    raise HTTPException(422, "Target runtime must be between 0.5 and 1440 minutes")
                name = form.get("name")
                if name is not None and (not isinstance(name, str) or len(name) > 120):
                    raise HTTPException(422, "Project name must contain at most 120 characters")
                await anyio.to_thread.run_sync(self._check_upload, upload)
                state = await anyio.to_thread.run_sync(
                    partial(
                        self.runner.create,
                        upload.file,
                        minutes,
                        name=name.strip() or None if isinstance(name, str) else None,
                        cfg=self.pipeline_settings,
                    )
                )
                if self.accounts and request.state.user_id is not None:
                    await anyio.to_thread.run_sync(
                        self.accounts.assign, state.job_id, request.state.user_id
                    )
                await anyio.to_thread.run_sync(self.runner.start, state.job_id)
                return self._json(
                    await anyio.to_thread.run_sync(self._detail, state.job_id), status_code=201
                )
            except BadZipFile:
                raise HTTPException(400, "Choose a valid ZIP archive") from None
            except RuntimeError:
                raise HTTPException(
                    409, "Another edit is running; try again when it stops"
                ) from None
            finally:
                await form.close()

    async def resume(self, request: Request) -> Response:
        """Resume an idle saved edit from its completed checkpoints."""
        self._require_auth(request)
        job_id = request.path_params["job_id"]
        async with self._mutation_lock:
            await anyio.to_thread.run_sync(self._state, job_id)
            try:
                await anyio.to_thread.run_sync(self.runner.start, job_id)
            except RuntimeError:
                raise HTTPException(
                    409, "Another edit is running; stop it before resuming"
                ) from None
            return self._json(await anyio.to_thread.run_sync(self._detail, job_id))

    async def stop(self, request: Request) -> Response:
        """Request a checkpoint-preserving stop without waiting for processing to finish."""
        self._require_auth(request)
        job_id = request.path_params["job_id"]
        async with self._mutation_lock:
            await anyio.to_thread.run_sync(self._state, job_id)
            await anyio.to_thread.run_sync(self.runner.request_stop, job_id)
            return self._json(await anyio.to_thread.run_sync(self._detail, job_id))

    async def refine(self, request: Request) -> Response:
        """Create a new story edit with protected ranges while retaining the earlier film."""
        self._require_auth(request)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 64 * 1024:
                raise HTTPException(413, "The list of protected moments is too large")
        try:
            refinements = WebRefineRequestDoc.model_validate_json(bytes(body))
        except ValidationError:
            raise HTTPException(
                422, "Provide valid episode, start, and end values for each moment"
            ) from None
        job_id = request.path_params["job_id"]
        async with self._mutation_lock:
            await anyio.to_thread.run_sync(self._state, job_id)
            try:
                state = await anyio.to_thread.run_sync(
                    partial(
                        self.runner.refine, job_id, must_keep_ranges=refinements.must_keep_ranges
                    )
                )
                if self.accounts and request.state.user_id is not None:
                    await anyio.to_thread.run_sync(
                        self.accounts.assign, state.job_id, request.state.user_id
                    )
                await anyio.to_thread.run_sync(self.runner.start, state.job_id)
            except RuntimeError:
                raise HTTPException(
                    409, "Another edit is running; stop it before refining"
                ) from None
            except ValueError:
                raise HTTPException(
                    422, "Protected moments must be inside saved episodes"
                ) from None
            return self._json(
                await anyio.to_thread.run_sync(self._detail, state.job_id), status_code=201
            )

    def _ticket(
        self, job_id: str, filename: str, expires: int, session_digest: str | None = None
    ) -> str:
        payload = json.dumps(
            {
                "job_id": job_id,
                "filename": filename,
                "expires": expires,
                "nonce": secrets.token_urlsafe(16),
                **({"session_digest": session_digest} if session_digest else {}),
            },
            separators=(",", ":"),
        ).encode()
        encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        signature = hmac.new(self.token.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def _valid_ticket(self, ticket: str, job_id: str, filename: str) -> bool:
        if len(ticket) > 2048:
            return False
        encoded, separator, signature = ticket.partition(".")
        expected = hmac.new(self.token.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not separator or not secrets.compare_digest(signature.encode(), expected.encode()):
            return False
        try:
            data = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
            return (
                isinstance(data, dict)
                and set(data)
                in (
                    {"job_id", "filename", "expires", "nonce"},
                    {"job_id", "filename", "expires", "nonce", "session_digest"},
                )
                and self._ticket_owner_valid(data, job_id)
                and data["job_id"] == job_id
                and data["filename"] == filename
                and type(data["expires"]) is int
                and data["expires"] > int(time.time())
                and isinstance(data["nonce"], str)
            )
        except (ValueError, UnicodeError):
            return False

    def _ticket_owner_valid(self, data: dict[str, Any], job_id: str) -> bool:
        digest = data.get("session_digest")
        if digest is None:
            return True  # Existing private operator tickets retain their bounded lifetime.
        if not self.accounts or not isinstance(digest, str):
            return False
        user = self.accounts.resolve_digest(digest)
        return user is not None and self.accounts.owns(job_id, user.id)

    def _ready_outputs(self, job_id: str) -> None:
        if self._state(job_id).stages.render.status != "done":
            raise HTTPException(409, "The film is still processing")
        self._output(job_id, _MOVIE)
        self._output(job_id, _SUBTITLES)

    def _output(self, job_id: str, filename: str) -> Path:
        if filename not in {_MOVIE, _SUBTITLES}:
            raise HTTPException(404, "Film output not found")
        path = self._artifact(job_id, f"output/{filename}")
        output_directory = (self._job_directory(job_id) / "output").resolve()
        if path.parent != output_directory or path.name != filename:
            raise HTTPException(404, "Film output not found")
        return path

    async def media_ticket(self, request: Request) -> Response:
        """Issue short-lived, job-and-file-scoped URLs for browser video and downloads."""
        self._require_auth(request)
        job_id = request.path_params["job_id"]
        await anyio.to_thread.run_sync(self._ready_outputs, job_id)
        base = self.settings.public_url or str(request.base_url).rstrip("/")
        expires = int(time.time()) + MEDIA_TICKET_SECONDS
        urls = {
            f"{kind}_url": f"{base}/api/jobs/{job_id}/{route}?"
            + urlencode(
                {
                    "ticket": self._ticket(
                        job_id,
                        filename,
                        expires,
                        self.accounts.session_digest(self._bearer(request))
                        if self.accounts and request.state.user_id is not None
                        else None,
                    )
                }
            )
            for kind, route, filename in (
                ("movie", "movie", _MOVIE),
                ("subtitles", "subtitles", _SUBTITLES),
            )
        }
        return self._json(
            WebMediaTicketDoc(**urls, expires_in=MEDIA_TICKET_SECONDS).model_dump(mode="json")
        )

    async def media(self, request: Request) -> Response:
        """Serve a confined export with bounded range support for playback and seeking."""
        job_id = request.path_params["job_id"]
        filename = _MOVIE if request.url.path.endswith("/movie") else _SUBTITLES
        authorized = self._authorized(request)
        if authorized:
            self._require_auth(request)
        if not authorized and not self._valid_ticket(
            request.query_params.get("ticket", ""), job_id, filename
        ):
            raise HTTPException(401, "This film link expired; reconnect to the workspace")
        await anyio.to_thread.run_sync(self._ready_outputs, job_id)
        path = await anyio.to_thread.run_sync(self._output, job_id, filename)
        return FileResponse(
            path,
            media_type="video/mp4" if filename == _MOVIE else "application/x-subrip",
            filename=filename,
            content_disposition_type=(
                "attachment" if request.query_params.get("download") == "1" else "inline"
            ),
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )


def create_web_app(
    settings: WebSettings | None = None, runner: JobRunner | None = None
) -> Starlette:
    """Build the persistent API without starting a server or loading any trained models."""
    configuration = settings if settings is not None else WebSettings()
    job_runner = runner if runner is not None else JobRunner(configuration.jobs_dir)
    if job_runner.jobs_dir.resolve() != configuration.jobs_dir.resolve():
        raise ValueError("The API and job runner must share the same jobs directory")
    api = _WebAPI(configuration, job_runner)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Stop the worker at its next checkpoint before releasing server resources."""
        try:
            yield
        finally:
            active = job_runner.active_job_id
            if active is not None:
                await anyio.to_thread.run_sync(job_runner.request_stop, active)
            await anyio.to_thread.run_sync(job_runner.shutdown)

    async def handle_error(request: Request, exc: Exception) -> Response:
        """Return safe JSON diagnostics without publishing exception bodies or credentials."""
        if isinstance(exc, HTTPException):
            response = api._json({"error": exc.detail}, status_code=exc.status_code)
            if exc.headers:
                response.headers.update(exc.headers)
            return response
        logging.getLogger(__name__).error("A web request failed; saved project data is unchanged")
        return api._json(
            {"error": "The workspace could not complete this request"}, status_code=500
        )

    routes = [
        Route("/api/health", api.health, methods=["GET"]),
        Route("/api/auth/register", api.account_auth, methods=["POST"]),
        Route("/api/auth/login", api.account_auth, methods=["POST"]),
        Route("/api/auth/me", api.account_me, methods=["GET"]),
        Route("/api/auth/logout", api.account_logout, methods=["POST"]),
        Route("/api/jobs", api.jobs, methods=["GET"]),
        Route("/api/jobs", api.create, methods=["POST"]),
        Route("/api/jobs/{job_id}", api.detail, methods=["GET"]),
        Route("/api/jobs/{job_id}/resume", api.resume, methods=["POST"]),
        Route("/api/jobs/{job_id}/stop", api.stop, methods=["POST"]),
        Route("/api/jobs/{job_id}/refine", api.refine, methods=["POST"]),
        Route("/api/jobs/{job_id}/media-ticket", api.media_ticket, methods=["POST"]),
        Route("/api/jobs/{job_id}/movie", api.media, methods=["GET", "HEAD"]),
        Route("/api/jobs/{job_id}/subtitles", api.media, methods=["GET", "HEAD"]),
    ]
    return Starlette(
        routes=routes,
        lifespan=lifespan,
        exception_handlers={HTTPException: handle_error, Exception: handle_error},
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=configuration.allowed_origins,
                allow_methods=["GET", "HEAD", "POST", "OPTIONS"],
                allow_headers=["Authorization", "Content-Type", "Range"],
                expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"],
                allow_credentials=False,
            )
        ],
    )


def main() -> None:
    """Launch the persistent service with rich diagnostics and no secret-bearing access logs."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), markup=False)],
    )
    try:
        settings = WebSettings()
    except ValidationError:
        logging.getLogger(__name__).error(
            "Set STS_WEB_ACCESS_TOKEN and STS_WEB_ALLOWED_ORIGINS in .env before starting the API"
        )
        raise SystemExit(2) from None
    uvicorn.run(
        create_web_app(settings),
        host=settings.host,
        port=settings.port,
        access_log=False,
        log_config=None,
        forwarded_allow_ips=settings.trusted_proxy_ips,
    )


if __name__ == "__main__":
    main()
