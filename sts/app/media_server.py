"""Serve local film outputs without copying whole videos into Streamlit memory."""

from __future__ import annotations

import atexit
import logging
import re
import secrets
import stat
from email.utils import formatdate
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from os import fstat, stat_result
from pathlib import Path
from threading import Lock, Thread
from typing import BinaryIO, cast
from urllib.parse import parse_qs, quote, unquote, urlsplit

_CHUNK_BYTES = 256 * 1024
_FILENAMES = frozenset({"condensed_movie.mp4", "condensed.srt"})
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z")


def _byte_range(value: str, size: int) -> tuple[int, int]:
    match = _RANGE.fullmatch(value.strip())
    if match is None or size == 0:
        raise ValueError("Only one satisfiable byte range is supported")
    first, last = match.groups()
    if not first:
        if not last or int(last) == 0:
            raise ValueError("A suffix range must have positive length")
        return max(0, size - int(last)), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("Requested byte range is outside this file")
    return start, end


class _MediaHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, jobs_dir: Path, token: str) -> None:
        self.jobs_dir = jobs_dir
        self.token = token
        super().__init__(("127.0.0.1", 0), _MediaHandler)


class _MediaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Route request diagnostics through the application's configured logger."""
        logging.getLogger(__name__).debug(format, *args)

    def do_GET(self) -> None:  # noqa: N802 - names required by BaseHTTPRequestHandler
        """Stream the requested output or a single byte range in bounded chunks."""
        self._serve(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        """Return the same output metadata as GET without reading file contents."""
        self._serve(head_only=True)

    def _empty_response(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _output_path(self) -> tuple[Path, bool] | None:
        server = cast(_MediaHTTPServer, self.server)
        try:
            request = urlsplit(self.path)
            parts = [unquote(part, errors="strict") for part in request.path.split("/")]
            if (
                len(parts) != 5
                or parts[:3] != ["", "media", server.token]
                or _JOB_ID.fullmatch(parts[3]) is None
                or parts[4] not in _FILENAMES
            ):
                return None
            job_dir = (server.jobs_dir / parts[3]).resolve(strict=True)
            output_dir = (job_dir / "output").resolve(strict=True)
            path = (output_dir / parts[4]).resolve(strict=True)
            if (
                job_dir.parent != server.jobs_dir
                or not output_dir.is_relative_to(job_dir)
                or path.parent != output_dir
                or path.name != parts[4]
            ):
                return None
            download = parse_qs(request.query).get("download") == ["1"]
            return path, download
        except (OSError, RuntimeError, ValueError):
            return None

    def _serve(self, *, head_only: bool) -> None:
        output = self._output_path()
        if output is None:
            self._empty_response(HTTPStatus.NOT_FOUND)
            return
        path, download = output
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                self._empty_response(HTTPStatus.NOT_FOUND)
                return
            with path.open("rb") as stream:
                file_stat = fstat(stream.fileno())
                if not stat.S_ISREG(file_stat.st_mode):
                    self._empty_response(HTTPStatus.NOT_FOUND)
                    return
                self._serve_open_file(stream, path, file_stat, download, head_only)
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            self._empty_response(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browsers cancel range requests when the viewer seeks or leaves a page.
            self.close_connection = True
        except OSError:
            self.close_connection = True
            logging.getLogger(__name__).debug("Local output request interrupted", exc_info=True)

    def _serve_open_file(
        self,
        stream: BinaryIO,
        path: Path,
        metadata: stat_result,
        download: bool,
        head_only: bool,
    ) -> None:
        size = metadata.st_size
        etag = f'"{metadata.st_mtime_ns:x}-{size:x}"'
        modified = formatdate(metadata.st_mtime, usegmt=True)
        if self.headers.get("If-None-Match") in {etag, "*"}:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", modified)
            self.end_headers()
            return
        range_header = self.headers.get("Range")
        if_range = self.headers.get("If-Range")
        if if_range is not None and if_range not in {etag, modified}:
            range_header = None
        start, end = 0, size - 1
        if range_header is not None:
            try:
                start, end = _byte_range(range_header, size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return
        self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", modified)
        self.send_header("Cache-Control", "private, no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Type",
            "video/mp4"
            if path.name == "condensed_movie.mp4"
            else "application/x-subrip; charset=utf-8",
        )
        if range_header is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        if head_only:
            return
        stream.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = stream.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                self.close_connection = True
                return
            self.wfile.write(chunk)
            remaining -= len(chunk)


class LocalMediaServer:
    """Own a loopback HTTP server whose URLs expose only completed job outputs."""

    def __init__(self, jobs_dir: Path) -> None:
        """Start a daemon server on an available local port for the supplied jobs root."""
        self.jobs_dir = jobs_dir.resolve()
        self._server = _MediaHTTPServer(self.jobs_dir, secrets.token_urlsafe(32))
        self._lock = Lock()
        self._closed = False
        self._thread = Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="sts-media",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    def url(self, job_id: str, filename: str, *, download: bool = False) -> str:
        """Return a private local preview/download URL for an allowed job output."""
        if _JOB_ID.fullmatch(job_id) is None:
            raise ValueError("Invalid job identifier")
        if filename not in _FILENAMES:
            raise ValueError("Only the condensed movie and subtitle outputs are available")
        with self._lock:
            if self._closed:
                raise RuntimeError("The local media server has been closed")
            port = self._server.server_address[1]
            path = f"/media/{self._server.token}/{quote(job_id)}/{filename}"
            return f"http://127.0.0.1:{port}{path}" + ("?download=1" if download else "")

    def close(self) -> None:
        """Stop accepting requests and release the listening socket exactly once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=1)
            atexit.unregister(self.close)


def start_media_server(jobs_dir: Path) -> LocalMediaServer:
    """Start the low-memory loopback resource that the UI caches across reruns."""
    return LocalMediaServer(jobs_dir)
