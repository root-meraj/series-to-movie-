"""Exercise real local HTTP media requests without reading a whole movie at once."""

from __future__ import annotations

import io
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from sts.app.media_server import LocalMediaServer, start_media_server


@pytest.fixture
def media(
    tmp_path: Path,
) -> Iterator[tuple[LocalMediaServer, Path, bytes]]:
    """Start an isolated loopback resource and always release its socket afterward."""
    root = tmp_path / "jobs"
    output = root / "job_a" / "output"
    output.mkdir(parents=True)
    payload = bytes(range(256)) * 3073
    (output / "condensed_movie.mp4").write_bytes(payload)
    (output / "condensed.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello.\n", encoding="utf-8"
    )
    server = start_media_server(root)
    try:
        yield server, root, payload
    finally:
        server.close()


def _request(
    url: str, *, method: str = "GET", headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    request = Request(url, method=method, headers=headers or {})
    try:
        response = urlopen(request, timeout=3)
    except HTTPError as error:
        response = error
    with response:
        return (
            response.status,
            {key.lower(): value for key, value in response.headers.items()},
            response.read(),
        )


def test_full_response_and_explicit_movie_headers(
    media: tuple[LocalMediaServer, Path, bytes],
) -> None:
    """A preview streams the exact movie with range metadata and no broad CORS access."""
    server, _, payload = media
    url = server.url("job_a", "condensed_movie.mp4")
    assert urlsplit(url).hostname == "127.0.0.1"
    status, headers, body = _request(url)
    assert status == 200
    assert body == payload
    assert headers["content-type"] == "video/mp4"
    assert int(headers["content-length"]) == len(payload)
    assert headers["accept-ranges"] == "bytes"
    assert headers["last-modified"]
    assert headers["etag"]
    assert headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in headers
    assert "content-disposition" not in headers


@pytest.mark.parametrize(
    ("range_header", "start", "end"),
    [
        ("bytes=2-8", 2, 8),
        ("bytes=786680-", 786680, 786687),
        ("bytes=-7", 786681, 786687),
        ("bytes=786680-9999999", 786680, 786687),
        ("bytes=-9999999", 0, 786687),
        ("bytes=0-0", 0, 0),
    ],
)
def test_single_ranges_support_seeking(
    media: tuple[LocalMediaServer, Path, bytes], range_header: str, start: int, end: int
) -> None:
    """Closed, open, suffix, and clipped ranges return only the requested bytes."""
    server, _, payload = media
    status, headers, body = _request(
        server.url("job_a", "condensed_movie.mp4"), headers={"Range": range_header}
    )
    assert status == 206
    assert body == payload[start : end + 1]
    assert headers["content-range"] == f"bytes {start}-{end}/{len(payload)}"
    assert int(headers["content-length"]) == end - start + 1


@pytest.mark.parametrize(
    "range_header",
    ["bytes=3-2", "bytes=9999999-", "bytes=-0", "bytes=-", "bytes=0-2,4-6", "items=0-2"],
)
def test_invalid_or_multiple_ranges_return_416(
    media: tuple[LocalMediaServer, Path, bytes], range_header: str
) -> None:
    """Malformed and unsatisfiable requests never silently become full downloads."""
    server, _, payload = media
    status, headers, body = _request(
        server.url("job_a", "condensed_movie.mp4"), headers={"Range": range_header}
    )
    assert status == 416
    assert headers["content-range"] == f"bytes */{len(payload)}"
    assert headers["content-length"] == "0"
    assert body == b""


def test_empty_output_has_no_satisfiable_range(
    media: tuple[LocalMediaServer, Path, bytes],
) -> None:
    """An empty output supports HEAD/GET but cannot satisfy byte-range requests."""
    server, root, _ = media
    (root / "job_a" / "output" / "condensed_movie.mp4").write_bytes(b"")
    url = server.url("job_a", "condensed_movie.mp4")
    status, headers, body = _request(url)
    assert status == 200
    assert headers["content-length"] == "0"
    assert body == b""
    assert _request(url, method="HEAD")[1]["content-length"] == "0"
    status, headers, body = _request(url, headers={"Range": "bytes=0-"})
    assert status == 416
    assert headers["content-range"] == "bytes */0"
    assert body == b""


class _TrackingReader:
    def __init__(self, stream: BinaryIO, reads: list[int]) -> None:
        self.stream = stream
        self.reads = reads

    def __enter__(self) -> _TrackingReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stream.close()

    def fileno(self) -> int:
        return self.stream.fileno()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.stream.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        self.reads.append(size)
        return self.stream.read(size)


def test_full_download_reads_bounded_chunks_and_head_reads_nothing(
    media: tuple[LocalMediaServer, Path, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual disk reads stay bounded; metadata/short seeks never read unrelated bytes."""
    server, root, payload = media
    movie = root / "job_a" / "output" / "condensed_movie.mp4"
    reads: list[int] = []
    original_open = Path.open

    def tracked_open(path: Path, *args: object, **kwargs: object) -> BinaryIO:
        stream = cast(BinaryIO, original_open(path, *args, **kwargs))
        if path == movie:
            return cast(BinaryIO, _TrackingReader(stream, reads))
        return stream

    monkeypatch.setattr(Path, "open", tracked_open)
    url = server.url("job_a", "condensed_movie.mp4")
    status, headers, body = _request(url, method="HEAD")
    assert status == 200
    assert body == b""
    assert int(headers["content-length"]) == len(payload)
    assert reads == []
    status, headers, body = _request(url, method="HEAD", headers={"Range": "bytes=9-14"})
    assert status == 206
    assert headers["content-length"] == "6"
    assert body == b""
    assert reads == []
    assert _request(url, headers={"Range": "bytes=9-14"})[2] == payload[9:15]
    assert reads == [6]
    reads.clear()
    assert _request(url)[2] == payload
    assert len(reads) > 1
    assert all(0 < size <= 256 * 1024 for size in reads)
    assert sum(reads) == len(payload)


def test_download_and_subtitle_content_type(
    media: tuple[LocalMediaServer, Path, bytes],
) -> None:
    """Direct download links keep the expected filename and correct subtitle MIME."""
    server, root, _ = media
    status, headers, body = _request(server.url("job_a", "condensed.srt", download=True))
    assert status == 200
    assert headers["content-type"] == "application/x-subrip; charset=utf-8"
    assert headers["content-disposition"] == 'attachment; filename="condensed.srt"'
    assert body == (root / "job_a" / "output" / "condensed.srt").read_bytes()
    _, headers, _ = _request(
        server.url("job_a", "condensed_movie.mp4", download=True), method="HEAD"
    )
    assert headers["content-disposition"] == 'attachment; filename="condensed_movie.mp4"'


def test_etag_revalidation_and_if_range(
    media: tuple[LocalMediaServer, Path, bytes],
) -> None:
    """Unchanged output revalidation avoids file reads, and stale ranges return a full file."""
    server, _, payload = media
    url = server.url("job_a", "condensed_movie.mp4")
    etag = _request(url, method="HEAD")[1]["etag"]
    status, headers, body = _request(url, headers={"If-None-Match": etag})
    assert status == 304
    assert headers["etag"] == etag
    assert body == b""
    assert _request(url, headers={"Range": "bytes=3-5", "If-Range": etag})[2] == payload[3:6]
    status, _, body = _request(url, headers={"Range": "bytes=3-5", "If-Range": '"stale"'})
    assert status == 200
    assert body == payload


@pytest.mark.parametrize("job_id", ["../job_a", "", "job/a", "job\\a", ".", "..", "a" * 65])
def test_url_rejects_unsafe_job_identifiers(
    media: tuple[LocalMediaServer, Path, bytes], job_id: str
) -> None:
    """The public URL builder cannot encode arbitrary filesystem paths."""
    server, _, _ = media
    with pytest.raises(ValueError, match="job identifier"):
        server.url(job_id, "condensed_movie.mp4")


def test_missing_outputs_and_direct_traversal_are_not_exposed(
    media: tuple[LocalMediaServer, Path, bytes],
) -> None:
    """HTTP requests independently validate IDs, tokens, and allowed output names."""
    server, root, _ = media
    (root / "job_a" / "state.json").write_text('"private"', encoding="utf-8")
    url = server.url("job_a", "condensed_movie.mp4")
    for request_url in (
        server.url("missing", "condensed_movie.mp4"),
        url.replace("/job_a/", "/%2e%2e/"),
        url.replace("/job_a/", "/job_a%2f..%2f/"),
        url.replace("condensed_movie.mp4", "../state.json"),
        url.replace("condensed_movie.mp4", "state.json"),
        url.replace("/media/", "/unknown/"),
        url.replace(urlsplit(url).path.split("/")[2], "wrong-token"),
    ):
        assert _request(request_url)[0] == 404
    with pytest.raises(ValueError, match="outputs"):
        server.url("job_a", "state.json")


def test_output_symlink_outside_job_is_refused(
    media: tuple[LocalMediaServer, Path, bytes],
    tmp_path: Path,
) -> None:
    """Even an allowed output filename cannot resolve to an external local file."""
    server, root, _ = media
    external = tmp_path / "outside" / "condensed_movie.mp4"
    external.parent.mkdir()
    external.write_bytes(b"private external bytes")
    movie = root / "job_a" / "output" / "condensed_movie.mp4"
    movie.unlink()
    try:
        movie.symlink_to(external)
    except OSError as error:
        pytest.skip(f"Platform does not permit test symlinks: {error}")
    assert _request(server.url("job_a", "condensed_movie.mp4"))[0] == 404


def test_keep_alive_requests_and_shutdown(
    media: tuple[LocalMediaServer, Path, bytes],
) -> None:
    """Seeking can reuse one HTTP connection, and shutdown is safe to call repeatedly."""
    server, _, payload = media
    url = urlsplit(server.url("job_a", "condensed_movie.mp4"))
    connection = HTTPConnection(url.hostname, url.port, timeout=3)
    try:
        for range_header in ("bytes=1-4", "bytes=-4"):
            connection.request("GET", url.path, headers={"Range": range_header})
            response = connection.getresponse()
            assert response.status == 206
            assert response.read() == (
                payload[1:5] if range_header == "bytes=1-4" else payload[-4:]
            )
    finally:
        connection.close()
    server.close()
    server.close()
    with pytest.raises(RuntimeError, match="closed"):
        server.url("job_a", "condensed_movie.mp4")
    with pytest.raises(URLError):
        _request(url.geturl())
