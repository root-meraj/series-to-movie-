"""Generate a 90-second synthetic episode and a ZIP with tricky episode names."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile

from rich.logging import RichHandler


def make_fixture(output_dir: Path) -> Path:
    """Create ep_test.mp4 and show.zip without downloading copyrighted media.

    The archive deliberately stores episode 10 before episodes 2 and 1. The MKV
    entry is remuxed in full, without cutting, so its container matches its suffix.
    All three archive entries contain the same synthetic 90-second test pattern.
    """
    logger = logging.getLogger(__name__)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is required. Install with winget/choco (Windows), "
            "apt (Linux), or brew (macOS), then add it to PATH."
        )
    if shutil.which("ffprobe") is None:
        raise RuntimeError("FFprobe is required; install the full FFmpeg distribution.")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Fixture generation started: %s", output_dir)
    with TemporaryDirectory(prefix="fixture-", dir=output_dir) as temporary_dir:
        temporary_path = Path(temporary_dir)
        video_path = temporary_path / "ep_test.mp4"
        mkv_path = temporary_path / "show_1x02.mkv"
        common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        subprocess.run(
            [
                *common,
                "-f",
                "lavfi",
                "-i",
                "testsrc2=duration=90:size=640x360:rate=30",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=90",
                "-shortest",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            check=True,
        )
        logger.info("Fixture unit-progress: generated 90-second MP4 with audio")
        # Full-file container conversion only: no stream-copy cuts are performed.
        subprocess.run(
            [*common, "-i", str(video_path), "-map", "0", "-c", "copy", str(mkv_path)],
            check=True,
        )
        zip_path = temporary_path / "show.zip"
        with ZipFile(zip_path, "w", compression=ZIP_STORED) as archive:
            archive.write(video_path, arcname="ep10.mp4")
            archive.write(mkv_path, arcname="show_1x02.mkv")
            archive.write(video_path, arcname="show.S01E01.mp4")
        logger.info("Fixture unit-progress: archived three deliberately unordered episodes")
        video_path.replace(output_dir / "ep_test.mp4")
        zip_path.replace(output_dir / "show.zip")
    logger.info("Fixture generation done: %s", output_dir / "show.zip")
    return output_dir / "show.zip"


def main() -> int:
    """Run the synthetic fixture generator from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures",
        help="Destination directory (default: project tests/fixtures).",
    )
    args = parser.parse_args()
    try:
        make_fixture(args.output_dir)
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        logging.getLogger(__name__).exception("Fixture generation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
