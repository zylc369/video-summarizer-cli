"""FFmpeg subprocess wrapper with error handling."""

import logging
import shutil
import subprocess
from typing import Final

logger = logging.getLogger(__name__)

from src.context import FFmpegError, FFmpegNotFoundError


FFMPEG_INSTALL_URL: Final = "https://ffmpeg.org"


def check_ffmpeg_available() -> None:
    """
    Check if FFmpeg is installed and available in PATH.

    Raises:
        FFmpegNotFoundError: If FFmpeg binary not found in system PATH.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FFmpegNotFoundError(
            f"FFmpeg not found. Install from {FFMPEG_INSTALL_URL}"
        )


def run_ffmpeg(args: list[str], timeout: int | None = None) -> str:
    """
    Run FFmpeg with the given arguments.

    Args:
        args: List of FFmpeg command-line arguments (excluding 'ffmpeg' itself).
        timeout: Optional timeout in seconds. If None, waits indefinitely.

    Returns:
        Empty string (FFmpeg stdout is not consumed by callers).

    Raises:
        FFmpegNotFoundError: If FFmpeg binary not found in system PATH.
        FFmpegError: If FFmpeg exits with a non-zero return code or times out.
    """
    check_ffmpeg_available()

    try:
        result = subprocess.run(
            ["ffmpeg", *args],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"FFmpeg command timed out after {timeout} seconds") from e

    if result.returncode != 0:
        try:
            stderr_text = result.stderr.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode FFmpeg stderr: {e}", exc_info=True)
            stderr_text = f"<undecodable stderr, {len(result.stderr)} bytes>"
        raise FFmpegError(
            f"FFmpeg command failed with exit code {result.returncode}: {stderr_text}"
        )

    return ""
