"""Tests for FFmpeg subprocess wrapper."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from src.utils.ffmpeg import FFmpegError, FFmpegNotFoundError, check_ffmpeg_available, run_ffmpeg


class TestCheckFfmpegAvailable:
    """Tests for check_ffmpeg_available function."""

    def test_check_ffmpeg_available_success(self) -> None:
        """When FFmpeg is found, no exception is raised."""
        with patch("src.utils.ffmpeg.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ffmpeg"

            # Should not raise any exception
            check_ffmpeg_available()

            mock_which.assert_called_once_with("ffmpeg")

    def test_check_ffmpeg_available_not_found(self) -> None:
        """When FFmpeg is not found, raises FFmpegNotFoundError."""
        with patch("src.utils.ffmpeg.shutil.which") as mock_which:
            mock_which.return_value = None

            with pytest.raises(FFmpegNotFoundError) as exc_info:
                check_ffmpeg_available()

            assert "FFmpeg not found" in str(exc_info.value)
            assert "ffmpeg.org" in str(exc_info.value)
            mock_which.assert_called_once_with("ffmpeg")


class TestRunFfmpeg:
    """Tests for run_ffmpeg function."""

    def test_run_ffmpeg_success(self) -> None:
        """When FFmpeg succeeds, returns empty string (stdout not consumed)."""
        with patch("src.utils.ffmpeg.check_ffmpeg_available"), patch(
            "src.utils.ffmpeg.subprocess.run"
        ) as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = b"FFmpeg output"
            mock_result.stderr = b""
            mock_run.return_value = mock_result

            args = ["-i", "input.mp4", "output.wav"]
            result = run_ffmpeg(args)

            assert result == ""
            mock_run.assert_called_once_with(
                ["ffmpeg", "-i", "input.mp4", "output.wav"],
                capture_output=True,
                timeout=None,
                check=False,
            )

    def test_run_ffmpeg_failure(self) -> None:
        """When FFmpeg fails with non-zero exit, raises FFmpegError."""
        with patch("src.utils.ffmpeg.check_ffmpeg_available"), patch(
            "src.utils.ffmpeg.subprocess.run"
        ) as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = b""
            mock_result.stderr = b"Invalid data found when processing input"
            mock_run.return_value = mock_result

            with pytest.raises(FFmpegError) as exc_info:
                run_ffmpeg(["-i", "input.mp4"])

            assert "exit code 1" in str(exc_info.value)
            assert "Invalid data found" in str(exc_info.value)

    def test_run_ffmpeg_timeout(self) -> None:
        """When FFmpeg times out, raises FFmpegError."""
        with patch("src.utils.ffmpeg.check_ffmpeg_available"), patch(
            "src.utils.ffmpeg.subprocess.run"
        ) as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 30)

            with pytest.raises(FFmpegError) as exc_info:
                run_ffmpeg(["-i", "input.mp4"], timeout=30)

            assert "timed out after 30 seconds" in str(exc_info.value)

    def test_run_ffmpeg_passes_args(self) -> None:
        """Verify args are passed correctly to subprocess.run."""
        with patch("src.utils.ffmpeg.check_ffmpeg_available"), patch(
            "src.utils.ffmpeg.subprocess.run"
        ) as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = b"output"
            mock_result.stderr = b""
            mock_run.return_value = mock_result

            args = ["-i", "video.mp4", "-vn", "-acodec", "pcm_s16le", "audio.wav"]
            run_ffmpeg(args)

            call_args = mock_run.call_args[0][0]
            assert call_args == ["ffmpeg", "-i", "video.mp4", "-vn", "-acodec", "pcm_s16le", "audio.wav"]

    def test_run_ffmpeg_with_timeout(self) -> None:
        """Verify timeout is passed to subprocess.run."""
        with patch("src.utils.ffmpeg.check_ffmpeg_available"), patch(
            "src.utils.ffmpeg.subprocess.run"
        ) as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = b"output"
            mock_result.stderr = b""
            mock_run.return_value = mock_result

            run_ffmpeg(["-i", "input.mp4"], timeout=60)

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["timeout"] == 60
