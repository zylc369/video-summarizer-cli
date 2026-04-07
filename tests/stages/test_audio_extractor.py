"""Tests for audio extraction stage."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.context import FFmpegError, RuntimeContext
from src.stages.audio_extractor import extract_audio


@pytest.fixture
def mock_context(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        env="mac",
        os_name="darwin",
        has_cuda=False,
        has_mps=True,
        gpu_name=None,
        gpu_vram_mb=None,
        cpu_count=10,
        total_ram_gb=16.0,
        config={},
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    )


def test_extract_audio_creates_output_dir(mock_context: RuntimeContext, tmp_path: Path):
    """Verify output directory is created based on video stem."""
    video_path = tmp_path / "test_video.mp4"
    _ = video_path.touch()

    with patch("src.stages.audio_extractor.run_ffmpeg"):
        _ = extract_audio(mock_context, video_path)

    expected_dir = mock_context.output_dir / "test_video"
    assert expected_dir.exists()
    assert expected_dir.is_dir()


def test_extract_audio_calls_ffmpeg_with_correct_args(mock_context: RuntimeContext, tmp_path: Path):
    """Verify run_ffmpeg called with correct FFmpeg args for 16kHz mono WAV."""
    video_path = tmp_path / "sample.mp4"
    _ = video_path.touch()

    expected_args = [
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(mock_context.output_dir / "sample" / "audio.wav"),
    ]

    with patch("src.stages.audio_extractor.run_ffmpeg") as mock_run:
        _ = extract_audio(mock_context, video_path)
        mock_run.assert_called_once_with(expected_args)


def test_extract_audio_returns_correct_path(mock_context: RuntimeContext, tmp_path: Path):
    """Verify returns audio.wav path in output dir."""
    video_path = tmp_path / "my_video.mp4"
    _ = video_path.touch()

    expected_path = mock_context.output_dir / "my_video" / "audio.wav"

    with patch("src.stages.audio_extractor.run_ffmpeg"):
        result = extract_audio(mock_context, video_path)

    assert result == expected_path


def test_extract_audio_uses_video_stem_for_dir_name(mock_context: RuntimeContext, tmp_path: Path):
    """Verify directory name matches video stem (removes extension)."""
    video_path = tmp_path / "presentation_recording_2024.mp4"
    _ = video_path.touch()

    expected_dir = mock_context.output_dir / "presentation_recording_2024"

    with patch("src.stages.audio_extractor.run_ffmpeg"):
        _ = extract_audio(mock_context, video_path)

    assert expected_dir.exists()


def test_extract_audio_raises_on_ffmpeg_failure(mock_context: RuntimeContext, tmp_path: Path):
    """Verify FFmpegError is propagated when run_ffmpeg raises."""
    video_path = tmp_path / "corrupt.mp4"
    _ = video_path.touch()

    with patch(
        "src.stages.audio_extractor.run_ffmpeg",
        side_effect=FFmpegError("FFmpeg failed: invalid data"),
    ):
        with pytest.raises(FFmpegError, match="FFmpeg failed: invalid data"):
            _ = extract_audio(mock_context, video_path)
