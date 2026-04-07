"""Tests for keyframe extraction stage."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.context import RuntimeContext
from src.stages.keyframe import extract_keyframes, parse_frame_timestamp


@pytest.fixture
def mock_context(tmp_path: Path) -> RuntimeContext:
    """Create a mock RuntimeContext for testing."""
    return RuntimeContext(
        env="mac",
        os_name="darwin",
        has_cuda=False,
        has_mps=True,
        gpu_name=None,
        gpu_vram_mb=None,
        cpu_count=10,
        total_ram_gb=16.0,
        config={
            "keyframe": {
                "method": "keyframe",
                "format": "png",
                "dedup_threshold": 0.95,
                "max_frames": 500,
            },
        },
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    )


def create_mock_frames(keyframe_dir: Path, count: int) -> list[Path]:
    """Create mock PNG files for testing.

    Args:
        keyframe_dir: Directory to create frames in
        count: Number of frames to create

    Returns:
        Sorted list of created frame paths
    """
    frames = []
    for i in range(count):
        path = keyframe_dir / f"key_{i:06d}.png"
        path.write_bytes(b"")
        _ = frames.append(path)
    return sorted(frames)


def test_extract_keyframes_creates_output_dir(
    mock_context: RuntimeContext, tmp_path: Path
) -> None:
    video_path = tmp_path / "test.mp4"
    _ = video_path.write_bytes(b"fake video")

    with patch("src.stages.keyframe.run_ffmpeg"), patch(
        "src.stages.keyframe.deduplicate_frames", return_value=[]
    ):
        result = extract_keyframes(mock_context, video_path)

    expected_dir = mock_context.output_dir / "test" / "keyframes"
    assert expected_dir.exists()
    assert expected_dir.is_dir()
    assert result == []


def test_extract_keyframes_calls_ffmpeg_correctly(
    mock_context: RuntimeContext, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    video_path = tmp_path / "test.mp4"
    _ = video_path.write_bytes(b"fake video")

    with caplog.at_level(logging.INFO):
        with patch("src.stages.keyframe.run_ffmpeg") as mock_ffmpeg, patch(
            "src.stages.keyframe.deduplicate_frames", return_value=[]
        ):
            extract_keyframes(mock_context, video_path)

    mock_ffmpeg.assert_called_once()
    call_args = mock_ffmpeg.call_args[0][0]

    assert "-skip_frame" in call_args
    assert "nokey" in call_args
    assert "-i" in call_args
    assert str(video_path) in call_args
    assert "-vsync" in call_args
    assert "vfr" in call_args
    assert "-frame_pts" in call_args
    assert "true" in call_args
    assert "-f" in call_args
    assert "image2" in call_args


def test_extract_keyframes_returns_deduplicated_frames(
    mock_context: RuntimeContext, tmp_path: Path
) -> None:
    video_path = tmp_path / "test.mp4"
    _ = video_path.write_bytes(b"fake video")

    keyframe_dir = mock_context.output_dir / "test" / "keyframes"
    keyframe_dir.mkdir(parents=True)

    mock_frames = create_mock_frames(keyframe_dir, 10)

    with patch("src.stages.keyframe.run_ffmpeg"), patch(
        "src.stages.keyframe.deduplicate_frames", return_value=mock_frames
    ):
        result = extract_keyframes(mock_context, video_path)

    assert result == mock_frames


def test_extract_keyframes_enforces_max_frames(
    mock_context: RuntimeContext, tmp_path: Path
) -> None:
    video_path = tmp_path / "test.mp4"
    _ = video_path.write_bytes(b"fake video")

    keyframe_dir = mock_context.output_dir / "test" / "keyframes"
    keyframe_dir.mkdir(parents=True)

    mock_frames = create_mock_frames(keyframe_dir, 10)

    with patch("src.stages.keyframe.run_ffmpeg"), patch(
        "src.stages.keyframe.deduplicate_frames", return_value=mock_frames
    ):
        result = extract_keyframes(mock_context, video_path)

    assert len(result) == 10

    mock_context.config["keyframe"]["max_frames"] = 3

    with patch("src.stages.keyframe.run_ffmpeg"), patch(
        "src.stages.keyframe.deduplicate_frames", return_value=mock_frames
    ):
        result = extract_keyframes(mock_context, video_path)

    assert len(result) == 3
    assert result == mock_frames[:3]


def test_extract_keyframes_returns_sorted_paths(
    mock_context: RuntimeContext, tmp_path: Path
) -> None:
    video_path = tmp_path / "test.mp4"
    _ = video_path.write_bytes(b"fake video")

    keyframe_dir = mock_context.output_dir / "test" / "keyframes"
    keyframe_dir.mkdir(parents=True)

    mock_frames = create_mock_frames(keyframe_dir, 5)

    with patch("src.stages.keyframe.run_ffmpeg"), patch(
        "src.stages.keyframe.deduplicate_frames", return_value=mock_frames
    ):
        result = extract_keyframes(mock_context, video_path)

    assert result == sorted(mock_frames, key=lambda p: p.name)


def test_parse_frame_timestamp_extracts_number() -> None:
    assert parse_frame_timestamp(Path("key_000123.png")) == 123.0
    assert parse_frame_timestamp(Path("key_000001.png")) == 1.0
    assert parse_frame_timestamp(Path("key_999999.png")) == 999999.0
    assert parse_frame_timestamp(Path("key_045678.jpg")) == 45678.0


def test_parse_frame_timestamp_invalid_returns_none() -> None:
    assert parse_frame_timestamp(Path("not_a_frame.png")) is None
    assert parse_frame_timestamp(Path("frame_001.png")) is None
    assert parse_frame_timestamp(Path("key.png")) is None
    assert parse_frame_timestamp(Path("key_abc.png")) is None
    assert parse_frame_timestamp(Path("key_001.txt")) is None


def test_extract_keyframes_empty_result(
    mock_context: RuntimeContext, tmp_path: Path
) -> None:
    video_path = tmp_path / "test.mp4"
    _ = video_path.write_bytes(b"fake video")

    keyframe_dir = mock_context.output_dir / "test" / "keyframes"
    keyframe_dir.mkdir(parents=True)

    with patch("src.stages.keyframe.run_ffmpeg"), patch(
        "src.stages.keyframe.deduplicate_frames", return_value=[]
    ):
        result = extract_keyframes(mock_context, video_path)

    assert result == []
