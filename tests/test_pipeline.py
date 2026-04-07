"""Tests for pipeline orchestration module."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.context import RuntimeContext
from src.pipeline import run_pipeline
from src.stages.asr import TranscriptSegment
from src.stages.visual import VisualAnalysisResult


@pytest.fixture
def mock_context(tmp_path: Path) -> RuntimeContext:
    """Create a RuntimeContext with test configuration."""
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
            "asr": {"engine": "faster_whisper", "engines": {"faster_whisper": {"model": "small"}}},
            "keyframe": {"method": "keyframe", "format": "png", "dedup_threshold": 0.9, "max_frames": 30},
            "visual": {"model": "glm-4.6v-flashx", "concurrency": 2, "max_tokens": 4096},
            "summary": {"model": "glm-4-flashx", "max_tokens": 4096, "timeout": 120},
        },
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    )


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a dummy video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    return video_path


def _make_transcript_segments() -> list[TranscriptSegment]:
    """Create sample transcript segments for testing."""
    return [
        TranscriptSegment(text="Hello world", start=0.0, end=2.0, language="en"),
        TranscriptSegment(text="Testing pipeline", start=2.0, end=4.0, language="en"),
    ]


def _make_visual_results() -> list[VisualAnalysisResult]:
    """Create sample visual analysis results for testing."""
    return [
        VisualAnalysisResult(
            frame_path=Path("/tmp/key_000001.png"),
            timestamp=1.0,
            frame_type="ide",
            text_content="print('hello')",
            description="Code editor showing hello world",
        ),
    ]


@patch("src.pipeline.generate_summary")
@patch("src.pipeline.analyze_frames")
@patch("src.pipeline.extract_keyframes")
@patch("src.pipeline.run_asr")
@patch("src.pipeline.extract_audio")
def test_run_pipeline_full_runs_all_stages(
    mock_extract_audio: MagicMock,
    mock_run_asr: MagicMock,
    mock_extract_keyframes: MagicMock,
    mock_analyze_frames: MagicMock,
    mock_generate_summary: MagicMock,
    mock_context: RuntimeContext,
    sample_video: Path,
) -> None:
    """Verify all stages are called in correct order in full pipeline."""
    mock_extract_audio.return_value = mock_context.output_dir / "test_video" / "audio.wav"
    mock_run_asr.return_value = _make_transcript_segments()
    mock_extract_keyframes.return_value = [Path("/tmp/key_000001.png")]
    mock_analyze_frames.return_value = _make_visual_results()
    mock_generate_summary.return_value = "# Summary"

    run_pipeline(mock_context, sample_video)

    mock_extract_audio.assert_called_once()
    mock_run_asr.assert_called_once()
    mock_extract_keyframes.assert_called_once()
    mock_analyze_frames.assert_called_once()
    mock_generate_summary.assert_called_once()


@patch("src.pipeline.generate_summary")
@patch("src.pipeline.analyze_frames")
@patch("src.pipeline.extract_keyframes")
@patch("src.pipeline.run_asr")
@patch("src.pipeline.extract_audio")
def test_run_pipeline_parallel_execution(
    mock_extract_audio: MagicMock,
    mock_run_asr: MagicMock,
    mock_extract_keyframes: MagicMock,
    mock_analyze_frames: MagicMock,
    mock_generate_summary: MagicMock,
    mock_context: RuntimeContext,
    sample_video: Path,
) -> None:
    """Verify both audio and visual branches execute via ThreadPoolExecutor."""
    mock_extract_audio.return_value = mock_context.output_dir / "test_video" / "audio.wav"
    mock_run_asr.return_value = _make_transcript_segments()
    mock_extract_keyframes.return_value = [Path("/tmp/key_000001.png")]
    mock_analyze_frames.return_value = _make_visual_results()
    mock_generate_summary.return_value = "# Summary"

    run_pipeline(mock_context, sample_video)

    mock_extract_audio.assert_called_once_with(mock_context, sample_video)
    mock_run_asr.assert_called_once()
    mock_extract_keyframes.assert_called_once_with(mock_context, sample_video)
    mock_analyze_frames.assert_called_once()


@patch("src.pipeline.generate_summary")
@patch("src.pipeline.analyze_frames")
@patch("src.pipeline.extract_keyframes")
@patch("src.pipeline.run_asr")
@patch("src.pipeline.extract_audio")
def test_run_pipeline_returns_summary_path(
    mock_extract_audio: MagicMock,
    mock_run_asr: MagicMock,
    mock_extract_keyframes: MagicMock,
    mock_analyze_frames: MagicMock,
    mock_generate_summary: MagicMock,
    mock_context: RuntimeContext,
    sample_video: Path,
) -> None:
    """Verify pipeline returns correct summary path."""
    mock_extract_audio.return_value = mock_context.output_dir / "test_video" / "audio.wav"
    mock_run_asr.return_value = _make_transcript_segments()
    mock_extract_keyframes.return_value = [Path("/tmp/key_000001.png")]
    mock_analyze_frames.return_value = _make_visual_results()
    mock_generate_summary.return_value = "# Summary"

    result = run_pipeline(mock_context, sample_video)

    expected_path = mock_context.output_dir / "test_video" / "summary.md"
    assert result == expected_path


@patch("src.pipeline.analyze_frames")
@patch("src.pipeline.extract_keyframes")
@patch("src.pipeline.run_asr")
@patch("src.pipeline.extract_audio")
def test_run_pipeline_only_audio(
    mock_extract_audio: MagicMock,
    mock_run_asr: MagicMock,
    mock_extract_keyframes: MagicMock,
    mock_analyze_frames: MagicMock,
    mock_context: RuntimeContext,
    sample_video: Path,
) -> None:
    """Verify only extract_audio is called with --only audio."""
    mock_extract_audio.return_value = mock_context.output_dir / "test_video" / "audio.wav"

    run_pipeline(mock_context, sample_video, only="audio")

    mock_extract_audio.assert_called_once()
    mock_run_asr.assert_not_called()
    mock_extract_keyframes.assert_not_called()
    mock_analyze_frames.assert_not_called()


@patch("src.pipeline.analyze_frames")
@patch("src.pipeline.extract_keyframes")
@patch("src.pipeline.run_asr")
@patch("src.pipeline.extract_audio")
def test_run_pipeline_only_asr(
    mock_extract_audio: MagicMock,
    mock_run_asr: MagicMock,
    mock_extract_keyframes: MagicMock,
    mock_analyze_frames: MagicMock,
    mock_context: RuntimeContext,
    sample_video: Path,
) -> None:
    """Verify extract_audio + run_asr called with --only asr, visual/summary skipped."""
    audio_path = mock_context.output_dir / "test_video" / "audio.wav"
    mock_extract_audio.return_value = audio_path
    mock_run_asr.return_value = _make_transcript_segments()

    run_pipeline(mock_context, sample_video, only="asr")

    mock_extract_audio.assert_called_once()
    mock_run_asr.assert_called_once_with(mock_context, audio_path)
    mock_extract_keyframes.assert_not_called()
    mock_analyze_frames.assert_not_called()


@patch("src.pipeline.generate_summary")
@patch("src.pipeline.extract_audio")
@patch("src.pipeline.run_asr")
@patch("src.pipeline.analyze_frames")
@patch("src.pipeline.extract_keyframes")
def test_run_pipeline_only_visual(
    mock_extract_keyframes: MagicMock,
    mock_analyze_frames: MagicMock,
    mock_run_asr: MagicMock,
    mock_extract_audio: MagicMock,
    mock_generate_summary: MagicMock,
    mock_context: RuntimeContext,
    sample_video: Path,
) -> None:
    """Verify extract_keyframes + analyze_frames called with --only visual."""
    frame_paths = [Path("/tmp/key_000001.png")]
    mock_extract_keyframes.return_value = frame_paths

    run_pipeline(mock_context, sample_video, only="visual")

    mock_extract_keyframes.assert_called_once()
    mock_analyze_frames.assert_called_once()
    mock_extract_audio.assert_not_called()
    mock_run_asr.assert_not_called()
    mock_generate_summary.assert_not_called()


@patch("src.pipeline.generate_summary")
@patch("src.pipeline.analyze_frames")
@patch("src.pipeline.extract_keyframes")
@patch("src.pipeline.run_asr")
@patch("src.pipeline.extract_audio")
def test_run_pipeline_resume_skips_existing(
    mock_extract_audio: MagicMock,
    mock_run_asr: MagicMock,
    mock_extract_keyframes: MagicMock,
    mock_analyze_frames: MagicMock,
    mock_generate_summary: MagicMock,
    mock_context: RuntimeContext,
    sample_video: Path,
) -> None:
    """Verify ASR is skipped when transcript.json already exists with resume=True."""
    video_output_dir = mock_context.output_dir / "test_video"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    transcript_data = [
        {"text": "Hello", "start": 0.0, "end": 1.0, "language": "en"},
    ]
    transcript_path = video_output_dir / "transcript.json"
    transcript_path.write_text(json.dumps(transcript_data), encoding="utf-8")

    mock_extract_keyframes.return_value = [Path("/tmp/key_000001.png")]
    mock_analyze_frames.return_value = _make_visual_results()
    mock_generate_summary.return_value = "# Summary"

    run_pipeline(mock_context, sample_video, resume=True)

    mock_extract_audio.assert_not_called()
    mock_run_asr.assert_not_called()
    mock_extract_keyframes.assert_called_once()
    mock_analyze_frames.assert_called_once()
    mock_generate_summary.assert_called_once()


def test_run_pipeline_invalid_video_raises(
    mock_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    """Verify FileNotFoundError for non-existent video file."""
    fake_video = tmp_path / "nonexistent.mp4"

    with pytest.raises(FileNotFoundError, match="Video file not found"):
        run_pipeline(mock_context, fake_video)


@patch("src.pipeline.generate_summary")
@patch("src.pipeline.analyze_frames")
@patch("src.pipeline.extract_keyframes")
@patch("src.pipeline.run_asr")
@patch("src.pipeline.extract_audio")
def test_run_pipeline_creates_output_dirs(
    mock_extract_audio: MagicMock,
    mock_run_asr: MagicMock,
    mock_extract_keyframes: MagicMock,
    mock_analyze_frames: MagicMock,
    mock_generate_summary: MagicMock,
    mock_context: RuntimeContext,
    sample_video: Path,
) -> None:
    """Verify output directory is created for the video."""
    mock_extract_audio.return_value = mock_context.output_dir / "test_video" / "audio.wav"
    mock_run_asr.return_value = _make_transcript_segments()
    mock_extract_keyframes.return_value = [Path("/tmp/key_000001.png")]
    mock_analyze_frames.return_value = _make_visual_results()

    run_pipeline(mock_context, sample_video)

    expected_dir = mock_context.output_dir / "test_video"
    assert expected_dir.exists()
    assert expected_dir.is_dir()
    mock_generate_summary.assert_called_once()
