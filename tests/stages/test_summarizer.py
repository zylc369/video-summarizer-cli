"""Tests for the summarizer stage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.context import RuntimeContext
from src.stages.asr import TranscriptSegment
from src.stages.summarizer import (
    _build_summary_prompt,
    _format_timestamp,
    _format_transcript,
    generate_summary,
)
from src.stages.visual import VisualAnalysisResult


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
        config={
            "summary": {
                "model": "glm-4.7",
                "api_key": "test-key",
                "max_tokens": 8000,
                "timeout": 120,
                "retry": 3,
                "prompt_template": None,
            },
            "output": {
                "include_screenshots": True,
                "screenshot_rel_path": "./keyframes/",
            },
        },
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    )


@pytest.fixture
def sample_transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text="Hello, today we'll learn about Python.", start=0.0, end=5.0),
        TranscriptSegment(text="Let's start with variables.", start=5.5, end=10.0),
    ]


@pytest.fixture
def sample_visual_results(tmp_path: Path) -> list[VisualAnalysisResult]:
    return [
        VisualAnalysisResult(
            frame_path=tmp_path / "key_000001.png",
            timestamp=1.0,
            frame_type="ide",
            text_content="x = 42\nprint(x)",
            description="Showing variable assignment in IDE",
        ),
    ]


def test_generate_summary_returns_markdown(
    mock_context: RuntimeContext,
    sample_transcript: list[TranscriptSegment],
    sample_visual_results: list[VisualAnalysisResult],
) -> None:
    with patch("src.stages.summarizer.create_zhipu_client") as mock_create:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "# Python Tutorial\n\nOverview here."
        mock_create.return_value = mock_client

        result = generate_summary(mock_context, sample_transcript, sample_visual_results, "test_video")

        assert isinstance(result, str)
        assert "# Python Tutorial" in result


def test_generate_summary_saves_file(
    mock_context: RuntimeContext,
    sample_transcript: list[TranscriptSegment],
    sample_visual_results: list[VisualAnalysisResult],
) -> None:
    expected_content = "# Python Tutorial\n\nSaved content here."
    with patch("src.stages.summarizer.create_zhipu_client") as mock_create:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = expected_content
        mock_create.return_value = mock_client

        generate_summary(mock_context, sample_transcript, sample_visual_results, "test_video")

        summary_path = mock_context.output_dir / "test_video" / "summary.md"
        assert summary_path.exists()
        assert summary_path.read_text(encoding="utf-8") == expected_content


def test_generate_summary_uses_correct_model(
    mock_context: RuntimeContext,
    sample_transcript: list[TranscriptSegment],
    sample_visual_results: list[VisualAnalysisResult],
) -> None:
    with patch("src.stages.summarizer.create_zhipu_client") as mock_create:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "# Summary"
        mock_create.return_value = mock_client

        generate_summary(mock_context, sample_transcript, sample_visual_results, "test_video")

        call_kwargs = mock_client.chat_completion.call_args
        assert call_kwargs.kwargs["model"] == "glm-4.7"
        assert call_kwargs.kwargs["temperature"] == 0.7
        assert call_kwargs.kwargs["max_tokens"] == 8000
        assert call_kwargs.kwargs["timeout"] == 120


def test_build_summary_prompt_structure(
    sample_transcript: list[TranscriptSegment],
    sample_visual_results: list[VisualAnalysisResult],
) -> None:
    messages = _build_summary_prompt(
        transcript_segments=sample_transcript,
        visual_results=sample_visual_results,
        video_name="test_video",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_summary_prompt_includes_transcript(
    sample_transcript: list[TranscriptSegment],
) -> None:
    messages = _build_summary_prompt(
        transcript_segments=sample_transcript,
        visual_results=[],
        video_name="test_video",
    )

    user_content = messages[1]["content"]
    assert "Hello, today we'll learn about Python." in user_content
    assert "Let's start with variables." in user_content
    assert "[00:00:00]" in user_content


def test_build_summary_prompt_includes_visual(
    sample_visual_results: list[VisualAnalysisResult],
) -> None:
    messages = _build_summary_prompt(
        transcript_segments=[],
        visual_results=sample_visual_results,
        video_name="test_video",
    )

    user_content = messages[1]["content"]
    assert "ide" in user_content
    assert "x = 42" in user_content
    assert "Showing variable assignment in IDE" in user_content
    assert "./keyframes/key_000001.png" in user_content


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (3725.5, "01:02:05"),
        (0.0, "00:00:00"),
        (59.9, "00:00:59"),
        (3600.0, "01:00:00"),
        (65.0, "00:01:05"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert _format_timestamp(seconds) == expected


def test_format_timestamp_negative() -> None:
    assert _format_timestamp(-1.0) == "00:00:00"


def test_format_timestamp_none() -> None:
    assert _format_timestamp(None) == "00:00:00"  # type: ignore[arg-type]


def test_format_transcript() -> None:
    segments = [
        TranscriptSegment(text="First sentence.", start=0.0, end=3.0),
        TranscriptSegment(text="Second sentence.", start=5.0, end=8.0),
    ]
    result = _format_transcript(segments)

    assert "[00:00:00] First sentence.\n" in result
    assert "[00:00:05] Second sentence.\n" in result


def test_generate_summary_with_empty_transcript(
    mock_context: RuntimeContext,
    sample_visual_results: list[VisualAnalysisResult],
) -> None:
    with patch("src.stages.summarizer.create_zhipu_client") as mock_create:
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = "# Summary\n\nNo transcript available."
        mock_create.return_value = mock_client

        result = generate_summary(mock_context, [], sample_visual_results, "test_video")

        assert isinstance(result, str)
        mock_client.chat_completion.assert_called_once()
