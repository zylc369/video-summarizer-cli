"""Tests for ASR transcription stage."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.context import ASREngineError
from src.context import RuntimeContext
from src.stages.asr import TranscriptSegment
from src.stages.asr import resolve_asr_engine
from src.stages.asr import run_asr


def _make_segment(text: str, start: float, end: float) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


@pytest.fixture
def mock_context(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        env="windows_gpu",
        os_name="windows",
        has_cuda=True,
        has_mps=False,
        gpu_name="NVIDIA GeForce RTX 4070 Ti",
        gpu_vram_mb=12288,
        cpu_count=16,
        total_ram_gb=128.0,
        config={
            "asr": {
                "engine": "faster_whisper",
                "engines": {
                    "faster_whisper": {
                        "model": "large-v3-turbo",
                        "device": "cuda",
                        "compute_type": "float16",
                        "task": "transcribe",
                        "language": None,
                    },
                },
            },
        },
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    )


class TestResolveAsrEngine:
    def test_resolve_asr_engine_user_specified(self, mock_context: RuntimeContext) -> None:
        mock_context.config["asr"]["engine"] = "mlx_whisper"
        assert resolve_asr_engine(mock_context) == "mlx_whisper"

    def test_resolve_asr_engine_auto_cuda(self, mock_context: RuntimeContext) -> None:
        mock_context.config["asr"]["engine"] = "auto"
        mock_context.has_cuda = True
        mock_context.os_name = "linux"
        assert resolve_asr_engine(mock_context) == "faster_whisper"

    def test_resolve_asr_engine_auto_mac(self, mock_context: RuntimeContext) -> None:
        mock_context.config["asr"]["engine"] = "auto"
        mock_context.has_cuda = False
        mock_context.os_name = "darwin"
        assert resolve_asr_engine(mock_context) == "mlx_whisper"

    def test_resolve_asr_engine_auto_cpu_fallback(self, mock_context: RuntimeContext) -> None:
        mock_context.config["asr"]["engine"] = "auto"
        mock_context.has_cuda = False
        mock_context.os_name = "linux"
        assert resolve_asr_engine(mock_context) == "faster_whisper"


class TestRunAsr:
    @patch("src.stages.asr._run_faster_whisper")
    def test_run_asr_returns_transcript_segments(
        self,
        mock_run_fw: MagicMock,
        mock_context: RuntimeContext,
        tmp_path: Path,
    ) -> None:
        mock_run_fw.return_value = [
            TranscriptSegment(text="Hello world", start=0.0, end=2.5, language="en"),
            TranscriptSegment(text="Second segment", start=2.5, end=5.0, language="en"),
        ]
        audio_path = tmp_path / "test_video.wav"
        audio_path.touch()

        result = run_asr(mock_context, audio_path)

        assert len(result) == 2
        assert result[0].text == "Hello world"
        assert result[0].start == 0.0
        assert result[0].end == 2.5
        assert result[1].text == "Second segment"

    @patch("src.stages.asr._run_faster_whisper")
    def test_run_asr_saves_transcript_json(
        self,
        mock_run_fw: MagicMock,
        mock_context: RuntimeContext,
        tmp_path: Path,
    ) -> None:
        mock_run_fw.return_value = [
            TranscriptSegment(text="Hello", start=0.0, end=1.0, language="en"),
        ]
        audio_path = tmp_path / "my_video.wav"
        audio_path.touch()

        run_asr(mock_context, audio_path)

        transcript_path = mock_context.output_dir / "my_video" / "transcript.json"
        assert transcript_path.exists()
        data = json.loads(transcript_path.read_text())
        assert len(data) == 1
        assert data[0]["text"] == "Hello"
        assert data[0]["start"] == 0.0
        assert data[0]["end"] == 1.0
        assert data[0]["language"] == "en"

    @patch("faster_whisper.WhisperModel")
    def test_run_asr_faster_whisper_params(
        self,
        mock_whisper_cls: MagicMock,
        mock_context: RuntimeContext,
        tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = (iter([]), mock_info)
        mock_whisper_cls.return_value = mock_model

        audio_path = tmp_path / "video.wav"
        audio_path.touch()

        run_asr(mock_context, audio_path)

        mock_whisper_cls.assert_called_once_with(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
        )
        mock_model.transcribe.assert_called_once_with(
            str(audio_path),
            language=None,
            task="transcribe",
        )

    @patch.dict("sys.modules", {"faster_whisper": None})
    def test_run_asr_import_error_raises_engine_error(
        self,
        mock_context: RuntimeContext,
        tmp_path: Path,
    ) -> None:
        audio_path = tmp_path / "video.wav"
        audio_path.touch()

        with pytest.raises(ASREngineError, match="faster_whisper is not installed"):
            run_asr(mock_context, audio_path)
