"""Tests for visual analysis stage."""

import json
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.context import RuntimeContext
from src.stages.visual import (
    VisualAnalysisResult,
    _build_analysis_prompt,
    _parse_visual_response,
    analyze_frames,
)


def _create_minimal_png() -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc & 0xFFFFFFFF)
    raw = zlib.compress(bytes([0, 0]))
    idat = (
        struct.pack(">I", len(raw))
        + b"IDAT"
        + raw
        + struct.pack(">I", zlib.crc32(b"IDAT" + raw) & 0xFFFFFFFF)
    )
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return signature + ihdr + idat + iend


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
            "visual": {
                "model": "glm-4.6v-flashx",
                "api_key": "test-key",
                "max_tokens": 2000,
                "concurrency": 5,
                "timeout": 30,
                "retry": 3,
            },
        },
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    )


@pytest.fixture
def mock_frames(tmp_path: Path) -> list[Path]:
    keyframe_dir = tmp_path / "output" / "test_video" / "keyframes"
    keyframe_dir.mkdir(parents=True)
    frames = []
    for i in range(3):
        path = keyframe_dir / f"key_{i:06d}.png"
        path.write_bytes(_create_minimal_png())
        frames.append(path)
    return frames


def test_analyze_frames_returns_results(mock_context: RuntimeContext, mock_frames: list[Path]) -> None:
    mock_client = MagicMock()
    mock_client.vision_analysis.return_value = (
        "TYPE: ide\nTEXT: print('hello')\nDESCRIPTION: A code editor showing a hello world program"
    )

    with patch("src.stages.visual.create_zhipu_client", return_value=mock_client):
        results = analyze_frames(mock_context, mock_frames)

    assert len(results) == 3
    assert all(isinstance(r, VisualAnalysisResult) for r in results)
    assert all(r.frame_type == "ide" for r in results)


def test_analyze_frames_saves_json(mock_context: RuntimeContext, mock_frames: list[Path]) -> None:
    mock_client = MagicMock()
    mock_client.vision_analysis.return_value = (
        "TYPE: terminal\nTEXT: ls -la\nDESCRIPTION: A terminal listing files"
    )

    with patch("src.stages.visual.create_zhipu_client", return_value=mock_client):
        results = analyze_frames(mock_context, mock_frames)

    json_path = mock_context.output_dir / "test_video" / "visual_analysis.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data) == len(results)
    assert all("frame_path" in item for item in data)
    assert all("timestamp" in item for item in data)
    assert all("frame_type" in item for item in data)
    assert all("text_content" in item for item in data)
    assert all("description" in item for item in data)


def test_analyze_frames_respects_concurrency(mock_context: RuntimeContext, mock_frames: list[Path]) -> None:
    mock_client = MagicMock()
    mock_client.vision_analysis.return_value = "TYPE: other\nTEXT: \nDESCRIPTION: empty"

    with patch("src.stages.visual.create_zhipu_client", return_value=mock_client):
        with patch("src.stages.visual.ThreadPoolExecutor", wraps=__import__("concurrent.futures").futures.ThreadPoolExecutor) as mock_tpe:
            analyze_frames(mock_context, mock_frames)
            mock_tpe.assert_called_once_with(max_workers=5)


def test_analyze_frames_handles_failure_gracefully(mock_context: RuntimeContext, mock_frames: list[Path]) -> None:
    mock_client = MagicMock()
    call_count = 0

    def _side_effect(**kwargs: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("API failure")
        return "TYPE: ide\nTEXT: code\nDESCRIPTION: description"

    mock_client.vision_analysis.side_effect = _side_effect

    with patch("src.stages.visual.create_zhipu_client", return_value=mock_client):
        results = analyze_frames(mock_context, mock_frames)

    assert len(results) == 3
    error_results = [r for r in results if r.frame_type == "error"]
    assert len(error_results) == 1
    assert error_results[0].text_content == ""
    assert error_results[0].description == ""


def test_build_analysis_prompt_contains_required_fields() -> None:
    prompt = _build_analysis_prompt()
    assert "TYPE:" in prompt
    assert "TEXT:" in prompt
    assert "DESCRIPTION:" in prompt
    assert "ide" in prompt
    assert "terminal" in prompt
    assert "ppt" in prompt
    assert "ida_pro" in prompt
    assert "other" in prompt


def test_parse_visual_response_correct() -> None:
    response = "TYPE: ide\nTEXT: def hello():\n    print('world')\nDESCRIPTION: Python function in IDE"
    result = _parse_visual_response(response, Path("/tmp/frame.png"), 1.0)

    assert result.frame_type == "ide"
    assert "def hello():" in result.text_content
    assert "print('world')" in result.text_content
    assert result.description == "Python function in IDE"
    assert result.timestamp == 1.0


def test_parse_visual_response_malformed() -> None:
    result = _parse_visual_response("this is garbage text with no structure", Path("/tmp/frame.png"), None)

    assert result.frame_type == "other"
    assert result.text_content == ""
    assert result.description == ""
    assert result.timestamp is None


def test_analyze_frames_sorts_by_timestamp(mock_context: RuntimeContext, tmp_path: Path) -> None:
    keyframe_dir = tmp_path / "output" / "sort_video" / "keyframes"
    keyframe_dir.mkdir(parents=True)

    frame_paths = []
    for ts in [200, 50, 150]:
        path = keyframe_dir / f"key_{ts:06d}.png"
        path.write_bytes(_create_minimal_png())
        frame_paths.append(path)

    mock_client = MagicMock()
    mock_client.vision_analysis.return_value = "TYPE: other\nTEXT: test\nDESCRIPTION: test"

    with patch("src.stages.visual.create_zhipu_client", return_value=mock_client):
        results = analyze_frames(mock_context, frame_paths)

    timestamps = [r.timestamp for r in results]
    assert timestamps == sorted(timestamps)
