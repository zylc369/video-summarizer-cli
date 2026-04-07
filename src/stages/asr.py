"""ASR transcription stage with multi-engine support."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.context import ASREngineError
from src.context import RuntimeContext

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single transcript segment with timing information."""

    text: str
    start: float
    end: float
    language: str | None = None


def resolve_asr_engine(context: RuntimeContext) -> str:
    """Resolve which ASR engine to use based on config and environment.

    If config explicitly specifies an engine (not "auto"), return it as-is.
    Otherwise, auto-detect based on hardware capabilities.

    Args:
        context: Runtime context with config and environment info.

    Returns:
        Engine name string (e.g. "faster_whisper", "mlx_whisper").
    """
    configured_engine: str = context.config["asr"]["engine"]
    if configured_engine != "auto":
        return configured_engine

    if context.has_cuda:
        return "faster_whisper"
    if context.os_name == "darwin":
        return "mlx_whisper"
    return "faster_whisper"


def run_asr(context: RuntimeContext, audio_path: Path) -> list[TranscriptSegment]:
    """Run ASR transcription on the given audio file.

    Resolves the engine, loads engine-specific config, runs transcription,
    saves transcript.json, and returns structured segments.

    Args:
        context: Runtime context with config and output directory info.
        audio_path: Path to the audio WAV file.

    Returns:
        List of TranscriptSegment with text, timing, and language info.

    Raises:
        ASREngineError: If the engine is unsupported or fails to run.
    """
    engine = resolve_asr_engine(context)
    logger.info("ASR engine resolved: %s", engine)

    engine_config: dict[str, Any] = context.config["asr"]["engines"][engine]

    if engine == "faster_whisper":
        segments = _run_faster_whisper(engine_config, audio_path)
    else:
        raise ASREngineError(f"Engine {engine} not yet supported")

    video_stem = audio_path.stem
    output_dir = context.output_dir / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.json"

    data = [
        {"text": seg.text, "start": seg.start, "end": seg.end, "language": seg.language}
        for seg in segments
    ]
    with transcript_path.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        "ASR completed: %d segments, transcript saved to %s",
        len(segments),
        transcript_path,
    )
    return segments


def _run_faster_whisper(engine_config: dict[str, Any], audio_path: Path) -> list[TranscriptSegment]:
    """Run transcription using faster-whisper engine.

    Args:
        engine_config: Engine-specific configuration dict with model, device, compute_type, etc.
        audio_path: Path to the audio WAV file.

    Returns:
        List of TranscriptSegment from the transcription.

    Raises:
        ASREngineError: If faster_whisper is not installed or transcription fails.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ASREngineError("faster_whisper is not installed. Install with: pip install faster-whisper") from e

    model = WhisperModel(
        engine_config["model"],
        device=engine_config["device"],
        compute_type=engine_config["compute_type"],
    )

    segments_generator, info = model.transcribe(
        str(audio_path),
        language=engine_config.get("language"),
        task=engine_config.get("task", "transcribe"),
    )

    detected_language: str | None = info.language if hasattr(info, "language") else None
    logger.info("Detected language: %s", detected_language)

    segments: list[TranscriptSegment] = [
        TranscriptSegment(
            text=seg.text,
            start=seg.start,
            end=seg.end,
            language=detected_language,
        )
        for seg in segments_generator
    ]

    return segments
