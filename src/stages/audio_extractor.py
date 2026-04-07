"""Audio extraction stage using FFmpeg."""

import logging
from pathlib import Path

from src.context import RuntimeContext
from src.utils.ffmpeg import run_ffmpeg

logger = logging.getLogger(__name__)


def extract_audio(context: RuntimeContext, video_path: Path) -> Path:
    """Extract audio from video file as 16kHz mono WAV.

    Args:
        context: Runtime context with output directory info.
        video_path: Path to the input video file.

    Returns:
        Path to the extracted WAV file.
    """
    video_stem = video_path.stem
    audio_dir = context.output_dir / video_stem
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "audio.wav"

    logger.info("Extracting audio from %s to %s", video_path, audio_path)

    _ = run_ffmpeg([
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(audio_path),
    ])

    logger.info("Audio extracted: %s", audio_path)
    return audio_path
