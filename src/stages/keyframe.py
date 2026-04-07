"""Keyframe extraction from video using FFmpeg I-frame extraction."""

import logging
import re
from pathlib import Path
from typing import Final

from src.context import RuntimeContext
from src.utils.ffmpeg import run_ffmpeg
from src.utils.image_hash import deduplicate_frames

logger = logging.getLogger(__name__)

_FRAME_TIMESTAMP_PATTERN: Final = re.compile(r"key_(\d+)\.(png|jpg|jpeg)$")


def extract_keyframes(context: RuntimeContext, video_path: Path) -> list[Path]:
    """Extract keyframes from video using FFmpeg I-frame extraction.

    Args:
        context: Runtime context containing configuration and output directories
        video_path: Path to the input video file

    Returns:
        List of deduplicated keyframe paths, capped at max_frames limit
    """
    keyframe_config = context.config["keyframe"]
    output_format = keyframe_config["format"]
    dedup_threshold = keyframe_config["dedup_threshold"]
    max_frames = keyframe_config["max_frames"]

    video_stem = video_path.stem
    keyframe_dir = context.output_dir / video_stem / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting keyframes from %s to %s", video_path, keyframe_dir)

    run_ffmpeg(
        [
            "-skip_frame",
            "nokey",
            "-i",
            str(video_path),
            "-vsync",
            "vfr",
            "-frame_pts",
            "true",
            "-f",
            "image2",
            str(keyframe_dir / f"key_%06d.{output_format}"),
        ]
    )

    frame_pattern = f"key_*.{output_format}"
    raw_frames = sorted(keyframe_dir.glob(frame_pattern))

    logger.info(
        "Extracted %d raw frames, deduplicating with threshold %.2f",
        len(raw_frames),
        dedup_threshold,
    )

    deduplicated_frames = deduplicate_frames(raw_frames, threshold=dedup_threshold)

    logger.info(
        "After deduplication: %d frames, applying max_frames limit %d",
        len(deduplicated_frames),
        max_frames,
    )

    final_frames = deduplicated_frames[:max_frames]

    logger.info("Final keyframe count: %d", len(final_frames))

    return final_frames


def parse_frame_timestamp(frame_path: Path) -> float | None:
    """Extract timestamp from FFmpeg -frame_pts naming pattern.

    FFmpeg with -frame_pts creates files like key_000123.png where the number
    represents the presentation timestamp in the video's timebase units.

    Args:
        frame_path: Path to the keyframe file

    Returns:
        Timestamp as float, or None if filename doesn't match the expected pattern
    """
    match = _FRAME_TIMESTAMP_PATTERN.search(frame_path.name)
    if match is None:
        return None

    return float(match.group(1))
