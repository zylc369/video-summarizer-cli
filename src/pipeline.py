"""Pipeline orchestration: runs stages 1-4 in parallel, then stage 5."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.context import RuntimeContext
from src.stages.asr import TranscriptSegment
from src.stages.asr import run_asr
from src.stages.audio_extractor import extract_audio
from src.stages.keyframe import extract_keyframes
from src.stages.summarizer import generate_summary
from src.stages.visual import VisualAnalysisResult
from src.stages.visual import analyze_frames

logger = logging.getLogger(__name__)


def run_pipeline(
    context: RuntimeContext,
    video_path: Path,
    only: str | None = None,
    resume: bool = False,
) -> Path:
    """Run the video summarization pipeline.

    Stages 1-2 (audio + ASR) and stages 3-4 (keyframes + visual) run in parallel.
    Stage 5 (summary) runs after both branches complete.

    Args:
        context: Runtime context with config and output directories.
        video_path: Path to the input video file.
        only: If specified, run only that stage or stage pair.
            Supported values: "audio", "asr", "keyframe", "visual", "summary".
        resume: If True, skip stages whose artifacts already exist on disk.

    Returns:
        Path to the generated summary.md file.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_stem = video_path.stem
    video_output_dir = context.output_dir / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = video_output_dir / "summary.md"
    transcript_path = video_output_dir / "transcript.json"
    visual_path = video_output_dir / "visual_analysis.json"

    if only is not None:
        logger.info("Running only stage(s): %s", only)
        _run_only_stages(context, video_path, only, transcript_path, visual_path)
        return summary_path

    transcript_segments: list[TranscriptSegment] | None = None
    visual_results: list[VisualAnalysisResult] | None = None

    if resume:
        if _check_artifact_exists(transcript_path):
            logger.info("Resume: loading existing transcript from %s", transcript_path)
            transcript_segments = _load_transcript(transcript_path)
        if _check_artifact_exists(visual_path):
            logger.info("Resume: loading existing visual analysis from %s", visual_path)
            visual_results = _load_visual_results(visual_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        audio_future = None
        visual_future = None

        if transcript_segments is None:
            audio_future = executor.submit(_run_audio_branch, context, video_path)
        if visual_results is None:
            visual_future = executor.submit(_run_visual_branch, context, video_path)

        if audio_future is not None:
            transcript_segments = audio_future.result()
        if visual_future is not None:
            visual_results = visual_future.result()

    if transcript_segments is None:
        transcript_segments = []
    if visual_results is None:
        visual_results = []

    logger.info("Generating summary for %s", video_stem)
    _ = generate_summary(context, transcript_segments, visual_results, video_stem)

    return summary_path


def _run_only_stages(
    context: RuntimeContext,
    video_path: Path,
    only: str,
    transcript_path: Path,
    visual_path: Path,
) -> None:
    """Execute a single stage or stage pair based on the --only flag.

    Args:
        context: Runtime context with config and output directories.
        video_path: Path to the input video file.
        only: Stage name to run exclusively.
        transcript_path: Expected path to transcript.json.
        visual_path: Expected path to visual_analysis.json.
    """
    if only == "audio":
        _ = extract_audio(context, video_path)
    elif only == "asr":
        audio_path = extract_audio(context, video_path)
        _ = run_asr(context, audio_path)
    elif only == "keyframe":
        _ = extract_keyframes(context, video_path)
    elif only == "visual":
        frame_paths = extract_keyframes(context, video_path)
        _ = analyze_frames(context, frame_paths)
    elif only == "summary":
        video_stem = video_path.stem
        segments = (
            _load_transcript(transcript_path) if _check_artifact_exists(transcript_path) else []
        )
        results = (
            _load_visual_results(visual_path) if _check_artifact_exists(visual_path) else []
        )
        _ = generate_summary(context, segments, results, video_stem)
    else:
        logger.warning("Unknown --only stage: %s, no action taken", only)


def _run_audio_branch(context: RuntimeContext, video_path: Path) -> list[TranscriptSegment]:
    """Run audio extraction followed by ASR transcription.

    Args:
        context: Runtime context with config and output directories.
        video_path: Path to the input video file.

    Returns:
        List of transcript segments from ASR.
    """
    logger.info("Audio branch: extracting audio from %s", video_path)
    audio_path = extract_audio(context, video_path)
    logger.info("Audio branch: running ASR on %s", audio_path)
    return run_asr(context, audio_path)


def _run_visual_branch(context: RuntimeContext, video_path: Path) -> list[VisualAnalysisResult]:
    """Run keyframe extraction followed by visual analysis.

    Args:
        context: Runtime context with config and output directories.
        video_path: Path to the input video file.

    Returns:
        List of visual analysis results.
    """
    logger.info("Visual branch: extracting keyframes from %s", video_path)
    frame_paths = extract_keyframes(context, video_path)
    logger.info("Visual branch: analyzing %d frames", len(frame_paths))
    return analyze_frames(context, frame_paths)


def _check_artifact_exists(path: Path) -> bool:
    """Check if an artifact file exists and has content.

    Args:
        path: Path to check.

    Returns:
        True if file exists and has size > 0.
    """
    return path.exists() and path.stat().st_size > 0


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    """Load transcript segments from a JSON artifact file.

    Args:
        path: Path to transcript.json.

    Returns:
        List of TranscriptSegment loaded from the file.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        TranscriptSegment(
            text=item["text"],
            start=item["start"],
            end=item["end"],
            language=item.get("language"),
        )
        for item in data
    ]


def _load_visual_results(path: Path) -> list[VisualAnalysisResult]:
    """Load visual analysis results from a JSON artifact file.

    Args:
        path: Path to visual_analysis.json.

    Returns:
        List of VisualAnalysisResult loaded from the file.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        VisualAnalysisResult(
            frame_path=Path(item["frame_path"]),
            timestamp=item["timestamp"],
            frame_type=item["frame_type"],
            text_content=item["text_content"],
            description=item["description"],
        )
        for item in data
    ]
