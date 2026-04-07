"""Visual analysis of keyframes via Zhipu Vision API with concurrent execution."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.context import RuntimeContext
from src.stages.keyframe import parse_frame_timestamp
from src.utils.logger import get_logger
from src.utils.zhipu_client import ZhipuClient, create_zhipu_client

logger = get_logger(__name__)


@dataclass
class VisualAnalysisResult:
    """Result of visual analysis for a single keyframe."""

    frame_path: Path
    timestamp: float | None
    frame_type: str  # "ide" | "terminal" | "ppt" | "ida_pro" | "other" | "error"
    text_content: str
    description: str


def analyze_frames(context: RuntimeContext, frame_paths: list[Path]) -> list[VisualAnalysisResult]:
    """Analyze keyframes concurrently using Zhipu Vision API.

    Args:
        context: Runtime context containing configuration and output directories.
        frame_paths: List of keyframe image paths to analyze.

    Returns:
        List of VisualAnalysisResult sorted by timestamp, with failed frames
        represented as error-type results.
    """
    visual_config = context.config["visual"]
    model: str = visual_config["model"]
    api_key: str | None = visual_config.get("api_key")
    max_tokens: int = visual_config["max_tokens"]
    concurrency: int = visual_config["concurrency"]
    timeout: float | None = visual_config.get("timeout")

    client = create_zhipu_client(api_key=api_key)

    logger.info("Starting visual analysis of %d frames (concurrency=%d)", len(frame_paths), concurrency)

    results: list[VisualAnalysisResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_analyze_single_frame, client, model, fp, max_tokens, timeout): fp
            for fp in frame_paths
        }
        for future in as_completed(futures):
            frame_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error("Unexpected error analyzing frame %s: %s", frame_path, e)
                results.append(
                    VisualAnalysisResult(
                        frame_path=frame_path,
                        timestamp=parse_frame_timestamp(frame_path),
                        frame_type="error",
                        text_content="",
                        description="",
                    )
                )

    results.sort(key=lambda r: r.timestamp if r.timestamp is not None else float("inf"))

    successful = sum(1 for r in results if r.frame_type != "error")
    failed = len(results) - successful
    logger.info(
        "Visual analysis complete: %d successful, %d failed out of %d frames",
        successful,
        failed,
        len(results),
    )

    _save_results(results, context)

    return results


def _build_analysis_prompt() -> str:
    """Build the prompt instructing the vision model on how to analyze a screenshot.

    Returns:
        A prompt string requesting structured output with TYPE, TEXT, and DESCRIPTION fields.
    """
    return """Analyze this screenshot and respond in the following structured format:

TYPE: [ide/terminal/ppt/ida_pro/other]
TEXT: [extracted text content, including any code with proper indentation]
DESCRIPTION: [semantic description of what's being shown]

Guidelines:
- TYPE: Classify the screenshot type:
  - "ide": Code editor or IDE interface (VSCode, PyCharm, etc.)
  - "terminal": Terminal or command-line interface
  - "ppt": Presentation slides (PowerPoint, Keynote, etc.)
  - "ida_pro": IDA Pro or other reverse engineering tools
  - "other": Anything that doesn't fit the above categories
- TEXT: Extract ALL visible text verbatim. If there is code, preserve indentation exactly.
  If there are multiple text regions, include them all.
- DESCRIPTION: Provide a brief semantic summary of what the screenshot shows,
  including context like what activity is being performed."""


def _parse_visual_response(response_text: str, frame_path: Path, timestamp: float | None) -> VisualAnalysisResult:
    """Parse the structured response from the vision model.

    Args:
        response_text: Raw response text from the vision API.
        frame_path: Path to the analyzed frame.
        timestamp: Parsed timestamp from the frame filename.

    Returns:
        VisualAnalysisResult with extracted fields, or defaults for malformed responses.
    """
    frame_type = "other"
    text_content = ""
    description = ""

    lines = response_text.strip().splitlines()
    current_field: str | None = None
    text_lines: list[str] = []
    desc_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TYPE:"):
            frame_type = stripped[len("TYPE:"):].strip().lower()
            valid_types = {"ide", "terminal", "ppt", "ida_pro", "other"}
            if frame_type not in valid_types:
                frame_type = "other"
            current_field = None
        elif stripped.startswith("TEXT:"):
            text_part = stripped[len("TEXT:"):]
            if text_part.strip():
                text_lines.append(text_part)
            current_field = "text"
        elif stripped.startswith("DESCRIPTION:"):
            desc_part = stripped[len("DESCRIPTION:"):]
            if desc_part.strip():
                desc_lines.append(desc_part)
            current_field = "description"
        elif current_field == "text":
            text_lines.append(line)
        elif current_field == "description":
            desc_lines.append(line)

    text_content = "\n".join(text_lines).strip()
    description = "\n".join(desc_lines).strip()

    return VisualAnalysisResult(
        frame_path=frame_path,
        timestamp=timestamp,
        frame_type=frame_type,
        text_content=text_content,
        description=description,
    )


def _analyze_single_frame(
    client: ZhipuClient,
    model: str,
    frame_path: Path,
    max_tokens: int,
    timeout: float | None,
) -> VisualAnalysisResult:
    """Analyze a single frame using the vision API.

    Args:
        client: ZhipuClient instance for API calls.
        model: Vision model identifier.
        frame_path: Path to the keyframe image.
        max_tokens: Maximum tokens in the API response.
        timeout: Request timeout in seconds.

    Returns:
        VisualAnalysisResult on success, or an error-type result on failure.
    """
    try:
        image_data = frame_path.read_bytes()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        prompt = _build_analysis_prompt()
        response_text = client.vision_analysis(
            model=model,
            image_base64=base64_image,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        timestamp = parse_frame_timestamp(frame_path)
        return _parse_visual_response(response_text, frame_path, timestamp)
    except Exception as e:
        logger.warning("Failed to analyze frame %s: %s", frame_path, e)
        return VisualAnalysisResult(
            frame_path=frame_path,
            timestamp=parse_frame_timestamp(frame_path),
            frame_type="error",
            text_content="",
            description="",
        )


def _save_results(results: list[VisualAnalysisResult], context: RuntimeContext) -> None:
    """Save visual analysis results to JSON file.

    Args:
        results: List of analysis results to save.
        context: Runtime context for determining output directory.
    """
    if not results:
        return

    first_frame = results[0].frame_path
    video_stem = first_frame.parent.parent.name
    output_file = context.output_dir / video_stem / "visual_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "frame_path": str(result.frame_path),
            "timestamp": result.timestamp,
            "frame_type": result.frame_type,
            "text_content": result.text_content,
            "description": result.description,
        }
        for result in results
    ]

    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved visual analysis results to %s", output_file)
