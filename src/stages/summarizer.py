"""AI summarization stage: combines ASR transcript + visual analysis into structured Markdown."""

import logging

from src.context import RuntimeContext
from src.stages.asr import TranscriptSegment
from src.stages.visual import VisualAnalysisResult
from src.utils.zhipu_client import ZhipuClient, create_zhipu_client

logger = logging.getLogger(__name__)


def generate_summary(
    context: RuntimeContext,
    transcript_segments: list[TranscriptSegment],
    visual_results: list[VisualAnalysisResult],
    video_name: str,
) -> str:
    """Generate a structured Markdown summary from transcript and visual analysis.

    Args:
        context: Runtime context with config and output directory.
        transcript_segments: ASR transcript segments with timing.
        visual_results: Visual analysis results from keyframes.
        video_name: Name of the source video (used for title and output path).

    Returns:
        The generated Markdown summary content.

    Raises:
        SummarizerError: If the API call fails after retries.
    """
    summary_config: dict = context.config["summary"]
    output_config: dict = context.config.get("output", {})

    model: str = summary_config["model"]
    max_tokens: int = summary_config["max_tokens"]
    timeout: int = summary_config["timeout"]
    include_screenshots: bool = output_config.get("include_screenshots", True)
    screenshot_rel_path: str = output_config.get("screenshot_rel_path", "./keyframes/")

    client: ZhipuClient = create_zhipu_client(api_key=summary_config.get("api_key"))

    messages = _build_summary_prompt(
        transcript_segments=transcript_segments,
        visual_results=visual_results,
        video_name=video_name,
        include_screenshots=include_screenshots,
        screenshot_rel_path=screenshot_rel_path,
    )

    logger.info("Generating summary with model=%s, prompt_length=%d chars", model, len(messages[-1]["content"]))
    markdown_content = client.chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        timeout=timeout,
    )
    logger.info("Summary generated: %d chars", len(markdown_content))

    output_dir = context.output_dir / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(markdown_content, encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    return markdown_content


def _build_summary_prompt(
    transcript_segments: list[TranscriptSegment],
    visual_results: list[VisualAnalysisResult],
    video_name: str,
    include_screenshots: bool = True,
    screenshot_rel_path: str = "./keyframes/",
) -> list[dict[str, str]]:
    """Build the chat API messages array for summary generation.

    Args:
        transcript_segments: ASR transcript segments.
        visual_results: Visual analysis results from keyframes.
        video_name: Video name for title generation.
        include_screenshots: Whether to include screenshot references.
        screenshot_rel_path: Relative path to keyframe images.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    system_message = (
        "You are an expert technical content summarizer. "
        "Generate a comprehensive Markdown summary of the video content based on "
        "the provided transcript and visual analysis.\n\n"
        "IMPORTANT: You MUST write the entire summary in Chinese (简体中文).\n\n"
        "Structure the summary as follows:\n"
        "1. A title based on the video name (use # heading)\n"
        "2. An overview section (2-3 sentences summarizing the video)\n"
        "3. Detailed content sections organized by topics (use ## headings)\n"
        "4. Code examples from visual analysis where relevant (use fenced code blocks)\n"
        "5. Key takeaways as bullet points\n"
    )
    if include_screenshots:
        system_message += "6. Screenshot references using markdown image syntax like ![description](path)\n"

    user_parts: list[str] = []

    user_parts.append(f"Video: {video_name}\n")

    if transcript_segments:
        user_parts.append("## Transcript\n")
        user_parts.append(_format_transcript(transcript_segments))

    if visual_results:
        user_parts.append("\n## Visual Analysis\n")
        for result in visual_results:
            timestamp_str = _format_timestamp(result.timestamp)
            user_parts.append(f"\n### Frame at {timestamp_str}\n")
            user_parts.append(f"- Type: {result.frame_type}\n")
            if result.text_content:
                user_parts.append(f"- Text content:\n```\n{result.text_content}\n```\n")
            user_parts.append(f"- Description: {result.description}\n")
            if include_screenshots:
                user_parts.append(f"\n![{result.description}]({screenshot_rel_path}{result.frame_path.name})\n")

    user_parts.append(
        "\nPlease generate a comprehensive Markdown summary following the structure defined above. "
        "Reference relevant screenshots inline where they help illustrate the content."
    )

    user_message = "".join(user_parts)

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string like "01:02:05".
    """
    if seconds is None or seconds < 0:
        return "00:00:00"
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_transcript(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments with timestamps.

    Args:
        segments: List of transcript segments.

    Returns:
        Formatted transcript string with one segment per line.
    """
    lines: list[str] = []
    for segment in segments:
        timestamp = _format_timestamp(segment.start)
        lines.append(f"[{timestamp}] {segment.text}\n")
    return "".join(lines)
