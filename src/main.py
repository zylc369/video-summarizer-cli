"""CLI entry point: provides summarize and env-info commands."""

import tempfile
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.config_loader import load_config
from src.context import RuntimeContext
from src.env_detector import detect_system_info
from src.interactive import ensure_api_key
from src.interactive import load_dotenv_if_exists
from src.interactive import prompt_video_path
from src.pipeline import run_pipeline
from src.utils.logger import get_logger, setup_logging

console = Console()


@click.group()
def cli() -> None:
    """Video Summarizer CLI — AI-powered video summarization tool."""


@cli.command()
@click.argument("video_path", required=False, default=None)
@click.option("--interactive", "-i", is_flag=True, default=False, help="Enable interactive mode.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory.")
@click.option(
    "--asr-engine",
    type=str,
    default=None,
    help="ASR engine: faster_whisper / qwen3_asr / mlx_whisper",
)
@click.option("--visual-model", type=str, default=None, help="Vision model name.")
@click.option("--summary-model", type=str, default=None, help="Summary model name.")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None, help="Config file path.")
@click.option(
    "--only",
    type=click.Choice(["audio", "asr", "keyframe", "visual", "summary"]),
    default=None,
    help="Run only specified stage.",
)
@click.option("--resume", is_flag=True, default=False, help="Resume from existing artifacts.")
def summarize(
    video_path: str | None,
    interactive: bool,
    output: str | None,
    asr_engine: str | None,
    visual_model: str | None,
    summary_model: str | None,
    config_path: str | None,
    only: str | None,
    resume: bool,
) -> None:
    """Summarize a video file using AI."""
    if interactive:
        _run_interactive_mode(video_path, output, asr_engine, visual_model, summary_model, config_path, only, resume)
    else:
        if not video_path:
            console.print("[bold red]✗ 非交互模式下必须提供视频文件路径[/bold red]")
            console.print("用法: python -m src.main summarize <video_path>")
            console.print("或使用 [cyan]-i[/cyan] 参数进入交互模式: python -m src.main summarize -i")
            raise SystemExit(1)
        load_dotenv_if_exists()
        _execute_pipeline(video_path, output, asr_engine, visual_model, summary_model, config_path, only, resume)


def _run_interactive_mode(
    video_path: str | None,
    output: str | None,
    asr_engine: str | None,
    visual_model: str | None,
    summary_model: str | None,
    config_path: str | None,
    only: str | None,
    resume: bool,
) -> None:
    """Handle interactive mode: ensure API key and prompt for video path if needed."""
    console.print("[bold cyan]🎮 交互模式[/bold cyan]\n")

    ensure_api_key()

    resolved_path = video_path
    if not resolved_path:
        path_obj = prompt_video_path()
        if path_obj is None:
            console.print("[bold red]✗ 未提供有效的视频文件路径，退出[/bold red]")
            raise SystemExit(1)
        resolved_path = str(path_obj)
    else:
        path_obj = Path(resolved_path).resolve()
        if not path_obj.exists():
            console.print(f"[bold red]✗ 文件不存在: {resolved_path}[/bold red]")
            path_obj = prompt_video_path()
            if path_obj is None:
                console.print("[bold red]✗ 未提供有效的视频文件路径，退出[/bold red]")
                raise SystemExit(1)
            resolved_path = str(path_obj)

    _execute_pipeline(resolved_path, output, asr_engine, visual_model, summary_model, config_path, only, resume)

def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    parts: list[str] = []
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        parts.append(f"{days}d")
    if days or hours:
        parts.append(f"{hours}h")
    if days or hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s" if parts else f"{seconds:.1f}s")
    return " ".join(parts)

def _execute_pipeline(
    video_path: str,
    output: str | None,
    asr_engine: str | None,
    visual_model: str | None,
    summary_model: str | None,
    config_path: str | None,
    only: str | None,
    resume: bool,
) -> None:
    """Set up context and run the video summarization pipeline."""
    cli_overrides: dict[str, str] = {}
    if asr_engine:
        cli_overrides["asr.engine"] = asr_engine
    if visual_model:
        cli_overrides["visual.model"] = visual_model
    if summary_model:
        cli_overrides["summary.model"] = summary_model

    env, os_name, has_cuda, has_mps, gpu_name, gpu_vram_mb, cpu_count, total_ram_gb = detect_system_info()

    config = load_config(
        config_path=Path(config_path) if config_path else None,
        cli_overrides=cli_overrides if cli_overrides else None,
        env=env,
    )

    log_level: str = config.get("logging", {}).get("level", "INFO")
    log_fmt: str = config.get("logging", {}).get(
        "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    setup_logging(level=log_level, fmt=log_fmt)
    logger = get_logger(__name__)

    if output:
        output_dir = Path(output).resolve()
    else:
        config_output_dir = config.get("output", {}).get("dir")
        output_dir = Path(config_output_dir).resolve() if config_output_dir else Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="video-summarizer-"))

    context = RuntimeContext(
        env=env,
        os_name=os_name,
        has_cuda=has_cuda,
        has_mps=has_mps,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        cpu_count=cpu_count,
        total_ram_gb=total_ram_gb,
        config=config,
        output_dir=output_dir,
        temp_dir=temp_dir,
    )

    logger.info(
        "Starting pipeline for %s (env=%s, output=%s)",
        video_path,
        env,
        output_dir,
    )

    try:
        start_time = time.monotonic()
        summary_path = run_pipeline(
            context=context,
            video_path=Path(video_path).resolve(),
            only=only,
            resume=resume,
        )
        elapsed = time.monotonic() - start_time
        console.print(f"\n[bold green]✓ Summary generated:[/bold green] {summary_path}")
        console.print(f"[bold green]✓ Total time:[/bold green] {_format_elapsed(elapsed)}")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        console.print(f"\n[bold red]✗ Pipeline failed:[/bold red] {exc}")
        raise SystemExit(1) from exc


@cli.command("env-info")
def env_info() -> None:
    """Display current environment information."""
    env, os_name, has_cuda, has_mps, gpu_name, gpu_vram_mb, cpu_count, total_ram_gb = detect_system_info()

    table = Table(title="Environment Information", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Environment", env)
    table.add_row("OS", os_name)
    table.add_row("CUDA Available", str(has_cuda))
    table.add_row("MPS Available", str(has_mps))

    if gpu_name:
        gpu_display = f"{gpu_name} ({gpu_vram_mb} MB)" if gpu_vram_mb else gpu_name
    else:
        gpu_display = "N/A"
    table.add_row("GPU", gpu_display)

    table.add_row("CPU Cores", str(cpu_count))
    table.add_row("RAM", f"{total_ram_gb} GB")

    console.print(table)


if __name__ == "__main__":
    cli()
