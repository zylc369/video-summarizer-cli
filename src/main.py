"""CLI entry point: provides summarize and env-info commands."""

import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.config_loader import load_config
from src.context import RuntimeContext
from src.env_detector import detect_system_info
from src.pipeline import run_pipeline
from src.utils.logger import get_logger, setup_logging

console = Console()


@click.group()
def cli() -> None:
    """Video Summarizer CLI — AI-powered video summarization tool."""


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
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
    video_path: str,
    output: str | None,
    asr_engine: str | None,
    visual_model: str | None,
    summary_model: str | None,
    config_path: str | None,
    only: str | None,
    resume: bool,
) -> None:
    """Summarize a video file using AI."""
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
        summary_path = run_pipeline(
            context=context,
            video_path=Path(video_path).resolve(),
            only=only,
            resume=resume,
        )
        console.print(f"\n[bold green]✓ Summary generated:[/bold green] {summary_path}")
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
