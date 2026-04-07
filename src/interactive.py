"""交互式模式处理模块：处理 API Key 提示、保存以及文件路径输入。"""

import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from dotenv import set_key
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

_API_KEY_ENV_VAR = "ZAI_API_KEY"


def resolve_project_root() -> Path:
    """解析项目根目录：向上查找直到找到 pyproject.toml 或使用当前工作目录。

    Returns:
        项目根目录路径。
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def get_env_file_path() -> Path:
    """获取 .env 文件路径（位于项目根目录）。

    Returns:
        .env 文件的 Path 对象。
    """
    return resolve_project_root() / ".env"


def load_dotenv_if_exists() -> bool:
    """加载项目根目录下的 .env 文件（如果存在）。

    Returns:
        True 如果 .env 文件存在且被加载，否则 False。
    """
    env_file = get_env_file_path()
    if env_file.exists():
        _ = load_dotenv(env_file, override=False)
        logger.debug("Loaded .env file from: %s", env_file)
        return True
    return False


def get_api_key_source() -> str | None:
    """检查当前 ZAI_API_KEY 的来源。

    优先级：
    1. 系统环境变量（已存在于 os.environ 中，非 .env 文件加载的）
    2. .env 文件加载的环境变量

    Returns:
        "system_env" — 来自系统环境变量
        "dotenv_file" — 来自 .env 文件
        None — 未找到 API Key
    """
    api_key = os.environ.get(_API_KEY_ENV_VAR)
    if not api_key:
        return None

    env_file = get_env_file_path()
    if env_file.exists():
        with env_file.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(f"{_API_KEY_ENV_VAR}="):
                    return "dotenv_file"

    return "system_env"


def ensure_api_key() -> str | None:
    """交互式确保 ZAI_API_KEY 可用。

    流程：
    1. 先尝试加载 .env 文件
    2. 检查 API Key 来源并记录日志
    3. 如果没有 API Key，提示用户输入
    4. 输入后询问是否保存到 .env 文件

    Returns:
        API Key 字符串，如果用户未提供则返回 None。
    """
    _ = load_dotenv_if_exists()
    source = get_api_key_source()

    if source is not None:
        source_display = "系统环境变量" if source == "system_env" else ".env 文件"
        logger.info("ZAI_API_KEY 已通过 %s 获取", source_display)
        console.print(f"[bold green]✓[/bold green] ZAI_API_KEY 已通过 [cyan]{source_display}[/cyan] 获取")
        return os.environ.get(_API_KEY_ENV_VAR)

    console.print("\n[bold yellow]⚠ 未检测到 ZAI_API_KEY[/bold yellow]")
    console.print("智谱 API Key 是使用视觉分析和 AI 总结功能的必要条件。\n")

    raw_api_key: str = click.prompt(
        "请输入 ZAI_API_KEY",
        type=str,
        default="",
        show_default=False,
    )

    if not raw_api_key or not raw_api_key.strip():
        console.print("[bold red]✗ 未提供 API Key，部分功能可能不可用[/bold red]")
        return None

    api_key = raw_api_key.strip()
    os.environ[_API_KEY_ENV_VAR] = api_key

    save: bool = click.confirm(
        "是否将 API Key 保存到 .env 文件以便下次自动加载？",
        default=True,
    )

    if save:
        save_api_key_to_dotenv(api_key)

    return api_key


def save_api_key_to_dotenv(api_key: str) -> None:
    """将 API Key 写入项目根目录的 .env 文件。

    Args:
        api_key: 要保存的 API Key 字符串。
    """
    env_file = get_env_file_path()

    if not env_file.exists():
        _ = env_file.write_text(f"{_API_KEY_ENV_VAR}={api_key}\n", encoding="utf-8")
    else:
        _ = set_key(str(env_file), _API_KEY_ENV_VAR, api_key, quote_mode="never")

    console.print(f"[bold green]✓[/bold green] API Key 已保存到 [cyan]{env_file}[/cyan]")
    logger.info("ZAI_API_KEY saved to .env file: %s", env_file)


def prompt_video_path() -> Path | None:
    """交互式提示用户输入视频文件路径。

    Returns:
        视频文件路径的 Path 对象，如果用户未输入或文件不存在则返回 None。
    """
    raw_path: str = click.prompt(
        "请输入视频文件路径",
        type=str,
        default="",
        show_default=False,
    )

    if not raw_path or not raw_path.strip():
        console.print("[bold red]✗ 未提供视频文件路径[/bold red]")
        return None

    video_path = Path(raw_path.strip()).expanduser().resolve()

    if not video_path.exists():
        console.print(f"[bold red]✗ 文件不存在: {video_path}[/bold red]")
        return None

    if not video_path.is_file():
        console.print(f"[bold red]✗ 不是文件: {video_path}[/bold red]")
        return None

    return video_path
