"""核心上下文模块：定义运行时上下文和全局异常类型。"""

from dataclasses import dataclass
from pathlib import Path


class FFmpegNotFoundError(Exception):
    """FFmpeg 二进制文件未在系统中找到。"""


class FFmpegError(Exception):
    """FFmpeg 执行失败。"""


class ASREngineError(Exception):
    """ASR 引擎初始化或执行错误。"""


class VisualAnalysisError(Exception):
    """视觉分析（智谱视觉 API）错误。"""


class SummarizerError(Exception):
    """AI 总结生成错误。"""


class ConfigError(Exception):
    """配置加载或验证错误。"""


@dataclass
class RuntimeContext:
    """全局运行时上下文，程序启动时初始化一次。"""

    env: str
    os_name: str
    has_cuda: bool
    has_mps: bool
    gpu_name: str | None
    gpu_vram_mb: int | None
    cpu_count: int
    total_ram_gb: float
    config: dict
    output_dir: Path
    temp_dir: Path
