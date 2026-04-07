"""配置加载模块：支持 4 层优先级合并（代码默认值 < YAML < 环境覆盖 < CLI 参数）。"""

import copy
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict = {
    "asr": {
        "engine": "faster_whisper",
        "engines": {
            "faster_whisper": {
                "model": "large-v3-turbo",
                "compute_type": "float16",
                "device": "auto",
                "task": "transcribe",
                "language": None,
                "env_overrides": {
                    "mac": {"device": "cpu", "compute_type": "int8"},
                    "windows_gpu": {"device": "cuda", "compute_type": "float16"},
                },
            },
            "qwen3_asr": {
                "model": "Qwen/Qwen3-ASR-1.7B",
                "quantization": "q4",
                "device": "auto",
                "backend": "vllm",
                "gpu_memory_utilization": 0.8,
                "env_overrides": {
                    "mac": {"backend": "omlx", "quantization": "q8"},
                    "windows_gpu": {"backend": "vllm", "quantization": "q4"},
                },
            },
            "mlx_whisper": {
                "model": "large-v3-turbo",
                "language": None,
                "task": "transcribe",
                "env_overrides": {
                    "mac": {"model": "large-v3-turbo"},
                    "windows_gpu": {"enabled": False},
                },
            },
        },
    },
    "keyframe": {
        "method": "keyframe",
        "format": "png",
        "dedup_threshold": 0.95,
        "max_frames": 500,
    },
    "visual": {
        "model": "glm-4.6v-flashx",
        "api_key": None,
        "max_tokens": 2000,
        "concurrency": 5,
        "timeout": 30,
        "retry": 3,
    },
    "summary": {
        "model": "glm-4.7",
        "api_key": None,
        "max_tokens": 8000,
        "timeout": 120,
        "retry": 3,
        "prompt_template": None,
    },
    "output": {
        "dir": None,
        "include_screenshots": True,
        "screenshot_rel_path": "./keyframes/",
        "keep_intermediate": True,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归深度合并两个字典，override 中的值覆盖 base 中的同名键。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_project_root() -> Path:
    """解析项目根目录：向上查找直到找到 pyproject.toml 或使用当前工作目录。"""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_yaml_config(config_path: Path | None = None) -> dict:
    """加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径。如果为 None，则在项目根目录下查找 config/config.yml。

    Returns:
        解析后的配置字典。如果文件不存在则返回空字典。
    """
    if config_path is None:
        project_root = _resolve_project_root()
        config_path = project_root / "config" / "config.yml"

    if not config_path.exists():
        logger.debug("Config file not found: %s, using defaults", config_path)
        return {}

    logger.debug("Loading config from: %s", config_path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    return data


def _merge_env_overrides(config: dict, env: str) -> dict:
    """递归合并环境覆盖配置。

    遍历配置字典，查找所有包含 "env_overrides" 键的子字典，
    将指定环境的覆盖值合并到父级字典中，然后移除 "env_overrides" 键。

    Args:
        config: 配置字典。
        env: 环境标识（如 "mac"、"windows_gpu"）。

    Returns:
        合并后的配置字典。
    """
    result = copy.deepcopy(config)

    def _walk(d: dict) -> dict:
        merged: dict[str, object] = {}
        for key, value in d.items():
            if isinstance(value, dict):
                value = _walk(value)
            merged[key] = value

        if "env_overrides" in merged:
            env_overrides = merged["env_overrides"]
            if isinstance(env_overrides, dict) and env in env_overrides:
                override_values = env_overrides[env]
                if isinstance(override_values, dict):
                    for k, v in override_values.items():
                        merged[k] = v
            del merged["env_overrides"]

        return merged

    return _walk(result)


def apply_cli_overrides(config: dict, overrides: dict) -> dict:
    """将 CLI 参数覆盖应用到配置字典。

    支持点分隔的键路径，如 "asr.engine"、"visual.model"。
    仅更新 overrides 中非 None 的值。

    Args:
        config: 当前配置字典。
        overrides: CLI 覆盖值，键可以为点分隔路径。

    Returns:
        更新后的配置字典。
    """
    result = copy.deepcopy(config)

    for key_path, value in overrides.items():
        if value is None:
            continue

        parts = key_path.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return result


def load_config(
    config_path: Path | None = None,
    cli_overrides: dict | None = None,
    env: str | None = None,
) -> dict:
    """主配置加载入口，按优先级链式合并配置。

    优先级（从低到高）：
    1. 代码默认值 (DEFAULT_CONFIG)
    2. YAML 配置文件
    3. 环境覆盖 (env_overrides)
    4. CLI 参数覆盖
    5. 环境变量 (ZHIPUAI_API_KEY)

    Args:
        config_path: YAML 配置文件路径。None 表示使用默认路径。
        cli_overrides: CLI 参数覆盖。None 表示无覆盖。
        env: 运行环境标识（如 "mac"、"windows_gpu"）。None 表示不应用环境覆盖。

    Returns:
        完整合并后的配置字典。
    """
    # Step 1: 从默认配置开始
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Step 2: 加载 YAML 并深度合并
    yaml_config = load_yaml_config(config_path)
    if yaml_config:
        config = _deep_merge(config, yaml_config)

    # Step 3: 应用环境覆盖
    if env is not None:
        config = _merge_env_overrides(config, env)

    # Step 4: 应用 CLI 覆盖
    if cli_overrides:
        config = apply_cli_overrides(config, cli_overrides)

    # Step 5: 从环境变量设置 API Key
    api_key = os.environ.get("ZAI_API_KEY")
    if api_key:
        config["visual"]["api_key"] = api_key
        config["summary"]["api_key"] = api_key

    return config
