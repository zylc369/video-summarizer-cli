"""config_loader 模块测试：覆盖 YAML 加载、环境覆盖、CLI 覆盖及完整链式合并。"""

import copy
import os
from pathlib import Path
from unittest.mock import mock_open
from unittest.mock import patch

import pytest
import yaml

from src.config_loader import DEFAULT_CONFIG
from src.config_loader import _deep_merge
from src.config_loader import _merge_env_overrides
from src.config_loader import apply_cli_overrides
from src.config_loader import load_config
from src.config_loader import load_yaml_config


class TestLoadYamlConfig:
    def test_load_yaml_config_reads_file(self, tmp_path: Path) -> None:
        yaml_content = yaml.dump({"asr": {"engine": "mlx_whisper"}})
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml_content, encoding="utf-8")

        result = load_yaml_config(config_file)

        assert result == {"asr": {"engine": "mlx_whisper"}}

    def test_load_yaml_config_missing_file_returns_empty(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "nonexistent" / "config.yml"

        result = load_yaml_config(missing_path)

        assert result == {}

    def test_load_yaml_config_none_path_uses_default(self) -> None:
        yaml_content = yaml.dump({"asr": {"engine": "faster_whisper"}})
        fake_path = Path("/fake/project/config/config.yml")

        with patch("src.config_loader._resolve_project_root", return_value=Path("/fake/project")):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "open", mock_open(read_data=yaml_content)):
                    result = load_yaml_config(None)

        assert "asr" in result
        assert result["asr"]["engine"] == "faster_whisper"

    def test_load_yaml_config_empty_file_returns_empty(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.yml"
        empty_file.write_text("", encoding="utf-8")

        result = load_yaml_config(empty_file)

        assert result == {}


class TestMergeEnvOverrides:
    def test_merge_env_overrides_applies_mac_overrides(self) -> None:
        config = {
            "device": "auto",
            "compute_type": "float16",
            "env_overrides": {
                "mac": {"device": "cpu", "compute_type": "int8"},
                "windows_gpu": {"device": "cuda", "compute_type": "float16"},
            },
        }

        result = _merge_env_overrides(config, "mac")

        assert result["device"] == "cpu"
        assert result["compute_type"] == "int8"
        assert "env_overrides" not in result

    def test_merge_env_overrides_applies_windows_gpu_overrides(self) -> None:
        config = {
            "device": "auto",
            "compute_type": "float16",
            "env_overrides": {
                "mac": {"device": "cpu", "compute_type": "int8"},
                "windows_gpu": {"device": "cuda", "compute_type": "float16"},
            },
        }

        result = _merge_env_overrides(config, "windows_gpu")

        assert result["device"] == "cuda"
        assert result["compute_type"] == "float16"
        assert "env_overrides" not in result

    def test_merge_env_overrides_removes_env_overrides_key(self) -> None:
        config = {
            "device": "auto",
            "env_overrides": {
                "mac": {"device": "cpu"},
            },
        }

        result = _merge_env_overrides(config, "mac")

        assert "env_overrides" not in result
        assert result["device"] == "cpu"

    def test_merge_env_overrides_nested(self) -> None:
        config = {
            "asr": {
                "engines": {
                    "faster_whisper": {
                        "device": "auto",
                        "compute_type": "float16",
                        "env_overrides": {
                            "mac": {"device": "cpu", "compute_type": "int8"},
                        },
                    },
                    "mlx_whisper": {
                        "model": "large-v3-turbo",
                        "env_overrides": {
                            "mac": {"model": "large-v3"},
                        },
                    },
                },
            },
        }

        result = _merge_env_overrides(config, "mac")

        assert result["asr"]["engines"]["faster_whisper"]["device"] == "cpu"
        assert result["asr"]["engines"]["faster_whisper"]["compute_type"] == "int8"
        assert "env_overrides" not in result["asr"]["engines"]["faster_whisper"]
        assert result["asr"]["engines"]["mlx_whisper"]["model"] == "large-v3"
        assert "env_overrides" not in result["asr"]["engines"]["mlx_whisper"]

    def test_merge_env_overrides_no_env_overrides(self) -> None:
        config = {
            "keyframe": {
                "method": "keyframe",
                "format": "png",
            },
        }

        result = _merge_env_overrides(config, "mac")

        assert result == config

    def test_merge_env_overrides_unknown_env_keeps_defaults(self) -> None:
        config = {
            "device": "auto",
            "env_overrides": {
                "mac": {"device": "cpu"},
            },
        }

        result = _merge_env_overrides(config, "unknown_env")

        assert result["device"] == "auto"
        assert "env_overrides" not in result

    def test_merge_env_overrides_does_not_mutate_input(self) -> None:
        config = {
            "device": "auto",
            "env_overrides": {
                "mac": {"device": "cpu"},
            },
        }
        original = {"device": "auto", "env_overrides": {"mac": {"device": "cpu"}}}

        _merge_env_overrides(config, "mac")

        assert config == original


class TestApplyCliOverrides:
    def test_apply_cli_overrides_simple(self) -> None:
        config = {"asr": {"engine": "faster_whisper"}, "keyframe": {"method": "keyframe"}}

        result = apply_cli_overrides(config, {"asr.engine": "mlx_whisper"})

        assert result["asr"]["engine"] == "mlx_whisper"
        assert result["keyframe"]["method"] == "keyframe"

    def test_apply_cli_overrides_dotted_path(self) -> None:
        config = {
            "asr": {
                "engines": {
                    "faster_whisper": {"model": "large-v3-turbo", "device": "auto"},
                },
            },
        }

        result = apply_cli_overrides(config, {"asr.engines.faster_whisper.model": "small"})

        assert result["asr"]["engines"]["faster_whisper"]["model"] == "small"
        assert result["asr"]["engines"]["faster_whisper"]["device"] == "auto"

    def test_apply_cli_overrides_skips_none(self) -> None:
        config = {"asr": {"engine": "faster_whisper"}, "visual": {"model": "glm-4.6v-flashx"}}

        result = apply_cli_overrides(
            config,
            {"asr.engine": "mlx_whisper", "visual.model": None, "visual.timeout": None},
        )

        assert result["asr"]["engine"] == "mlx_whisper"
        assert result["visual"]["model"] == "glm-4.6v-flashx"
        assert "timeout" not in result["visual"]

    def test_apply_cli_overrides_creates_nested_path(self) -> None:
        config: dict = {"asr": {}}

        result = apply_cli_overrides(config, {"asr.engines.custom.model": "test-model"})

        assert result["asr"]["engines"]["custom"]["model"] == "test-model"

    def test_apply_cli_overrides_does_not_mutate_input(self) -> None:
        config = {"asr": {"engine": "faster_whisper"}}

        apply_cli_overrides(config, {"asr.engine": "mlx_whisper"})

        assert config["asr"]["engine"] == "faster_whisper"


class TestLoadConfig:
    def test_load_config_returns_defaults_when_no_file(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "nonexistent" / "config.yml"

        result = load_config(config_path=missing_path)

        assert result["asr"]["engine"] == DEFAULT_CONFIG["asr"]["engine"]
        assert result["keyframe"]["method"] == DEFAULT_CONFIG["keyframe"]["method"]
        assert result["visual"]["model"] == DEFAULT_CONFIG["visual"]["model"]

    def test_load_config_yaml_overrides_defaults(self, tmp_path: Path) -> None:
        yaml_content = yaml.dump({
            "asr": {"engine": "mlx_whisper"},
            "visual": {"concurrency": 10},
        })
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml_content, encoding="utf-8")

        result = load_config(config_path=config_file)

        assert result["asr"]["engine"] == "mlx_whisper"
        assert result["visual"]["concurrency"] == 10
        assert result["keyframe"]["method"] == DEFAULT_CONFIG["keyframe"]["method"]

    def test_load_config_full_chain(self, tmp_path: Path) -> None:
        yaml_content = yaml.dump({
            "asr": {"engine": "faster_whisper"},
            "visual": {"concurrency": 3},
        })
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml_content, encoding="utf-8")

        result = load_config(
            config_path=config_file,
            cli_overrides={"asr.engine": "mlx_whisper"},
            env="mac",
        )

        assert result["asr"]["engine"] == "mlx_whisper"
        assert "env_overrides" not in result["asr"]["engines"]["faster_whisper"]
        assert result["asr"]["engines"]["faster_whisper"]["device"] == "cpu"
        assert result["visual"]["concurrency"] == 3

    def test_load_config_env_var_api_key(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "nonexistent" / "config.yml"

        with patch.dict(os.environ, {"ZAI_API_KEY": "test-api-key-123"}):
            result = load_config(config_path=missing_path)

        assert result["visual"]["api_key"] == "test-api-key-123"
        assert result["summary"]["api_key"] == "test-api-key-123"

    def test_load_config_no_env_var_leaves_api_key_none(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "nonexistent" / "config.yml"

        with patch.dict(os.environ, {}, clear=True):
            result = load_config(config_path=missing_path)

        assert result["visual"]["api_key"] is None
        assert result["summary"]["api_key"] is None

    def test_load_config_env_overrides_applied(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "nonexistent" / "config.yml"

        result = load_config(config_path=missing_path, env="mac")

        assert "env_overrides" not in result["asr"]["engines"]["faster_whisper"]
        assert result["asr"]["engines"]["faster_whisper"]["device"] == "cpu"
        assert result["asr"]["engines"]["faster_whisper"]["compute_type"] == "int8"

    def test_load_config_does_not_mutate_defaults(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "nonexistent" / "config.yml"

        original_env_overrides = copy.deepcopy(
            DEFAULT_CONFIG["asr"]["engines"]["faster_whisper"]["env_overrides"]
        )

        load_config(config_path=missing_path, env="mac")

        assert DEFAULT_CONFIG["asr"]["engines"]["faster_whisper"]["env_overrides"] == original_env_overrides


class TestDeepMerge:
    def test_deep_merge_nested_dicts(self) -> None:
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10, "e": 5}, "f": 6}

        result = _deep_merge(base, override)

        assert result == {"a": {"b": 10, "c": 2, "e": 5}, "d": 3, "f": 6}

    def test_deep_merge_override_replaces_non_dict(self) -> None:
        base = {"a": "string"}
        override = {"a": {"nested": True}}

        result = _deep_merge(base, override)

        assert result == {"a": {"nested": True}}

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}

        result = _deep_merge(base, override)

        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}
        assert result == {"a": {"b": 1, "c": 2}}
