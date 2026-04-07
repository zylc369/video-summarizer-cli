"""interactive 模块测试：覆盖 API Key 提示、.env 文件加载和视频路径输入。"""

import os
from pathlib import Path
from unittest.mock import patch

from src.interactive import ensure_api_key
from src.interactive import get_api_key_source
from src.interactive import get_env_file_path
from src.interactive import load_dotenv_if_exists
from src.interactive import prompt_video_path
from src.interactive import save_api_key_to_dotenv


class TestGetEnvFilePath:
    def test_get_env_file_path_returns_dotenv_in_project_root(self) -> None:
        with patch("src.interactive.resolve_project_root", return_value=Path("/fake/project")):
            result = get_env_file_path()

        assert result == Path("/fake/project/.env")


class TestLoadDotenvIfExists:
    def test_load_dotenv_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=hello\n", encoding="utf-8")

        with patch("src.interactive.get_env_file_path", return_value=env_file):
            result = load_dotenv_if_exists()

        assert result is True
        assert os.environ.get("TEST_VAR") == "hello"
        os.environ.pop("TEST_VAR", None)

    def test_load_dotenv_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent" / ".env"

        with patch("src.interactive.get_env_file_path", return_value=missing):
            result = load_dotenv_if_exists()

        assert result is False

    def test_load_dotenv_does_not_override_existing_env(self, tmp_path: Path) -> None:
        os.environ["EXISTING_VAR"] = "original"
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=from_dotenv\n", encoding="utf-8")

        with patch("src.interactive.get_env_file_path", return_value=env_file):
            load_dotenv_if_exists()

        assert os.environ["EXISTING_VAR"] == "original"
        os.environ.pop("EXISTING_VAR", None)


class TestGetApiKeySource:
    def test_returns_none_when_no_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.interactive.get_env_file_path", return_value=Path("/fake/.env")):
                result = get_api_key_source()

        assert result is None

    def test_returns_system_env_when_no_dotenv_file(self) -> None:
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=False):
            with patch("src.interactive.get_env_file_path", return_value=Path("/fake/.env")):
                with patch.object(Path, "exists", return_value=False):
                    result = get_api_key_source()

        assert result == "system_env"

    def test_returns_dotenv_file_when_key_in_dotenv(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("ZAI_API_KEY=from-dotenv\n", encoding="utf-8")

        with patch.dict(os.environ, {"ZAI_API_KEY": "from-dotenv"}, clear=False):
            with patch("src.interactive.get_env_file_path", return_value=env_file):
                result = get_api_key_source()

        assert result == "dotenv_file"

    def test_returns_system_env_when_key_not_in_dotenv(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_VAR=value\n", encoding="utf-8")

        with patch.dict(os.environ, {"ZAI_API_KEY": "from-system"}, clear=False):
            with patch("src.interactive.get_env_file_path", return_value=env_file):
                result = get_api_key_source()

        assert result == "system_env"


class TestEnsureApiKey:
    def test_returns_existing_key_from_system_env(self) -> None:
        with patch.dict(os.environ, {"ZAI_API_KEY": "sys-key"}, clear=False):
            with patch("src.interactive.load_dotenv_if_exists", return_value=False):
                with patch("src.interactive.get_api_key_source", return_value="system_env"):
                    result = ensure_api_key()

        assert result == "sys-key"

    def test_returns_existing_key_from_dotenv(self) -> None:
        with patch.dict(os.environ, {"ZAI_API_KEY": "dotenv-key"}, clear=False):
            with patch("src.interactive.load_dotenv_if_exists", return_value=True):
                with patch("src.interactive.get_api_key_source", return_value="dotenv_file"):
                    result = ensure_api_key()

        assert result == "dotenv-key"

    def test_prompts_and_sets_key_without_saving(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.interactive.load_dotenv_if_exists", return_value=False):
                with patch("src.interactive.get_api_key_source", return_value=None):
                    with patch("click.prompt", return_value="new-api-key"):
                        with patch("click.confirm", return_value=False):
                            result = ensure_api_key()

                            assert result == "new-api-key"
                            assert os.environ.get("ZAI_API_KEY") == "new-api-key"

    def test_prompts_and_saves_key_to_dotenv(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.interactive.load_dotenv_if_exists", return_value=False):
                with patch("src.interactive.get_api_key_source", return_value=None):
                    with patch("click.prompt", return_value="saved-key"):
                        with patch("click.confirm", return_value=True):
                            with patch("src.interactive.save_api_key_to_dotenv") as mock_save:
                                result = ensure_api_key()

        assert result == "saved-key"
        mock_save.assert_called_once_with("saved-key")

    def test_returns_none_on_empty_input(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.interactive.load_dotenv_if_exists", return_value=False):
                with patch("src.interactive.get_api_key_source", return_value=None):
                    with patch("click.prompt", return_value=""):
                        result = ensure_api_key()

        assert result is None


class TestSaveApiKeyToDotenv:
    def test_creates_new_dotenv_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"

        with patch("src.interactive.get_env_file_path", return_value=env_file):
            save_api_key_to_dotenv("my-secret-key")

        assert env_file.exists()
        content = env_file.read_text(encoding="utf-8")
        assert "ZAI_API_KEY=my-secret-key" in content

    def test_updates_existing_dotenv_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_VAR=value\n", encoding="utf-8")

        with patch("src.interactive.get_env_file_path", return_value=env_file):
            save_api_key_to_dotenv("updated-key")

        content = env_file.read_text(encoding="utf-8")
        assert "ZAI_API_KEY=" in content
        assert "updated-key" in content
        assert "OTHER_VAR=value" in content


class TestPromptVideoPath:
    def test_returns_resolved_path_for_existing_file(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.write_text("fake video", encoding="utf-8")

        with patch("click.prompt", return_value=str(video)):
            result = prompt_video_path()

        assert result == video.resolve()
        assert result is not None and result.exists()

    def test_returns_none_on_empty_input(self) -> None:
        with patch("click.prompt", return_value=""):
            result = prompt_video_path()

        assert result is None

    def test_returns_none_for_nonexistent_file(self) -> None:
        with patch("click.prompt", return_value="/nonexistent/video.mp4"):
            result = prompt_video_path()

        assert result is None

    def test_returns_none_for_directory(self, tmp_path: Path) -> None:
        with patch("click.prompt", return_value=str(tmp_path)):
            result = prompt_video_path()

        assert result is None

    def test_expands_tilde_in_path(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.write_text("fake video", encoding="utf-8")

        with patch("click.prompt", return_value=str(video)):
            with patch("pathlib.Path.expanduser", return_value=video):
                result = prompt_video_path()

        assert result is not None
