"""Tests for Zhipu AI API client wrapper."""

from unittest.mock import Mock, patch

import pytest
import zai.core

from src.context import SummarizerError, VisualAnalysisError
from src.utils.zhipu_client import ZhipuClient, create_zhipu_client


def _make_mock_response(content: str = "test response") -> Mock:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = content
    return mock_response


def _make_mock_http_response(status_code: int = 500) -> Mock:
    mock_resp = Mock()
    mock_resp.status_code = status_code
    return mock_resp


@pytest.fixture
def mock_zhipuai_client() -> Mock:
    with patch("src.utils.zhipu_client.ZhipuAiClient") as mock_cls:
        mock_instance = Mock()
        mock_cls.return_value = mock_instance
        yield mock_instance


class TestChatCompletion:
    def test_chat_completion_returns_text(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.return_value = _make_mock_response("Hello world")

        client = ZhipuClient(api_key="test-key")
        result = client.chat_completion(model="glm-4.7", messages=[{"role": "user", "content": "Hi"}])

        assert result == "Hello world"

    def test_chat_completion_passes_correct_params(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.return_value = _make_mock_response()

        client = ZhipuClient(api_key="test-key")
        client.chat_completion(
            model="glm-4.7",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=2048,
            temperature=0.5,
            timeout=30.0,
        )

        call_kwargs = mock_zhipuai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "glm-4.7"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["timeout"] == 30.0

    def test_chat_completion_omits_timeout_when_none(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.return_value = _make_mock_response()

        client = ZhipuClient(api_key="test-key")
        client.chat_completion(model="glm-4.7", messages=[{"role": "user", "content": "Hi"}])

        call_kwargs = mock_zhipuai_client.chat.completions.create.call_args[1]
        assert "timeout" not in call_kwargs


class TestVisionAnalysis:
    def test_vision_analysis_returns_text(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.return_value = _make_mock_response("A cat sitting on a desk")

        client = ZhipuClient(api_key="test-key")
        result = client.vision_analysis(
            model="glm-4.6v-flashx",
            image_base64="base64data",
            prompt="Describe this image",
        )

        assert result == "A cat sitting on a desk"

    def test_vision_analysis_builds_correct_messages(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.return_value = _make_mock_response()

        client = ZhipuClient(api_key="test-key")
        client.vision_analysis(
            model="glm-4.6v-flashx",
            image_base64="abc123",
            prompt="What do you see?",
        )

        call_kwargs = mock_zhipuai_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "What do you see?"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,abc123"


class TestRetryLogic:
    def test_retry_on_rate_limit(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.side_effect = [
            zai.core.APIReachLimitError(message="rate limited", response=_make_mock_http_response(429)),
            zai.core.APIReachLimitError(message="rate limited", response=_make_mock_http_response(429)),
            _make_mock_response("success after retries"),
        ]

        client = ZhipuClient(api_key="test-key")
        with patch("src.utils.zhipu_client.time.sleep"):
            result = client.chat_completion(
                model="glm-4.7",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result == "success after retries"
        assert mock_zhipuai_client.chat.completions.create.call_count == 3

    def test_retry_on_timeout(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.side_effect = [
            zai.core.APITimeoutError(request=Mock()),
            _make_mock_response("recovered"),
        ]

        client = ZhipuClient(api_key="test-key")
        with patch("src.utils.zhipu_client.time.sleep"):
            result = client.chat_completion(
                model="glm-4.7",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result == "recovered"
        assert mock_zhipuai_client.chat.completions.create.call_count == 2

    def test_no_retry_on_auth_error(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.side_effect = (
            zai.core.APIAuthenticationError(
                message="invalid api key",
                response=_make_mock_http_response(401),
            )
        )

        client = ZhipuClient(api_key="test-key")
        with pytest.raises(SummarizerError) as exc_info:
            client.chat_completion(
                model="glm-4.7",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert "authentication error" in str(exc_info.value).lower()
        assert mock_zhipuai_client.chat.completions.create.call_count == 1

    def test_max_retries_exhausted_raises_error(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.side_effect = (
            zai.core.APIInternalError(message="server error", response=_make_mock_http_response(500))
        )

        client = ZhipuClient(api_key="test-key")
        with patch("src.utils.zhipu_client.time.sleep"):
            with pytest.raises(SummarizerError) as exc_info:
                client.chat_completion(
                    model="glm-4.7",
                    messages=[{"role": "user", "content": "Hi"}],
                )

        assert "failed after 3 attempts" in str(exc_info.value)
        assert mock_zhipuai_client.chat.completions.create.call_count == 3

    def test_retry_on_connection_error(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.side_effect = [
            zai.core.APIRequestFailedError(message="connection failed", response=_make_mock_http_response(502)),
            _make_mock_response("connected"),
        ]

        client = ZhipuClient(api_key="test-key")
        with patch("src.utils.zhipu_client.time.sleep"):
            result = client.chat_completion(
                model="glm-4.7",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result == "connected"

    def test_vision_analysis_uses_visual_analysis_error(self, mock_zhipuai_client: Mock) -> None:
        mock_zhipuai_client.chat.completions.create.side_effect = (
            zai.core.APIInternalError(message="server error", response=_make_mock_http_response(500))
        )

        client = ZhipuClient(api_key="test-key")
        with patch("src.utils.zhipu_client.time.sleep"):
            with pytest.raises(VisualAnalysisError):
                client.vision_analysis(
                    model="glm-4.6v-flashx",
                    image_base64="abc",
                    prompt="Describe",
                )


class TestCreateZhipuClient:
    def test_create_zhipu_client_factory(self) -> None:
        with patch("src.utils.zhipu_client.ZhipuAiClient"):
            client = create_zhipu_client(api_key="test-key")
            assert isinstance(client, ZhipuClient)

    def test_create_zhipu_client_factory_no_key(self) -> None:
        with patch("src.utils.zhipu_client.ZhipuAiClient"):
            client = create_zhipu_client()
            assert isinstance(client, ZhipuClient)
