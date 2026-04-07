"""Zhipu AI API client wrapper with retry, timeout, and error handling."""

import time
from collections.abc import Callable

import zai
import zai.core
from zai import ZhipuAiClient

from src.context import SummarizerError, VisualAnalysisError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Exceptions that are retryable (transient failures)
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    zai.core.APIReachLimitError,
    zai.core.APIInternalError,
    zai.core.APITimeoutError,
    zai.core.APIRequestFailedError,
)

# Exceptions that should NOT be retried (client errors)
_NON_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    zai.core.APIAuthenticationError,
)


class ZhipuClient:
    """Wrapper around the ZhipuAiClient SDK with retry logic and error handling."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the ZhipuAiClient.

        Args:
            api_key: ZhipuAI API key. If None, the SDK reads ZAI_API_KEY
                     from the environment.
        """
        self._client: ZhipuAiClient = ZhipuAiClient(api_key=api_key)

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, object]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> str:
        """Send a text chat completion request to the ZhipuAI API.

        Args:
            model: Model identifier (e.g. "glm-4.7").
            messages: List of message dicts with "role" and "content" keys.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0.0 - 1.0).
            timeout: Request timeout in seconds. None uses SDK default.

        Returns:
            The response text from the model.

        Raises:
            SummarizerError: On API failure after retries exhausted.
        """
        def _call() -> str:
            kwargs: dict[str, object] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if timeout is not None:
                kwargs["timeout"] = timeout

            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return content if content is not None else ""

        return self._call_with_retry(_call, error_cls=SummarizerError)

    def vision_analysis(
        self,
        model: str,
        image_base64: str,
        prompt: str,
        max_tokens: int = 2000,
        timeout: float | None = None,
    ) -> str:
        """Send a vision/multimodal analysis request to the ZhipuAI API.

        Args:
            model: Vision model identifier (e.g. "glm-4.6v-flashx").
            image_base64: Base64-encoded image data (raw base64, no data URI prefix).
            prompt: Text prompt describing what to analyze in the image.
            max_tokens: Maximum tokens in the response.
            timeout: Request timeout in seconds. None uses SDK default.

        Returns:
            The response text from the model.

        Raises:
            VisualAnalysisError: On API failure after retries exhausted.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]

        def _call() -> str:
            kwargs: dict[str, object] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if timeout is not None:
                kwargs["timeout"] = timeout

            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return content if content is not None else ""

        return self._call_with_retry(_call, error_cls=VisualAnalysisError)

    def _call_with_retry(
        self,
        func: Callable[[], str],
        max_retries: int = 3,
        base_delay: float = 1.0,
        error_cls: type[SummarizerError | VisualAnalysisError] = SummarizerError,
    ) -> str:
        """Execute a function with exponential backoff retry on transient errors.

        Retries on rate limits, server errors, timeouts, and connection errors.
        Does NOT retry on authentication errors or 4xx client errors.

        Args:
            func: Callable that performs the API request and returns response text.
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay in seconds for exponential backoff.
            error_cls: Exception class to raise on final failure.

        Returns:
            The response text from the successful API call.

        Raises:
            error_cls: On API failure after all retries exhausted, or on
                       non-retryable errors.
        """
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                return func()
            except _NON_RETRYABLE_ERRORS as e:
                logger.error(
                    "Non-retryable API error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                raise error_cls(f"API authentication error: {e}") from e
            except zai.core.APIReachLimitError as e:
                last_error = e
                logger.warning(
                    "Rate limited (HTTP %s, attempt %d/%d): %s",
                    getattr(e, "status_code", "?"),
                    attempt + 1,
                    max_retries,
                    e,
                )
            except zai.core.APIInternalError as e:
                last_error = e
                logger.warning(
                    "Server error (HTTP %s, attempt %d/%d): %s",
                    getattr(e, "status_code", "?"),
                    attempt + 1,
                    max_retries,
                    e,
                )
            except zai.core.APITimeoutError as e:
                last_error = e
                logger.warning(
                    "Request timeout (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
            except zai.core.APIRequestFailedError as e:
                last_error = e
                logger.warning(
                    "Connection error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
            except zai.core.APIStatusError as e:
                status_code = getattr(e, "status_code", None)
                if isinstance(status_code, int) and 400 <= status_code < 500:
                    logger.error(
                        "Client API error (HTTP %d, attempt %d/%d): %s",
                        status_code,
                        attempt + 1,
                        max_retries,
                        e,
                    )
                    raise error_cls(f"API client error (HTTP {status_code}): {e}") from e
                last_error = e
                logger.warning(
                    "Retryable API error (HTTP %d, attempt %d/%d): %s",
                    status_code,
                    attempt + 1,
                    max_retries,
                    e,
                )

            if attempt < max_retries - 1:
                delay = (2 ** attempt) * base_delay
                logger.info("Retrying in %.1f seconds...", delay)
                time.sleep(delay)

        raise error_cls(
            f"API call failed after {max_retries} attempts: {last_error}"
        ) from last_error


def create_zhipu_client(api_key: str | None = None) -> ZhipuClient:
    """Factory function to create a ZhipuClient instance.

    Args:
        api_key: ZhipuAI API key. If None, the SDK reads ZAI_API_KEY
                 from the environment.

    Returns:
        A configured ZhipuClient instance.
    """
    return ZhipuClient(api_key=api_key)
