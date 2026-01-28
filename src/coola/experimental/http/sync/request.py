r"""Contain utility functions for synchronous HTTP requests with
automatic retry logic."""

from __future__ import annotations

__all__ = ["request_with_automatic_retry"]

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from coola.experimental.http.exception import HttpRequestError
from coola.utils.imports import is_httpx_available

if TYPE_CHECKING or is_httpx_available():
    import httpx
else:  # pragma: no cover
    from coola.utils.fallback.httpx import httpx

logger: logging.Logger = logging.getLogger(__name__)


def request_with_automatic_retry(
    url: str,
    method: str,
    request_func: Callable[..., httpx.Response],
    *,
    max_retries: int,
    backoff_factor: float,
    status_forcelist: tuple[int, ...],
    **kwargs: Any,
) -> httpx.Response:
    """Perform an HTTP request with automatic retry logic.

    Args:
        url: The URL to send the request to.
        method: The HTTP method name (e.g., "GET", "POST") for logging.
        request_func: The function to call to make the request (e.g.,
            client.get, client.post).
        max_retries: Maximum number of retry attempts for failed requests.
            Must be >= 0.
        backoff_factor: Factor for exponential backoff between retries. The wait
            time is calculated as: {backoff_factor} * (2 ** attempt) seconds,
            where attempt is 0-indexed (0, 1, 2, ...).
        status_forcelist: Tuple of HTTP status codes that should trigger a retry.
        **kwargs: Additional keyword arguments passed to the request function.

    Returns:
        An httpx.Response object containing the server's HTTP response.

    Raises:
        HttpRequestError: If the request times out, encounters network errors,
            or fails after exhausting all retries.
    """
    response: httpx.Response | None = None

    for attempt in range(max_retries + 1):
        try:
            response = request_func(url=url, **kwargs)

            # Success case
            if response.status_code < 400:
                if attempt > 0:
                    logger.debug(f"{method} request to {url} succeeded on attempt {attempt + 1}")
                return response

            # Non-retryable HTTP error
            if response.status_code not in status_forcelist:
                logger.debug(
                    f"{method} request to {url} failed with non-retryable status {response.status_code}"
                )
                response.raise_for_status()

            # Retryable HTTP status - log and continue
            logger.debug(
                f"{method} request to {url} failed with status {response.status_code} "
                f"(attempt {attempt + 1}/{max_retries + 1})"
            )

        except httpx.TimeoutException as exc:
            if attempt == max_retries:
                raise HttpRequestError(
                    method=method,
                    url=url,
                    message=f"{method} request to {url} timed out ({max_retries + 1} attempts)",
                    cause=exc,
                ) from exc

        except httpx.RequestError as exc:
            if attempt == max_retries:
                raise HttpRequestError(
                    method=method,
                    url=url,
                    message=f"{method} request to {url} failed after {max_retries + 1} attempts: {exc}",
                    cause=exc,
                ) from exc

        # Exponential backoff (skip on last attempt since we're about to fail)
        if attempt < max_retries:
            sleep_time = backoff_factor * (2**attempt)
            logger.debug(f"Waiting {sleep_time:.2f}s before retry")
            time.sleep(sleep_time)

    raise HttpRequestError(
        method=method,
        url=url,
        message=(
            f"{method} request to {url} failed with status "
            f"{response.status_code} after {max_retries + 1} attempts"
        ),
        status_code=response.status_code,
        response=response,
    )
