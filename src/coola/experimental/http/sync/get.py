r"""Contain synchronous HTTP GET request with automatic retry logic."""

from __future__ import annotations

__all__ = ["get_with_automatic_retry"]

import logging
import time
from typing import TYPE_CHECKING, Any

from coola.experimental.http.constants import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    RETRY_STATUS_CODES,
)
from coola.experimental.http.exception import HttpRequestError
from coola.utils.imports import check_httpx, is_httpx_available

if TYPE_CHECKING or is_httpx_available():
    import httpx
else:  # pragma: no cover
    from coola.utils.fallback.httpx import httpx

logger: logging.Logger = logging.getLogger(__name__)


def get_with_automatic_retry(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: tuple[int, ...] = RETRY_STATUS_CODES,
    **kwargs: Any,
) -> httpx.Response:
    r"""Send an HTTP GET request with automatic retry logic for transient
    errors.

    This function performs an HTTP GET request with a configured retry policy
    for transient server errors (429, 500, 502, 503, 504). It applies an
    exponential backoff retry strategy. The function validates the HTTP
    response and raises detailed errors for failures.

    Args:
        url: The URL to send the GET request to.
        client: An optional httpx.Client object to use for making requests.
            If None, a new client will be created and closed after use.
        timeout: Maximum seconds to wait for the server response.
            Only used if client is None.
        max_retries: Maximum number of retry attempts for failed requests.
            Must be >= 0.
        backoff_factor: Factor for exponential backoff between retries. The wait
            time is calculated as: {backoff_factor} * (2 ** retry_number) seconds.
            Must be >= 0.
        status_forcelist: Tuple of HTTP status codes that should trigger a retry.
        **kwargs: Additional keyword arguments passed to ``httpx.Client.get()``.

    Returns:
        An httpx.Response object containing the server's HTTP response.

    Raises:
        HttpRequestError: If the request times out, encounters network errors,
            or fails after exhausting all retries.
        ValueError: If max_retries or backoff_factor are negative.

    Example:
        ```pycon
        >>> from coola.experimental.http.sync import get_with_automatic_retry
        >>> response = get_with_automatic_retry(
        ...     "https://api.example.com/data", json={"key": "value"}
        ... )  # doctest: +SKIP
        >>> response.json()  # doctest: +SKIP
        ```
    """
    check_httpx()

    # Input validation
    if max_retries < 0:
        msg = f"max_retries must be >= 0, got {max_retries}"
        raise ValueError(msg)
    if backoff_factor < 0:
        msg = f"backoff_factor must be >= 0, got {backoff_factor}"
        raise ValueError(msg)

    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        return _get_with_automatic_retry(
            client=client,
            url=url,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            **kwargs,
        )
    finally:
        if owns_client:
            client.close()


def _get_with_automatic_retry(
    client: httpx.Client,
    url: str,
    *,
    max_retries: int,
    backoff_factor: float,
    status_forcelist: tuple[int, ...],
    **kwargs: Any,
) -> httpx.Response:
    """Define an internal function to perform GET request with retries.

    Args:
        client: A httpx.Client object to use for making requests.
        url: The URL to send the GET request to.
        max_retries: Maximum number of retry attempts for failed requests.
            Must be >= 0.
        backoff_factor: Factor for exponential backoff between retries. The wait
            time is calculated as: {backoff_factor} * (2 ** retry_number) seconds.
            Must be >= 0.
        status_forcelist: Tuple of HTTP status codes that should trigger a retry.
        **kwargs: Additional keyword arguments passed to ``httpx.Client.get()``.

    Returns:
        An httpx.Response object containing the server's HTTP response.

    Raises:
        HttpRequestError: If the request times out, encounters network errors,
            or fails after exhausting all retries.
    """
    response: httpx.Response | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.get(url=url, **kwargs)

            # Success case
            if response.status_code < 400:
                if attempt > 0:
                    logger.debug(f"GET request to {url} succeeded on attempt {attempt + 1}")
                return response

            # Non-retryable HTTP error
            if response.status_code not in status_forcelist:
                logger.debug(
                    f"GET request to {url} failed with non-retryable status {response.status_code}"
                )
                response.raise_for_status()

            # Retryable HTTP status - log and continue
            logger.debug(
                f"GET request to {url} failed with status {response.status_code} "
                f"(attempt {attempt + 1}/{max_retries + 1})"
            )

        except httpx.TimeoutException as exc:
            if attempt == max_retries:
                raise HttpRequestError(
                    method="GET",
                    url=url,
                    message=f"GET request to {url} timed out ({max_retries + 1} attempts)",
                    cause=exc,
                ) from exc

        except httpx.RequestError as exc:
            if attempt == max_retries:
                raise HttpRequestError(
                    method="GET",
                    url=url,
                    message=f"GET request to {url} failed after {max_retries + 1} attempts: {exc}",
                    cause=exc,
                ) from exc

        # Exponential backoff (skip on last attempt since we're about to fail)
        if attempt < max_retries:
            sleep_time = backoff_factor * (2**attempt)
            logger.debug(f"Waiting {sleep_time:.2f}s before retry")
            time.sleep(sleep_time)

    raise HttpRequestError(
        method="GET",
        url=url,
        message=(
            f"GET request to {url} failed with status "
            f"{response.status_code} after {max_retries + 1} attempts"
        ),
        status_code=response.status_code,
        response=response,
    )
