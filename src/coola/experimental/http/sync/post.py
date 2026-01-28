r"""Contain synchronous HTTP POST request with automatic retry logic."""

from __future__ import annotations

__all__ = ["post_with_automatic_retry"]

from typing import TYPE_CHECKING, Any

from coola.experimental.http.constants import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    RETRY_STATUS_CODES,
)
from coola.experimental.http.sync.utils import request_with_automatic_retry
from coola.utils.imports import check_httpx, is_httpx_available

if TYPE_CHECKING or is_httpx_available():
    import httpx
else:  # pragma: no cover
    from coola.utils.fallback.httpx import httpx


def post_with_automatic_retry(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: tuple[int, ...] = RETRY_STATUS_CODES,
    **kwargs: Any,
) -> httpx.Response:
    r"""Send an HTTP POST request with automatic retry logic for
    transient errors.

    This function performs an HTTP POST request with a configured retry policy
    for transient server errors (429, 500, 502, 503, 504). It applies an
    exponential backoff retry strategy. The function validates the HTTP
    response and raises detailed errors for failures.

    Args:
        url: The URL to send the POST request to.
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
        **kwargs: Additional keyword arguments passed to ``httpx.Client.post()``.

    Returns:
        An httpx.Response object containing the server's HTTP response.

    Raises:
        HttpRequestError: If the request times out, encounters network errors,
            or fails after exhausting all retries.
        ValueError: If max_retries or backoff_factor are negative.

    Example:
        ```pycon
        >>> from coola.experimental.http.sync import post_with_automatic_retry
        >>> response = post_with_automatic_retry(
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
        return request_with_automatic_retry(
            url=url,
            method="POST",
            request_func=client.post,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            **kwargs,
        )
    finally:
        if owns_client:
            client.close()



