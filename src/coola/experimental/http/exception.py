r"""Contains custom exceptions for HTTP requests using the httpx library.

This module provides exceptions for handling HTTP request failures with
detailed error information including request method, URL, status codes,
and response objects.
"""

from __future__ import annotations

__all__ = ["HttpRequestError"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx


class HttpRequestError(RuntimeError):
    r"""Exception raised when an HTTP request fails.

    This exception captures comprehensive details about failed HTTP requests,
    including the request method, URL, status code, and the full response
    object when available. It supports exception chaining to preserve the
    original cause of the error.

    Args:
        method: The HTTP method used for the request (e.g., 'GET', 'POST').
        url: The target URL that was requested.
        message: A descriptive error message explaining the failure.
        status_code: The HTTP status code returned by the server, if the
            request reached the server. Defaults to ``None`` if the request
            failed before receiving a response.
        response: The complete httpx.Response object containing headers,
            body, and other response details. Defaults to ``None`` if no
            response was received.
        cause: The original exception that caused this error, used for
            exception chaining. Defaults to ``None``.

    Attributes:
        method (str): The HTTP method used for the request.
        url (str): The target URL that was requested.
        status_code (int | None): The HTTP status code, if available.
        response (httpx.Response | None): The full response object, if available.

    Examples:
        Raising an error for a failed GET request:

        ```pycon
        >>> raise HttpRequestError(
        ...     method="GET",
        ...     url="https://api.example.com/data",
        ...     message="Request failed with status 404",
        ...     status_code=404,
        ... )  # doctest: +SKIP

        ```

        Chaining with an original exception:

        ```pycon
        >>> try:
        ...     # Some httpx request
        ...     pass
        ... except httpx.RequestError as e:
        ...     raise HttpRequestError(
        ...         method="POST",
        ...         url="https://api.example.com/submit",
        ...         message="Connection failed",
        ...         cause=e,
        ...     )
        ...

        ```
    """

    def __init__(
        self,
        *,
        method: str,
        url: str,
        message: str,
        status_code: int | None = None,
        response: httpx.Response | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.url = url
        self.status_code = status_code
        self.response = response
        self.__cause__: Exception | None = cause

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(method={self.method!r}, "
            f"url={self.url!r}, status_code={self.status_code})"
        )
