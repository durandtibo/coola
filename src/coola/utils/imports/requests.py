r"""Contain utilities for optional requests dependency."""

from __future__ import annotations

__all__ = [
    "check_requests",
    "is_requests_available",
    "raise_requests_missing_error",
    "requests_available",
]

from functools import lru_cache
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from coola.utils.imports.universal import (
    decorator_package_available,
    package_available,
    raise_package_missing_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def check_requests() -> None:
    r"""Check if the ``requests`` package is installed.

    Raises:
        RuntimeError: if the ``requests`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_requests
        >>> check_requests()

        ```
    """
    if not is_requests_available():
        raise_requests_missing_error()


@lru_cache
def is_requests_available() -> bool:
    r"""Indicate if the ``requests`` package is installed or not.

    Returns:
        ``True`` if ``requests`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_requests_available
        >>> is_requests_available()

        ```
    """
    return package_available("requests")


def requests_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``requests``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``requests`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import requests_available
        >>> @requests_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_requests_available)


def raise_requests_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``requests`` package is
    missing."""
    raise_package_missing_error("requests", "requests")
