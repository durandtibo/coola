r"""Contain utilities for optional httpx dependency."""

from __future__ import annotations

__all__ = [
    "check_httpx",
    "httpx_available",
    "is_httpx_available",
    "raise_httpx_missing_error",
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


def check_httpx() -> None:
    r"""Check if the ``httpx`` package is installed.

    Raises:
        RuntimeError: if the ``httpx`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_httpx
        >>> check_httpx()

        ```
    """
    if not is_httpx_available():
        raise_httpx_missing_error()


@lru_cache
def is_httpx_available() -> bool:
    r"""Indicate if the ``httpx`` package is installed or not.

    Returns:
        ``True`` if ``httpx`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_httpx_available
        >>> is_httpx_available()

        ```
    """
    return package_available("httpx")


def httpx_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``httpx``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``httpx`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import httpx_available
        >>> @httpx_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_httpx_available)


def raise_httpx_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``httpx`` package is
    missing."""
    raise_package_missing_error("httpx", "httpx")
