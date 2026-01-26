r"""Contain utilities for optional polars dependency."""

from __future__ import annotations

__all__ = ["check_polars", "is_polars_available", "polars_available", "raise_polars_missing_error"]

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


def check_polars() -> None:
    r"""Check if the ``polars`` package is installed.

    Raises:
        RuntimeError: if the ``polars`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_polars
        >>> check_polars()

        ```
    """
    if not is_polars_available():
        raise_polars_missing_error()


@lru_cache
def is_polars_available() -> bool:
    r"""Indicate if the ``polars`` package is installed or not.

    Returns:
        ``True`` if ``polars`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_polars_available
        >>> is_polars_available()

        ```
    """
    return package_available("polars")


def polars_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``polars``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``polars`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import polars_available
        >>> @polars_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_polars_available)


def raise_polars_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``polars`` package is
    missing."""
    raise_package_missing_error("polars", "polars")
