r"""Contain utilities for optional packaging dependency."""

from __future__ import annotations

__all__ = [
    "check_packaging",
    "is_packaging_available",
    "packaging_available",
    "raise_packaging_missing_error",
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


def check_packaging() -> None:
    r"""Check if the ``packaging`` package is installed.

    Raises:
        RuntimeError: if the ``packaging`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_packaging
        >>> check_packaging()

        ```
    """
    if not is_packaging_available():
        raise_packaging_missing_error()


@lru_cache
def is_packaging_available() -> bool:
    r"""Indicate if the ``packaging`` package is installed or not.

    Returns:
        ``True`` if ``packaging`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_packaging_available
        >>> is_packaging_available()

        ```
    """
    return package_available("packaging")


def packaging_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``packaging``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``packaging`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import packaging_available
        >>> @packaging_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_packaging_available)


def raise_packaging_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``packaging`` package is
    missing."""
    raise_package_missing_error("packaging", "packaging")
