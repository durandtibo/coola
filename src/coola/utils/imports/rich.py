r"""Contain utilities for optional rich dependency."""

from __future__ import annotations

__all__ = [
    "check_rich",
    "is_rich_available",
    "raise_rich_missing_error",
    "rich_available",
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


def check_rich() -> None:
    r"""Check if the ``rich`` package is installed.

    Raises:
        RuntimeError: if the ``rich`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_rich
        >>> check_rich()

        ```
    """
    if not is_rich_available():
        raise_rich_missing_error()


@lru_cache(1)
def is_rich_available() -> bool:
    r"""Indicate if the ``rich`` package is installed or not.

    Returns:
        ``True`` if ``rich`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_rich_available
        >>> is_rich_available()

        ```
    """
    return package_available("rich")


def rich_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``rich`` is
    installed.

    Args:
        fn: The function to conditionally execute.

    Returns:
        A wrapper around ``fn`` if ``rich`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import rich_available
        >>> @rich_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_rich_available)


def raise_rich_missing_error() -> NoReturn:
    r"""Raise a ``RuntimeError`` to indicate the ``rich`` package is
    missing.

    Raises:
        RuntimeError: Always, with a message indicating that the
            ``rich`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import raise_rich_missing_error
        >>> raise_rich_missing_error()  # doctest: +SKIP

        ```
    """
    raise_package_missing_error("rich", "rich")
