r"""Contain utilities for optional colorlog dependency."""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "colorlog_available",
    "is_colorlog_available",
    "raise_colorlog_missing_error",
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


def check_colorlog() -> None:
    r"""Check if the ``colorlog`` package is installed.

    Raises:
        RuntimeError: if the ``colorlog`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_colorlog
        >>> check_colorlog()

        ```
    """
    if not is_colorlog_available():
        raise_colorlog_missing_error()


@lru_cache(1)
def is_colorlog_available() -> bool:
    r"""Indicate if the ``colorlog`` package is installed or not.

    Returns:
        ``True`` if ``colorlog`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_colorlog_available
        >>> is_colorlog_available()

        ```
    """
    return package_available("colorlog")


def colorlog_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``colorlog``
    is installed.

    Args:
        fn: The function to conditionally execute.

    Returns:
        A wrapper around ``fn`` if ``colorlog`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import colorlog_available
        >>> @colorlog_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_colorlog_available)


def raise_colorlog_missing_error() -> NoReturn:
    r"""Raise a ``RuntimeError`` to indicate the ``colorlog`` package is
    missing.

    Raises:
        RuntimeError: Always, with a message indicating that the
            ``colorlog`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import raise_colorlog_missing_error
        >>> raise_colorlog_missing_error()  # doctest: +SKIP

        ```
    """
    raise_package_missing_error("colorlog", "colorlog")
