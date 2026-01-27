r"""Contain utilities for optional pyarrow dependency."""

from __future__ import annotations

__all__ = [
    "check_pyarrow",
    "is_pyarrow_available",
    "pyarrow_available",
    "raise_pyarrow_missing_error",
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


def check_pyarrow() -> None:
    r"""Check if the ``pyarrow`` package is installed.

    Raises:
        RuntimeError: if the ``pyarrow`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_pyarrow
        >>> check_pyarrow()

        ```
    """
    if not is_pyarrow_available():
        raise_pyarrow_missing_error()


@lru_cache
def is_pyarrow_available() -> bool:
    r"""Indicate if the ``pyarrow`` package is installed or not.

    Returns:
        ``True`` if ``pyarrow`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_pyarrow_available
        >>> is_pyarrow_available()

        ```
    """
    return package_available("pyarrow")


def pyarrow_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``pyarrow``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``pyarrow`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import pyarrow_available
        >>> @pyarrow_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_pyarrow_available)


def raise_pyarrow_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``pyarrow`` package is
    missing."""
    raise_package_missing_error("pyarrow", "pyarrow")
