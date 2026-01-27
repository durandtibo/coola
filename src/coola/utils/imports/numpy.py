r"""Contain utilities for optional numpy dependency."""

from __future__ import annotations

__all__ = ["check_numpy", "is_numpy_available", "numpy_available", "raise_numpy_missing_error"]

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


def check_numpy() -> None:
    r"""Check if the ``numpy`` package is installed.

    Raises:
        RuntimeError: if the ``numpy`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_numpy
        >>> check_numpy()

        ```
    """
    if not is_numpy_available():
        raise_numpy_missing_error()


@lru_cache
def is_numpy_available() -> bool:
    r"""Indicate if the ``numpy`` package is installed or not.

    Returns:
        ``True`` if ``numpy`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_numpy_available
        >>> is_numpy_available()

        ```
    """
    return package_available("numpy")


def numpy_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``numpy``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``numpy`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import numpy_available
        >>> @numpy_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_numpy_available)


def raise_numpy_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``numpy`` package is
    missing."""
    raise_package_missing_error("numpy", "numpy")
