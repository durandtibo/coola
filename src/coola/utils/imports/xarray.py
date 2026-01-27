r"""Contain utilities for optional xarray dependency."""

from __future__ import annotations

__all__ = ["check_xarray", "is_xarray_available", "raise_xarray_missing_error", "xarray_available"]

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


def check_xarray() -> None:
    r"""Check if the ``xarray`` package is installed.

    Raises:
        RuntimeError: if the ``xarray`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_xarray
        >>> check_xarray()

        ```
    """
    if not is_xarray_available():
        raise_xarray_missing_error()


@lru_cache
def is_xarray_available() -> bool:
    r"""Indicate if the ``xarray`` package is installed or not.

    Returns:
        ``True`` if ``xarray`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_xarray_available
        >>> is_xarray_available()

        ```
    """
    return package_available("xarray")


def xarray_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``xarray``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``xarray`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import xarray_available
        >>> @xarray_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_xarray_available)


def raise_xarray_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``xarray`` package is
    missing."""
    raise_package_missing_error("xarray", "xarray")
