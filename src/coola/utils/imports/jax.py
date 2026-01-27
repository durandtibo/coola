r"""Contain utilities for optional jax dependency."""

from __future__ import annotations

__all__ = ["check_jax", "is_jax_available", "jax_available", "raise_jax_missing_error"]

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


def check_jax() -> None:
    r"""Check if the ``jax`` package is installed.

    Raises:
        RuntimeError: if the ``jax`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_jax
        >>> check_jax()

        ```
    """
    if not is_jax_available():
        raise_jax_missing_error()


@lru_cache
def is_jax_available() -> bool:
    r"""Indicate if the ``jax`` package is installed or not.

    Returns:
        ``True`` if ``jax`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_jax_available
        >>> is_jax_available()

        ```
    """
    return package_available("jax")


def jax_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``jax``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``jax`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import jax_available
        >>> @jax_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_jax_available)


def raise_jax_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``jax`` package is
    missing."""
    raise_package_missing_error("jax", "jax")
