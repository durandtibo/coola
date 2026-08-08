r"""Contain utilities for optional pydantic dependency."""

from __future__ import annotations

__all__ = [
    "check_pydantic",
    "is_pydantic_available",
    "pydantic_available",
    "raise_pydantic_missing_error",
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


def check_pydantic() -> None:
    r"""Check if the ``pydantic`` package is installed.

    Raises:
        RuntimeError: if the ``pydantic`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_pydantic
        >>> check_pydantic()

        ```
    """
    if not is_pydantic_available():
        raise_pydantic_missing_error()


@lru_cache(1)
def is_pydantic_available() -> bool:
    r"""Indicate if the ``pydantic`` package is installed or not.

    Returns:
        ``True`` if ``pydantic`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_pydantic_available
        >>> is_pydantic_available()

        ```
    """
    return package_available("pydantic")


def pydantic_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``pydantic``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``pydantic`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import pydantic_available
        >>> @pydantic_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_pydantic_available)


def raise_pydantic_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``pydantic`` package is
    missing."""
    raise_package_missing_error("pydantic", "pydantic")
