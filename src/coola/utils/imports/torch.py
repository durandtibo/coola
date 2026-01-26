r"""Contain utilities for optional torch dependency."""

from __future__ import annotations

__all__ = ["check_torch", "is_torch_available", "raise_torch_missing_error", "torch_available"]

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


def check_torch() -> None:
    r"""Check if the ``torch`` package is installed.

    Raises:
        RuntimeError: if the ``torch`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_torch
        >>> check_torch()

        ```
    """
    if not is_torch_available():
        raise_torch_missing_error()


@lru_cache
def is_torch_available() -> bool:
    r"""Indicate if the ``torch`` package is installed or not.

    Returns:
        ``True`` if ``torch`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_torch_available
        >>> is_torch_available()

        ```
    """
    return package_available("torch")


def torch_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``torch``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``torch`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import torch_available
        >>> @torch_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_torch_available)


def raise_torch_missing_error() -> NoReturn:
    r"""Raise a RuntimeError to indicate the ``torch`` package is
    missing."""
    raise_package_missing_error("torch", "torch")
