r"""Contain utilities for optional numpy dependency."""

from __future__ import annotations

__all__ = ["check_numpy", "is_numpy_available", "numpy_available", "raise_numpy_missing_error"]

from typing import TYPE_CHECKING, Any, Never, TypeVar

from coola.utils.imports.universal import (
    availability_cache,
    decorator_package_available,
    module_available,
    raise_package_missing_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def check_numpy() -> None:
    r"""Check if the ``numpy`` package is installed.

    Raises:
        MissingPackageError: if the ``numpy`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_numpy
        >>> check_numpy()

        ```
    """
    if not is_numpy_available():
        raise_numpy_missing_error()


@availability_cache
def is_numpy_available() -> bool:
    r"""Indicate if the ``numpy`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``numpy`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_numpy_available
        >>> is_numpy_available()

        ```
    """
    return module_available("numpy")


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


def raise_numpy_missing_error() -> Never:
    r"""Raise a ``MissingPackageError`` to indicate the ``numpy`` package
    is missing.

    Raises:
        MissingPackageError: Always, with a message indicating that the
            ``numpy`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import raise_numpy_missing_error
        >>> raise_numpy_missing_error()  # doctest: +SKIP

        ```
    """
    raise_package_missing_error("numpy", "numpy")
