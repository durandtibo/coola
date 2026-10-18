r"""Contain utilities for optional pandas dependency."""

from __future__ import annotations

__all__ = ["check_pandas", "is_pandas_available", "pandas_available", "raise_pandas_missing_error"]

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


def check_pandas() -> None:
    r"""Check if the ``pandas`` package is installed.

    Raises:
        MissingPackageError: if the ``pandas`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_pandas
        >>> check_pandas()

        ```
    """
    if not is_pandas_available():
        raise_pandas_missing_error()


@availability_cache
def is_pandas_available() -> bool:
    r"""Indicate if the ``pandas`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``pandas`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_pandas_available
        >>> is_pandas_available()

        ```
    """
    return module_available("pandas")


def pandas_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``pandas``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``pandas`` package is installed,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import pandas_available
        >>> @pandas_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_pandas_available)


def raise_pandas_missing_error() -> Never:
    r"""Raise a ``MissingPackageError`` to indicate the ``pandas``
    package is missing.

    Raises:
        MissingPackageError: Always, with a message indicating that the
            ``pandas`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import raise_pandas_missing_error
        >>> raise_pandas_missing_error()  # doctest: +SKIP

        ```
    """
    raise_package_missing_error("pandas", "pandas")
