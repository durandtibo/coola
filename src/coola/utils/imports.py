r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_jax",
    "check_numpy",
    "check_packaging",
    "check_pandas",
    "check_polars",
    "check_pyarrow",
    "check_torch",
    "check_xarray",
    "decorator_package_available",
    "is_jax_available",
    "is_numpy_available",
    "is_packaging_available",
    "is_pandas_available",
    "is_polars_available",
    "is_pyarrow_available",
    "is_torch_available",
    "is_xarray_available",
    "jax_available",
    "module_available",
    "numpy_available",
    "package_available",
    "packaging_available",
    "pandas_available",
    "polars_available",
    "pyarrow_available",
    "torch_available",
    "xarray_available",
]

import importlib
from contextlib import suppress
from functools import lru_cache, wraps
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@lru_cache
def package_available(name: str) -> bool:
    """Indicate if a package is available or not.

    Args:
        name: The package name to check.

    Returns:
        ``True`` if the package is available, otherwise ``False``.

    Example usage:

    ```pycon

    >>> package_available("os")
    True
    >>> package_available("missing_package")
    False

    ```
    """
    with suppress(ModuleNotFoundError):
        return find_spec(name) is not None
    return False


@lru_cache
def module_available(name: str) -> bool:
    """Indicate if a module is available or not.

    Args:
        name: The module name to check.

    Returns:
        ``True`` if the module is available, otherwise ``False``.

    Example usage:

    ```pycon

    >>> module_available("os")
    True
    >>> module_available("os.missing")
    False
    >>> module_available("missing.module")
    False

    ```
    """
    with suppress(ImportError):
        importlib.import_module(name)
        return True
    return False


def decorator_package_available(
    fn: Callable[..., Any], condition: Callable[[], bool]
) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if a package is
    installed.

    Args:
        fn: The function to execute.
        condition: The condition to
            check if a package is installed or not.

    Returns:
        A wrapper around ``fn`` if condition is true,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from functools import partial
    >>> from coola.utils.imports import decorator_package_available
    >>> decorator = partial(decorator_package_available, condition=is_numpy_available)
    >>> @decorator
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function(2)
        44

    ```
    """

    @wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
        if not condition():
            return None
        return fn(*args, **kwargs)

    return inner


###############
#     jax     #
###############


@lru_cache
def is_jax_available() -> bool:
    r"""Indicate if the ``jax`` package is installed or not.

    Returns:
        ``True`` if ``jax`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_jax_available
    >>> is_jax_available()

    ```
    """
    return package_available("jax")


def check_jax() -> None:
    r"""Check if the ``jax`` package is installed.

    Raises:
        RuntimeError: if the ``jax`` package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_jax
    >>> check_jax()

    ```
    """
    if not is_jax_available():
        msg = (
            "`jax` package is required but not installed. "
            "You can install `jax` package with the command:\n\n"
            "pip install jax\n"
        )
        raise RuntimeError(msg)


def jax_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``jax``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``jax`` package is installed,
            otherwise ``None``.

    Example usage:

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


#################
#     numpy     #
#################


@lru_cache
def is_numpy_available() -> bool:
    r"""Indicate if the ``numpy`` package is installed or not.

    Returns:
        ``True`` if ``numpy`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_numpy_available
    >>> is_numpy_available()

    ```
    """
    return package_available("numpy")


def check_numpy() -> None:
    r"""Check if the ``numpy`` package is installed.

    Raises:
        RuntimeError: if the ``numpy`` package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_numpy
    >>> check_numpy()

    ```
    """
    if not is_numpy_available():
        msg = (
            "`numpy` package is required but not installed. "
            "You can install `numpy` package with the command:\n\n"
            "pip install numpy\n"
        )
        raise RuntimeError(msg)


def numpy_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``numpy``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``numpy`` package is installed,
            otherwise ``None``.

    Example usage:

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


#####################
#     packaging     #
#####################


@lru_cache
def is_packaging_available() -> bool:
    r"""Indicate if the ``packaging`` package is installed or not.

    Returns:
        ``True`` if ``packaging`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_packaging_available
    >>> is_packaging_available()

    ```
    """
    return package_available("packaging")


def check_packaging() -> None:
    r"""Check if the ``packaging`` package is installed.

    Raises:
        RuntimeError: if the ``packaging`` package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_packaging
    >>> check_packaging()

    ```
    """
    if not is_packaging_available():
        msg = (
            "`packaging` package is required but not installed. "
            "You can install `packaging` package with the command:\n\n"
            "pip install packaging\n"
        )
        raise RuntimeError(msg)


def packaging_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``packaging``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``packaging`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import packaging_available
    >>> @packaging_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_packaging_available)


##################
#     pandas     #
##################


@lru_cache
def is_pandas_available() -> bool:
    r"""Indicate if the ``pandas`` package is installed or not.

    Returns:
        ``True`` if ``pandas`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_pandas_available
    >>> is_pandas_available()

    ```
    """
    return package_available("pandas")


def check_pandas() -> None:
    r"""Check if the ``pandas`` package is installed.

    Raises:
        RuntimeError: if the ``pandas`` package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_pandas
    >>> check_pandas()

    ```
    """
    if not is_pandas_available():
        msg = (
            "`pandas` package is required but not installed. "
            "You can install `pandas` package with the command:\n\n"
            "pip install pandas\n"
        )
        raise RuntimeError(msg)


def pandas_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``pandas``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``pandas`` package is installed,
            otherwise ``None``.

    Example usage:

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


##################
#     polars     #
##################


@lru_cache
def is_polars_available() -> bool:
    r"""Indicate if the ``polars`` package is installed or not.

    Returns:
        ``True`` if ``polars`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_polars_available
    >>> is_polars_available()

    ```
    """
    return package_available("polars")


def check_polars() -> None:
    r"""Check if the ``polars`` package is installed.

    Raises:
        RuntimeError: if the ``polars`` package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_polars
    >>> check_polars()

    ```
    """
    if not is_polars_available():
        msg = (
            "`polars` package is required but not installed. "
            "You can install `polars` package with the command:\n\n"
            "pip install polars\n"
        )
        raise RuntimeError(msg)


def polars_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``polars``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``polars`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import polars_available
    >>> @polars_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_polars_available)


###################
#     pyarrow     #
###################


@lru_cache
def is_pyarrow_available() -> bool:
    r"""Indicate if the ``pyarrow`` package is installed or not.

    Returns:
        ``True`` if ``pyarrow`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_pyarrow_available
    >>> is_pyarrow_available()

    ```
    """
    return package_available("pyarrow")


def check_pyarrow() -> None:
    r"""Check if the ``pyarrow`` package is installed.

    Raises:
        RuntimeError: if the ``pyarrow`` package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_pyarrow
    >>> check_pyarrow()

    ```
    """
    if not is_pyarrow_available():
        msg = (
            "`pyarrow` package is required but not installed. "
            "You can install `pyarrow` package with the command:\n\n"
            "pip install pyarrow\n"
        )
        raise RuntimeError(msg)


def pyarrow_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``pyarrow``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``pyarrow`` package is installed,
            otherwise ``None``.

    Example usage:

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


#################
#     torch     #
#################


@lru_cache
def is_torch_available() -> bool:
    r"""Indicate if the ``torch`` package is installed or not.

    Returns:
        ``True`` if ``torch`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_torch_available
    >>> is_torch_available()

    ```
    """
    return package_available("torch")


def check_torch() -> None:
    r"""Check if the ``torch`` package is installed.

    Raises:
        RuntimeError: if the ``torch`` package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_torch
    >>> check_torch()

    ```
    """
    if not is_torch_available():
        msg = (
            "`torch` package is required but not installed. "
            "You can install `torch` package with the command:\n\n"
            "pip install torch\n"
        )
        raise RuntimeError(msg)


def torch_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``torch``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``torch`` package is installed,
            otherwise ``None``.

    Example usage:

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


##################
#     xarray     #
##################


@lru_cache
def is_xarray_available() -> bool:
    r"""Indicate if the ``xarray`` package is installed or not.

    Returns:
        ``True`` if ``xarray`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_xarray_available
    >>> is_xarray_available()

    ```
    """
    return package_available("xarray")


def check_xarray() -> None:
    r"""Check if the ``xarray`` package is installed.

    Raises:
        RuntimeError: if the ``xarray`` package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_xarray
    >>> check_xarray()

    ```
    """
    if not is_xarray_available():
        msg = (
            "`xarray` package is required but not installed. "
            "You can install `xarray` package with the command:\n\n"
            "pip install xarray\n"
        )
        raise RuntimeError(msg)


def xarray_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``xarray``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``xarray`` package is installed,
            otherwise ``None``.

    Example usage:

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
