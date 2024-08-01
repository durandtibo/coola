r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "LazyModule",
    "check_jax",
    "check_numpy",
    "check_package",
    "check_packaging",
    "check_pandas",
    "check_polars",
    "check_pyarrow",
    "check_torch",
    "check_torch_numpy",
    "check_xarray",
    "decorator_package_available",
    "is_jax_available",
    "is_numpy_available",
    "is_packaging_available",
    "is_pandas_available",
    "is_polars_available",
    "is_pyarrow_available",
    "is_torch_available",
    "is_torch_numpy_available",
    "is_xarray_available",
    "jax_available",
    "lazy_import",
    "module_available",
    "numpy_available",
    "package_available",
    "packaging_available",
    "pandas_available",
    "polars_available",
    "pyarrow_available",
    "torch_available",
    "torch_numpy_available",
    "xarray_available",
]

import importlib
from contextlib import suppress
from functools import lru_cache, wraps
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

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

    >>> from coola.utils.imports import package_available
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

    >>> from coola.utils.imports import module_available
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


def check_package(package: str, command: str | None = None) -> None:
    r"""Check if the given package is installed.

    Args:
        package: The package name.
        command: The command to install the package.

    Raises:
        RuntimeError: if the package is not installed.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_package
    >>> check_package("numpy")

    ```
    """
    if not package_available(package):
        msg = f"'{package}' package is required but not installed."
        if command is not None:
            msg += f" You can install '{package}' package with the command:\n\n{command}"
        raise RuntimeError(msg)


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
            "'jax' package is required but not installed. "
            "You can install 'jax' package with the command:\n\n"
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
            "'numpy' package is required but not installed. "
            "You can install 'numpy' package with the command:\n\n"
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
            "'packaging' package is required but not installed. "
            "You can install 'packaging' package with the command:\n\n"
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
            "'pandas' package is required but not installed. "
            "You can install 'pandas' package with the command:\n\n"
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
            "'polars' package is required but not installed. "
            "You can install 'polars' package with the command:\n\n"
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
            "'pyarrow' package is required but not installed. "
            "You can install 'pyarrow' package with the command:\n\n"
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
            "'torch' package is required but not installed. "
            "You can install 'torch' package with the command:\n\n"
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


#######################
#     torch+numpy     #
#######################

# import is here because is_torch_available is defined above
if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


@lru_cache
def is_torch_numpy_available() -> bool:
    r"""Indicate if the ``torch`` and ``numpy`` packages are installed
    and are compatible.

    Returns:
        ``True`` if both packages are available and compatible,
            otherwise ``False``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import is_torch_numpy_available
    >>> is_torch_numpy_available()

    ```
    """
    if not is_torch_available():
        return False
    if not is_numpy_available():
        return False
    with suppress(RuntimeError):

        torch.tensor([1.0]).numpy()  # Check if the libraries are compatible
        return True
    return False


def check_torch_numpy() -> None:
    r"""Check if the ``torch`` and ``numpy`` packages are installed and
    are compatible.

    Raises:
        RuntimeError: if one of the packages is not installed or if
            they are not compatible.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import check_torch_numpy
    >>> check_torch_numpy()

    ```
    """
    if not is_torch_numpy_available():
        msg = (
            "'torch' and 'numpy' packages are required but one of the package is not "
            "installed or they are not compatible. "
        )
        raise RuntimeError(msg)


def torch_numpy_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``torch`` and
    ``numpy`` packages are installed and are compatible.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``torch`` and ``numpy`` packages
            are installed and are compatible., otherwise ``None``.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import torch_numpy_available
    >>> @torch_numpy_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_torch_numpy_available)


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
            "'xarray' package is required but not installed. "
            "You can install 'xarray' package with the command:\n\n"
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


# The implementation of LazyModule is based on
# https://github.com/Lightning-AI/utilities/blob/b4232544a6e974fee2bebeb029849bd53916bbda/src/lightning_utilities/core/imports.py#L254C1-L292C52
# with license found in the licenses/LICENSE_lightning_utilities
class LazyModule(ModuleType):
    """Define a proxy module that lazily imports a module the first time
    it is actually used.

    Args:
        name: The fully-qualified module name to import.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import LazyModule
    >>> # Lazy version of import numpy as np
    >>> np = LazyModule("numpy")
    >>> # The module is imported the first time it is actually used.
    >>> np.ones((2, 3))
    array([[1., 1., 1.],
           [1., 1., 1.]])

    ```
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._module: Any = None

    def __getattr__(self, item: str) -> Any:
        if self._module is None:
            self._import_module()
        return getattr(self._module, item)

    def __dir__(self) -> list[str]:
        if self._module is None:
            self._import_module()
        return dir(self._module)

    def _import_module(self) -> None:
        self._module = importlib.import_module(self.__name__)

        # Update this object's dict so that attribute references are efficient
        # (__getattr__ is only called on lookups that fail)
        self.__dict__.update(self._module.__dict__)


def lazy_import(name: str) -> LazyModule:
    r"""Return a proxy of the module/package to lazily import.

    Args:
        name: The fully-qualified module name to import.

    Returns:
        A proxy module that lazily imports a module the first time
            it is actually used.

    Example usage:

    ```pycon

    >>> from coola.utils.imports import lazy_import
    >>> # Lazy version of import numpy as np
    >>> np = lazy_import("numpy")
    >>> # The module is imported the first time it is actually used.
    >>> np.ones((2, 3))
    array([[1., 1., 1.],
           [1., 1., 1.]])

    ```
    """
    return LazyModule(name)
