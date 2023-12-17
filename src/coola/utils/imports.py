from __future__ import annotations

__all__ = [
    "check_jax",
    "check_numpy",
    "check_pandas",
    "check_polars",
    "check_torch",
    "check_xarray",
    "decorator_package_available",
    "is_jax_available",
    "is_numpy_available",
    "is_pandas_available",
    "is_polars_available",
    "is_torch_available",
    "is_xarray_available",
    "xarray_available",
]

from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from typing import Any


def decorator_package_available(
    fn: Callable[..., Any], condition: Callable[[], bool]
) -> Callable[..., Any]:
    r"""Implements a decorator to execute a function only if a package is
    installed.

    Args:
    ----
        fn (``Callable``): Specifies the function to execute.
        condition (``Callable``): Specifies the condition to
            check if a package is installed or not.

    Returns:
    -------
        ``Any``: The output of ``fn`` if ``xarray`` package is
            installed, otherwise ``None``.

    Example usage:

    .. code-block:: pycon

        >>> from functools import partial
        >>> from coola.utils.imports import decorator_package_available
        >>> decorator = partial(decorator_package_available, condition=is_numpy_available)
        >>> @decorator
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function(2)
        44
    """

    @wraps(fn)
    def inner(*args, **kwargs) -> Any:
        print(condition, condition())
        if not condition():
            return None
        return fn(*args, **kwargs)

    return inner


def is_jax_available() -> bool:
    r"""Indicates if the ``jax`` package is installed or not.

    Returns:
    -------
        bool: ``True`` if ``jax`` is available otherwise ``False``.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import is_jax_available
        >>> is_jax_available()
    """
    return find_spec("jax") is not None


def check_jax() -> None:
    r"""Checks if the ``jax`` package is installed.

    Raises:
    ------
        RuntimeError if the ``jax`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import check_jax
        >>> check_jax()  # doctest: +SKIP
    """
    if not is_jax_available():
        raise RuntimeError(
            "`jax` package is required but not installed. "
            "You can install `jax` package with the command:\n\n"
            "pip install jax\n"
        )


def is_numpy_available() -> bool:
    r"""Indicates if the ``numpy`` package is installed or not.

    Returns:
    -------
        bool: ``True`` if ``numpy`` is available otherwise ``False``.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import is_numpy_available
        >>> is_numpy_available()
    """
    return find_spec("numpy") is not None


def check_numpy() -> None:
    r"""Checks if the ``numpy`` package is installed.

    Raises:
    ------
        RuntimeError if the ``numpy`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import check_numpy
        >>> check_numpy()  # doctest: +SKIP
    """
    if not is_numpy_available():
        raise RuntimeError(
            "`numpy` package is required but not installed. "
            "You can install `numpy` package with the command:\n\n"
            "pip install numpy\n"
        )


def is_pandas_available() -> bool:
    r"""Indicates if the ``pandas`` package is installed or not.

    Returns:
    -------
        bool: ``True`` if ``pandas`` is available otherwise ``False``.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import is_pandas_available
        >>> is_pandas_available()
    """
    return find_spec("pandas") is not None


def check_pandas() -> None:
    r"""Checks if the ``pandas`` package is installed.

    Raises:
    ------
        RuntimeError if the ``pandas`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import check_pandas
        >>> check_pandas()  # doctest: +SKIP
    """
    if not is_pandas_available():
        raise RuntimeError(
            "`pandas` package is required but not installed. "
            "You can install `pandas` package with the command:\n\n"
            "pip install pandas\n"
        )


def is_polars_available() -> bool:
    r"""Indicates if the ``polars`` package is installed or not.

    Returns:
    -------
        bool: ``True`` if ``polars`` is available otherwise ``False``.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import is_polars_available
        >>> is_polars_available()
    """
    return find_spec("polars") is not None


def check_polars() -> None:
    r"""Checks if the ``polars`` package is installed.

    Raises:
    ------
        RuntimeError if the ``polars`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import check_polars
        >>> check_polars()  # doctest: +SKIP
    """
    if not is_polars_available():
        raise RuntimeError(
            "`polars` package is required but not installed. "
            "You can install `polars` package with the command:\n\n"
            "pip install polars\n"
        )


def is_torch_available() -> bool:
    r"""Indicates if the ``torch`` package is installed or not.

    Returns:
    -------
        bool: ``True`` if ``torch`` is available otherwise ``False``.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import is_torch_available
        >>> is_torch_available()
    """
    return find_spec("torch") is not None


def check_torch() -> None:
    r"""Checks if the ``torch`` package is installed.

    Raises:
        RuntimeError if the ``torch`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import check_torch
        >>> check_torch()  # doctest: +SKIP
    """
    if not is_torch_available():
        raise RuntimeError(
            "`torch` package is required but not installed. "
            "You can install `torch` package with the command:\n\n"
            "pip install torch\n"
        )


def is_xarray_available() -> bool:
    r"""Indicates if the ``xarray`` package is installed or not.

    Returns:
    -------
        bool: ``True`` if ``xarray`` is available otherwise ``False``.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import is_xarray_available
        >>> is_xarray_available()
    """
    return find_spec("xarray") is not None


def check_xarray() -> None:
    r"""Checks if the ``xarray`` package is installed.

    Raises:
    ------
        RuntimeError if the ``xarray`` package is not installed.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import check_xarray
        >>> check_xarray()  # doctest: +SKIP
    """
    if not is_xarray_available():
        raise RuntimeError(
            "`xarray` package is required but not installed. "
            "You can install `xarray` package with the command:\n\n"
            "pip install xarray\n"
        )


def xarray_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implements a decorator to execute a function only if ``xarray``
    package is installed.

    Args:
    ----
        fn (``Callable``): Specifies the function to execute.

    Returns:
    -------
        ``Any``: The output of ``fn`` if ``xarray`` package is
            installed, otherwise ``None``.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.imports import xarray_available
        >>> @xarray_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()
    """
    print("is_xarray_available", is_xarray_available)
    return decorator_package_available(fn, is_xarray_available)
