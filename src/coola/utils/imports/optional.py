r"""Contain the availability checks of the optional libraries.

``is_<package>_available`` indicates if a library can be imported and
``check_<package>`` raises a ``MissingPackageError`` if it cannot.
"""

from __future__ import annotations

__all__ = [
    "check_colorlog",
    "check_jax",
    "check_numpy",
    "check_packaging",
    "check_pandas",
    "check_polars",
    "check_pyarrow",
    "check_pydantic",
    "check_rich",
    "check_torch",
    "check_torch_numpy",
    "check_xarray",
    "is_colorlog_available",
    "is_jax_available",
    "is_numpy_available",
    "is_packaging_available",
    "is_pandas_available",
    "is_polars_available",
    "is_pyarrow_available",
    "is_pydantic_available",
    "is_rich_available",
    "is_torch_available",
    "is_torch_numpy_available",
    "is_xarray_available",
]

from contextlib import suppress

from coola.utils.imports.package import (
    MissingPackageError,
    availability_cache,
    module_available,
    raise_package_missing_error,
)


@availability_cache
def is_colorlog_available() -> bool:
    r"""Indicate if the ``colorlog`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``colorlog`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_colorlog_available
        >>> is_colorlog_available()

        ```
    """
    return module_available("colorlog")


def check_colorlog() -> None:
    r"""Check if the ``colorlog`` package is installed.

    Raises:
        MissingPackageError: if the ``colorlog`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_colorlog
        >>> check_colorlog()

        ```
    """
    if not is_colorlog_available():
        raise_package_missing_error("colorlog")


@availability_cache
def is_jax_available() -> bool:
    r"""Indicate if the ``jax`` package is installed and can be imported.

    Returns:
        ``True`` if ``jax`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_jax_available
        >>> is_jax_available()

        ```
    """
    return module_available("jax")


def check_jax() -> None:
    r"""Check if the ``jax`` package is installed.

    Raises:
        MissingPackageError: if the ``jax`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_jax
        >>> check_jax()

        ```
    """
    if not is_jax_available():
        raise_package_missing_error("jax")


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
        raise_package_missing_error("numpy")


@availability_cache
def is_packaging_available() -> bool:
    r"""Indicate if the ``packaging`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``packaging`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_packaging_available
        >>> is_packaging_available()

        ```
    """
    return module_available("packaging")


def check_packaging() -> None:
    r"""Check if the ``packaging`` package is installed.

    Raises:
        MissingPackageError: if the ``packaging`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_packaging
        >>> check_packaging()

        ```
    """
    if not is_packaging_available():
        raise_package_missing_error("packaging")


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
        raise_package_missing_error("pandas")


@availability_cache
def is_polars_available() -> bool:
    r"""Indicate if the ``polars`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``polars`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_polars_available
        >>> is_polars_available()

        ```
    """
    return module_available("polars")


def check_polars() -> None:
    r"""Check if the ``polars`` package is installed.

    Raises:
        MissingPackageError: if the ``polars`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_polars
        >>> check_polars()

        ```
    """
    if not is_polars_available():
        raise_package_missing_error("polars")


@availability_cache
def is_pyarrow_available() -> bool:
    r"""Indicate if the ``pyarrow`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``pyarrow`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_pyarrow_available
        >>> is_pyarrow_available()

        ```
    """
    return module_available("pyarrow")


def check_pyarrow() -> None:
    r"""Check if the ``pyarrow`` package is installed.

    Raises:
        MissingPackageError: if the ``pyarrow`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_pyarrow
        >>> check_pyarrow()

        ```
    """
    if not is_pyarrow_available():
        raise_package_missing_error("pyarrow")


@availability_cache
def is_pydantic_available() -> bool:
    r"""Indicate if the ``pydantic`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``pydantic`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_pydantic_available
        >>> is_pydantic_available()

        ```
    """
    return module_available("pydantic")


def check_pydantic() -> None:
    r"""Check if the ``pydantic`` package is installed.

    Raises:
        MissingPackageError: if the ``pydantic`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_pydantic
        >>> check_pydantic()

        ```
    """
    if not is_pydantic_available():
        raise_package_missing_error("pydantic")


@availability_cache
def is_rich_available() -> bool:
    r"""Indicate if the ``rich`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``rich`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_rich_available
        >>> is_rich_available()

        ```
    """
    return module_available("rich")


def check_rich() -> None:
    r"""Check if the ``rich`` package is installed.

    Raises:
        MissingPackageError: if the ``rich`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_rich
        >>> check_rich()

        ```
    """
    if not is_rich_available():
        raise_package_missing_error("rich")


@availability_cache
def is_torch_available() -> bool:
    r"""Indicate if the ``torch`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``torch`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_torch_available
        >>> is_torch_available()

        ```
    """
    return module_available("torch")


def check_torch() -> None:
    r"""Check if the ``torch`` package is installed.

    Raises:
        MissingPackageError: if the ``torch`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_torch
        >>> check_torch()

        ```
    """
    if not is_torch_available():
        raise_package_missing_error("torch")


@availability_cache
def is_xarray_available() -> bool:
    r"""Indicate if the ``xarray`` package is installed and can be
    imported.

    Returns:
        ``True`` if ``xarray`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_xarray_available
        >>> is_xarray_available()

        ```
    """
    return module_available("xarray")


def check_xarray() -> None:
    r"""Check if the ``xarray`` package is installed.

    Raises:
        MissingPackageError: if the ``xarray`` package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_xarray
        >>> check_xarray()

        ```
    """
    if not is_xarray_available():
        raise_package_missing_error("xarray")


@availability_cache
def is_torch_numpy_available() -> bool:
    r"""Indicate if the ``torch`` and ``numpy`` packages are installed
    and are compatible.

    The packages are compatible if a ``torch`` tensor can be converted
    to a ``numpy`` array. Unlike the other ``is_<package>_available``
    functions, this function checks a pair of packages. Use it for code
    that converts between tensors and arrays; if the code only needs
    one of the two, use ``is_torch_available`` or
    ``is_numpy_available``.

    Returns:
        ``True`` if both packages are available and compatible,
            otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_torch_numpy_available
        >>> is_torch_numpy_available()

        ```
    """
    if not is_torch_available():
        return False
    if not is_numpy_available():
        return False
    # Imported here and not at the top of the module, so importing
    # ``coola.utils.imports`` does not import ``torch``.
    import torch  # noqa: PLC0415

    with suppress(RuntimeError):
        torch.tensor([1.0]).numpy()  # Check if the libraries are compatible
        return True
    return False


def check_torch_numpy() -> None:
    r"""Check if the ``torch`` and ``numpy`` packages are installed and
    are compatible.

    Unlike the other ``check_<package>`` functions, this function
    checks a pair of packages (see ``is_torch_numpy_available``).

    Raises:
        MissingPackageError: if one of the packages is not installed or if
            they are not compatible.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_torch_numpy
        >>> check_torch_numpy()

        ```
    """
    if not is_torch_numpy_available():
        msg = (
            "'torch' and 'numpy' packages are required but one of the package is not "
            "installed or they are not compatible."
        )
        # ``name`` is the first package that is missing, or ``torch`` if both
        # are installed but not compatible.
        name = "torch" if not is_torch_available() or is_numpy_available() else "numpy"
        raise MissingPackageError(msg, name=name)
