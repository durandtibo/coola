from __future__ import annotations

__all__ = [
    "check_numpy",
    "check_torch",
    "check_xarray",
    "is_numpy_available",
    "is_torch_available",
    "is_xarray_available",
]

from importlib.util import find_spec


def is_numpy_available() -> bool:
    r"""Indicates if the NumPy package is installed or not."""
    return find_spec("numpy") is not None


def check_numpy() -> None:
    r"""Checks if the numpy package is installed.

    Raises:
        RuntimeError if the numpy package is not installed.
    """
    if not is_numpy_available():
        raise RuntimeError(
            "`numpy` package is required but not installed. "
            "You can install `numpy` package with the command:\n\n"
            "pip install numpy\n"
        )


def is_torch_available() -> bool:
    r"""Indicates if the torch package is installed or not."""
    return find_spec("torch") is not None


def check_torch() -> None:
    r"""Checks if the torch package is installed.

    Raises:
        RuntimeError if the torch package is not installed.
    """
    if not is_torch_available():
        raise RuntimeError(
            "`torch` package is required but not installed. "
            "You can install `torch` package with the command:\n\n"
            "pip install torch\n"
        )


def is_xarray_available() -> bool:
    r"""Indicates if the NumPy package is installed or not."""
    return find_spec("xarray") is not None


def check_xarray() -> None:
    r"""Checks if the xarray package is installed.

    Raises:
        RuntimeError if the xarray package is not installed.
    """
    if not is_xarray_available():
        raise RuntimeError(
            "`xarray` package is required but not installed. "
            "You can install `xarray` package with the command:\n\n"
            "pip install xarray\n"
        )
