r"""Contain fallback implementations used when ``xarray`` dependency is
not available."""

from __future__ import annotations

__all__ = ["xarray"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_error_xarray_missing


class FakeClass:
    r"""Fake class that raises an error because xarray is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: xarray is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_xarray_missing()


# Create a fake xarray package
xarray: ModuleType = ModuleType("xarray")
xarray.DataArray = FakeClass
xarray.Dataset = FakeClass
xarray.Variable = FakeClass
