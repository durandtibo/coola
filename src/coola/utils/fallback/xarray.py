r"""Contain fallback implementations used when ``xarray`` dependency is
not available."""

from __future__ import annotations

__all__ = ["xarray"]

from types import ModuleType

from coola.utils.fallback.factory import make_fake_class
from coola.utils.imports import raise_xarray_missing_error

FakeClass = make_fake_class(raise_xarray_missing_error)

# Create a fake xarray package
xarray: ModuleType = ModuleType("xarray")
xarray.DataArray = FakeClass
xarray.Dataset = FakeClass
xarray.Variable = FakeClass
