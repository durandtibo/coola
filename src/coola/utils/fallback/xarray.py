r"""Contain fallback implementations used when ``xarray`` dependency is
not available."""

from __future__ import annotations

__all__ = ["xarray"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class("xarray")

# Create a fake xarray package
xarray: ModuleType = make_fake_module(
    "xarray", DataArray=FakeClass, Dataset=FakeClass, Variable=FakeClass
)
