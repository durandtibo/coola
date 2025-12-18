r"""Contain fallback implementations used when ``xarray`` dependency is
not available."""

from __future__ import annotations

__all__ = ["xarray"]

from types import ModuleType

# Create a fake xarray package
xarray: ModuleType = ModuleType("xarray")
