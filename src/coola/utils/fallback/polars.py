r"""Contain fallback implementations used when ``polars`` dependency is
not available."""

from __future__ import annotations

__all__ = ["polars"]

from types import ModuleType

# Create a fake polars package
polars: ModuleType = ModuleType("polars")
