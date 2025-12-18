r"""Contain fallback implementations used when ``pandas`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pandas"]

from types import ModuleType

# Create a fake pandas package
pandas: ModuleType = ModuleType("pandas")
