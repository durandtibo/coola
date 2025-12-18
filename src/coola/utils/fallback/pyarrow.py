r"""Contain fallback implementations used when ``pyarrow`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pyarrow"]

from types import ModuleType

# Create a fake pyarrow package
pyarrow: ModuleType = ModuleType("pyarrow")
