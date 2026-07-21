r"""Contain fallback implementations used when ``pyarrow`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pyarrow"]

from types import ModuleType

from coola.utils.fallback.factory import make_fake_class
from coola.utils.imports import raise_pyarrow_missing_error

FakeClass = make_fake_class(raise_pyarrow_missing_error)

# Create a fake pyarrow package
pyarrow: ModuleType = ModuleType("pyarrow")
pyarrow.Array = FakeClass
