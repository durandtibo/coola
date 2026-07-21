r"""Contain fallback implementations used when ``numpy`` dependency is
not available."""

from __future__ import annotations

__all__ = ["numpy"]

from types import ModuleType

from coola.utils.fallback.factory import make_fake_class
from coola.utils.imports import raise_numpy_missing_error

FakeClass = make_fake_class(raise_numpy_missing_error)

# Create a fake numpy package
numpy: ModuleType = ModuleType("numpy")
numpy.ma = ModuleType("numpy.ma")
numpy.ma.MaskedArray = FakeClass
numpy.ndarray = FakeClass
