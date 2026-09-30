r"""Contain fallback implementations used when ``numpy`` dependency is
not available."""

from __future__ import annotations

__all__ = ["numpy"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module
from coola.utils.imports import raise_numpy_missing_error

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class(raise_numpy_missing_error)

# Create a fake numpy package
numpy: ModuleType = make_fake_module(
    "numpy",
    ma=make_fake_module("numpy.ma", MaskedArray=FakeClass),
    ndarray=FakeClass,
)
