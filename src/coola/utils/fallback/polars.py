r"""Contain fallback implementations used when ``polars`` dependency is
not available."""

from __future__ import annotations

__all__ = ["polars"]

from types import ModuleType

from coola.utils.fallback._factory import make_fake_class
from coola.utils.imports import raise_polars_missing_error

FakeClass = make_fake_class(raise_polars_missing_error)

# Create a fake polars package
polars: ModuleType = ModuleType("polars")
polars.DataFrame = FakeClass
polars.LazyFrame = FakeClass
polars.Series = FakeClass
