r"""Contain fallback implementations used when ``pandas`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pandas"]

from types import ModuleType

from coola.utils.fallback._factory import make_fake_class
from coola.utils.imports import raise_pandas_missing_error

FakeClass = make_fake_class(raise_pandas_missing_error)

# Create a fake pandas package
pandas: ModuleType = ModuleType("pandas")
pandas.DataFrame = FakeClass
pandas.Series = FakeClass
