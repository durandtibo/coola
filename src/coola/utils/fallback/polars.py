r"""Contain fallback implementations used when ``polars`` dependency is
not available."""

from __future__ import annotations

__all__ = ["polars"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module
from coola.utils.imports import raise_polars_missing_error

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class(raise_polars_missing_error)

# Create a fake polars package
polars: ModuleType = make_fake_module(
    "polars", DataFrame=FakeClass, LazyFrame=FakeClass, Series=FakeClass
)
