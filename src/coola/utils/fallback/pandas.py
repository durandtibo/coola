r"""Contain fallback implementations used when ``pandas`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pandas"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class("pandas")

# Create a fake pandas package
pandas: ModuleType = make_fake_module("pandas", DataFrame=FakeClass, Series=FakeClass)
