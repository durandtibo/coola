r"""Contain fallback implementations used when ``scipy`` dependency is
not available."""

from __future__ import annotations

__all__ = ["scipy"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class("scipy")

# Create a fake scipy package
scipy: ModuleType = make_fake_module(
    "scipy", sparse=make_fake_module("scipy.sparse", sparray=FakeClass, spmatrix=FakeClass)
)
