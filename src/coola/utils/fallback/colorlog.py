r"""Contain fallback implementations used when ``colorlog`` dependency
is not available."""

from __future__ import annotations

__all__ = ["colorlog"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class("colorlog")

# Create a fake colorlog package
colorlog: ModuleType = make_fake_module(
    "colorlog", StreamHandler=FakeClass, ColoredFormatter=FakeClass
)
