r"""Contain fallback implementations used when ``colorlog`` dependency
is not available."""

from __future__ import annotations

__all__ = ["colorlog"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module
from coola.utils.imports import raise_colorlog_missing_error

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class(raise_colorlog_missing_error)

# Create a fake colorlog package
colorlog: ModuleType = make_fake_module(
    "colorlog", StreamHandler=FakeClass, ColoredFormatter=FakeClass
)
