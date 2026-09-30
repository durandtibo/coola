r"""Contain fallback implementations used when ``pyarrow`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pyarrow"]

from typing import TYPE_CHECKING

from coola.utils.fallback.factory import make_fake_class, make_fake_module
from coola.utils.imports import raise_pyarrow_missing_error

if TYPE_CHECKING:
    from types import ModuleType

FakeClass: type = make_fake_class(raise_pyarrow_missing_error)

# Create a fake pyarrow package
pyarrow: ModuleType = make_fake_module("pyarrow", Array=FakeClass)
