from __future__ import annotations

from types import ModuleType

from coola.utils.fallback.xarray import xarray


def test_xarray() -> None:
    isinstance(xarray, ModuleType)
