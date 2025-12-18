from __future__ import annotations

from types import ModuleType

from coola.utils.fallback.polars import polars


def test_polars() -> None:
    isinstance(polars, ModuleType)
