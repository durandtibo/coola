from __future__ import annotations

from types import ModuleType

from coola.utils.fallback.pandas import pandas


def test_pandas() -> None:
    isinstance(pandas, ModuleType)
