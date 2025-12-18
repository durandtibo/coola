from __future__ import annotations

from types import ModuleType

from coola.utils.fallback.pyarrow import pyarrow


def test_pyarrow() -> None:
    isinstance(pyarrow, ModuleType)
