from __future__ import annotations

from types import ModuleType

from coola.utils.fallback.numpy import numpy


def test_numpy() -> None:
    isinstance(numpy, ModuleType)
