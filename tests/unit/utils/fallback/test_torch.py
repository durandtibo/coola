from __future__ import annotations

from types import ModuleType

from coola.utils.fallback.torch import torch


def test_torch() -> None:
    isinstance(torch, ModuleType)
