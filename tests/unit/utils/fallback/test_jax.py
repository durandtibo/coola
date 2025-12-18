from __future__ import annotations

from types import ModuleType

from coola.utils.fallback.jax import jax


def test_jax() -> None:
    isinstance(jax, ModuleType)
