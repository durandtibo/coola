r"""Contain fallback implementations used when ``jax`` dependency is not
available."""

from __future__ import annotations

__all__ = ["jax"]

from types import ModuleType

# Create a fake jax package
jax: ModuleType = ModuleType("jax")
