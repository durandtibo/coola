r"""General-purpose registry primitives used across the package."""

from __future__ import annotations

__all__ = ["BaseTypeDispatchRegistry", "Registry", "TypeRegistry"]

from coola.registry.dispatch import BaseTypeDispatchRegistry
from coola.registry.type import TypeRegistry
from coola.registry.vanilla import Registry
