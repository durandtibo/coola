r"""General-purpose registry primitives used across the package."""

from __future__ import annotations

__all__ = [
    "AlreadyRegisteredError",
    "BaseTypeDispatchRegistry",
    "KeyNotRegisteredError",
    "Registry",
    "TypeNotRegisteredError",
    "TypeRegistry",
]

from coola.registry.dispatch import BaseTypeDispatchRegistry
from coola.registry.exceptions import (
    AlreadyRegisteredError,
    KeyNotRegisteredError,
    TypeNotRegisteredError,
)
from coola.registry.type import TypeRegistry
from coola.registry.vanilla import Registry
