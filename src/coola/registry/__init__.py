r"""General-purpose registry primitives used across the package.

``Registry`` is a thread-safe key-value container and ``TypeRegistry`` adds
the lookup of the most specific registered type (Method Resolution Order).
Both are generic, keep the insertion order, return snapshots from ``items``,
``keys`` and ``values``, raise ``AlreadyRegisteredError`` on a duplicate
unless ``exist_ok=True``, copy independently with ``copy``, survive
``os.fork`` and load the entries of optional libraries lazily
(``register_lazy``). See the registry user guide for the details.
"""

from __future__ import annotations

__all__ = [
    "AlreadyRegisteredError",
    "KeyNotRegisteredError",
    "Registry",
    "TypeNotRegisteredError",
    "TypeRegistry",
]

from coola.registry.exceptions import (
    AlreadyRegisteredError,
    KeyNotRegisteredError,
    TypeNotRegisteredError,
)
from coola.registry.registry import Registry
from coola.registry.type import TypeRegistry
