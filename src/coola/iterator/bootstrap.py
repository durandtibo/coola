r"""Shared bootstrap helpers for the BFS and DFS default registries.

``coola.iterator.bfs`` and ``coola.iterator.dfs`` each expose a
``get_default_registry()`` function backed by a lazily-built singleton
registry pre-populated with handlers (child finders or iterators) for
the same set of common Python types. This module centralizes that common
bootstrap logic so the two packages only need to supply their own
handler instances.
"""

from __future__ import annotations

__all__ = ["SupportsRegisterMany", "register_default_handlers"]

from collections.abc import Iterable, Mapping
from typing import Protocol, TypeVar

T = TypeVar("T")

# Scalar types are treated as leaves: they never need to be expanded further.
_SCALAR_TYPES = (object, str, bytes, int, float, complex, bool)
# Iterables (excluding mappings) are expanded into their elements.
_ITERABLE_TYPES = (list, tuple, range, Iterable, set, frozenset)
# Mappings are expanded into their values.
_MAPPING_TYPES = (dict, Mapping)


class SupportsRegisterMany(Protocol):
    r"""Protocol for registries that ``register_default_handlers`` can
    populate.

    Both ``ChildFinderRegistry`` and ``IteratorRegistry`` satisfy this
    protocol structurally, without needing to inherit from it.
    """

    def register_many(
        self, mapping: Mapping[type, T], exist_ok: bool = False
    ) -> None: ...  # pragma: no cover


def register_default_handlers(
    registry: SupportsRegisterMany,
    default_handler: T,
    iterable_handler: T,
    mapping_handler: T,
) -> None:
    r"""Register the default per-type handlers shared by the BFS and DFS
    default registries.

    Args:
        registry: The registry to populate. It must expose a
            ``register_many`` method (e.g. ``ChildFinderRegistry`` or
            ``IteratorRegistry``).
        default_handler: The handler used for scalar/leaf types
            (``object``, ``str``, ``int``, etc.).
        iterable_handler: The handler used for iterable types
            (``list``, ``tuple``, ``set``, etc.).
        mapping_handler: The handler used for mapping types
            (``dict``, ``Mapping``).
    """
    mapping: dict[type, T] = {}
    mapping.update(dict.fromkeys(_SCALAR_TYPES, default_handler))
    mapping.update(dict.fromkeys(_ITERABLE_TYPES, iterable_handler))
    mapping.update(dict.fromkeys(_MAPPING_TYPES, mapping_handler))
    registry.register_many(mapping)
