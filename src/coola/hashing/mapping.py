r"""Define the mapping hasher."""

from __future__ import annotations

__all__ = ["MappingHasher"]

from collections.abc import Mapping
from typing import TYPE_CHECKING

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class MappingHasher(BaseHasher[Mapping]):
    r"""Hasher for mapping types.

    This hasher sorts the mapping by key, hashes each key and value
    separately via the registry, concatenates the key and value hashes
    per item, concatenates all per-item strings, and hashes the result.

    Sorting by key ensures that mappings with the same key-value pairs
    but different insertion orders produce the same hash.

    This hasher handles any type that is an instance of
    ``collections.abc.Mapping``, including ``dict``.

    Example:
        ```pycon
        >>> from collections.abc import Mapping
        >>> from coola.hashing import MappingHasher, DefaultHasher, HasherRegistry
        >>> registry = HasherRegistry({object: DefaultHasher(), Mapping: MappingHasher()})
        >>> hasher = MappingHasher()
        >>> hasher
        MappingHasher()
        >>> hasher.hash({"a": 1, "b": 2}, registry=registry)
        'a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f'

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(self, data: Mapping, registry: HasherRegistry, length: int = 64) -> str:
        parts = []
        for key in sorted(data.keys()):
            key_hash = registry.hash(key, length=length)
            val_hash = registry.hash(data[key], length=length)
            parts.append(key_hash + val_hash)
        return hash_string("".join(parts), length=length)
