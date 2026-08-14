r"""Define the mapping hasher."""

from __future__ import annotations

__all__ = ["MappingHasher"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class MappingHasher(BaseHasher[Mapping[Any, Any]]):
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
        >>> from coola.hashing import MappingHasher, StrHasher, HasherRegistry
        >>> registry = HasherRegistry({object: StrHasher(), Mapping: MappingHasher()})
        >>> hasher = MappingHasher()
        >>> hasher
        MappingHasher()
        >>> hasher.hash({"a": 1, "b": 2}, registry=registry)
        'a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f'

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(
        self,
        data: Mapping[Any, Any],
        registry: HasherRegistry,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        r"""Compute a deterministic hash of a mapping.

        Args:
            data: The mapping to hash.
            registry: The hasher registry used to hash each key and
                value, and to recurse into any nested data structures.
            length: The desired length of the returned hex string. Must
                be an even number between 2 and 128 inclusive. Defaults
                to 64.
            ignore_unhashable: If ``True``, keys/values for which no
                hasher is registered are replaced by a deterministic
                placeholder hash instead of raising an error.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            KeyError: If a key or value has a type for which no hasher
                is registered in ``registry`` and ``ignore_unhashable``
                is ``False``.
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        parts = []
        for key in sorted(data.keys()):
            key_hash = registry.hash(key, length=length, ignore_unhashable=ignore_unhashable)
            val_hash = registry.hash(data[key], length=length, ignore_unhashable=ignore_unhashable)
            parts.append(key_hash + val_hash)
        return hash_string("".join(parts), length=length)
