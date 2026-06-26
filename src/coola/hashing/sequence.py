r"""Define the sequence hasher."""

from __future__ import annotations

__all__ = ["SequenceHasher"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class SequenceHasher(BaseHasher[Sequence[Any]]):
    r"""Hasher for sequence types.

    This hasher computes the hash of each item in the sequence
    recursively using the registry, concatenates the intermediate hash
    strings, and then hashes the concatenated result.

    This hasher handles any type that is an instance of
    ``collections.abc.Sequence``, including ``list``, ``tuple``, and
    ``str``.

    Example:
        ```pycon
        >>> from coola.hashing import SequenceHasher, StrHasher, HasherRegistry
        >>> registry = HasherRegistry({object: StrHasher(), Sequence: SequenceHasher()})
        >>> hasher = SequenceHasher()
        >>> hasher
        SequenceHasher()
        >>> hasher.hash([1, 2, 3], registry=registry)

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(self, data: Sequence[Any], registry: HasherRegistry, length: int = 64) -> str:
        intermediate = "".join(registry.hash(item, length=length) for item in data)
        return hash_string(intermediate, length=length)
