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

    def hash(
        self,
        data: Sequence[Any],
        registry: HasherRegistry,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        r"""Compute a deterministic hash of a sequence.

        Args:
            data: The sequence to hash.
            registry: The hasher registry used to hash each item, and
                to recurse into any nested data structures.
            length: The desired length of the returned hex string. Must
                be an even number between 2 and 128 inclusive. Defaults
                to 64.
            ignore_unhashable: If ``True``, items for which no hasher
                is registered are replaced by a deterministic
                placeholder hash instead of raising an error.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            KeyError: If an item has a type for which no hasher is
                registered in ``registry`` and ``ignore_unhashable`` is
                ``False``.
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        intermediate = "".join(
            registry.hash(item, length=length, ignore_unhashable=ignore_unhashable) for item in data
        )
        return hash_string(intermediate, length=length)
