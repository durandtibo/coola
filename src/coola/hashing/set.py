r"""Define the set hasher."""

from __future__ import annotations

__all__ = ["SetHasher"]

from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class SetHasher(InlineDisplayMixin, BaseHasher[AbstractSet[Any]]):
    r"""Hasher for set types.

    This hasher hashes each item via the registry, sorts the item
    hashes, concatenates them, and hashes the result under a type tag.
    Sorting by item hash (not by item) makes the result independent of
    iteration order without requiring items to be orderable.

    The tag identifies the set family, so a ``set`` and a ``frozenset``
    with the same items hash differently, and neither collides with a
    ``list`` of the same items.

    Args:
        tag: The type tag mixed into the hash. Defaults to ``"set"``.

    Example:
        ```pycon
        >>> from coola.hashing import SetHasher, StrHasher, HasherRegistry
        >>> registry = HasherRegistry({object: StrHasher()})
        >>> hasher = SetHasher()
        >>> hasher
        SetHasher(tag='set')
        >>> len(hasher.hash({1, 2, 3}, registry=registry))
        64

        ```
    """

    def __init__(self, tag: str = "set") -> None:
        self._tag = tag

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"tag": self._tag}

    def hash(
        self,
        data: AbstractSet[Any],
        registry: HasherRegistry,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        r"""Compute a deterministic hash of a set.

        Args:
            data: The set to hash.
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
        parts = sorted(
            registry.hash(item, length=length, ignore_unhashable=ignore_unhashable)
            for item in data
        )
        return hash_tagged(self._tag, "".join(parts), length=length)
