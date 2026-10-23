r"""Define the mapping hasher."""

from __future__ import annotations

__all__ = ["MappingHasher"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class MappingHasher(InlineDisplayMixin, BaseHasher[Mapping[Any, Any]]):
    r"""Hasher for mapping types.

    This hasher hashes each key and value separately via the registry,
    concatenates the key and value hashes per item, sorts the items by
    key hash, concatenates all per-item strings, and hashes the result
    under a type tag.

    Sorting by key hash (not by key) ensures that mappings with the
    same key-value pairs but different insertion orders produce the
    same hash, without requiring keys to be orderable. This matches
    ``coola`` equality, which ignores the order of the keys of a
    ``dict``. Register the hasher with ``ordered=True`` for a mapping
    type whose order is meaningful, like ``OrderedDict``: its equality
    is order-sensitive too.

    The tag identifies the mapping family, so a ``dict`` and an
    ``OrderedDict`` with the same items hash differently.

    Args:
        tag: The type tag mixed into the hash. Defaults to
            ``"mapping"``.
        ordered: If ``True``, items are hashed in iteration order
            instead of being sorted. Defaults to ``False``.

    Example:
        ```pycon
        >>> from collections.abc import Mapping
        >>> from coola.hashing import MappingHasher, StrHasher, HasherRegistry
        >>> registry = HasherRegistry({object: StrHasher(), Mapping: MappingHasher()})
        >>> hasher = MappingHasher()
        >>> hasher
        MappingHasher(tag='mapping', ordered=False)
        >>> len(hasher.hash({"a": 1, "b": 2}, registry=registry))
        64

        ```
    """

    def __init__(self, tag: str = "mapping", ordered: bool = False) -> None:
        self._tag = tag
        self._ordered = ordered

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"tag": self._tag, "ordered": self._ordered}

    def hash(
        self,
        data: Mapping[Any, Any],
        registry: HasherRegistry,
        *,
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
        items: list[str] = []
        for key, value in data.items():
            key_hash = registry.hash(key, length=length, ignore_unhashable=ignore_unhashable)
            val_hash = registry.hash(value, length=length, ignore_unhashable=ignore_unhashable)
            items.append(key_hash + val_hash)
        if not self._ordered:
            # Sort by the full entry: keys can share a hash (placeholder of
            # unhashable keys, short ``length``), and equal mappings must
            # not depend on the insertion order.
            items.sort()
        return hash_tagged(self._tag, "".join(items), length=length)
