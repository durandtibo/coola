r"""Define the sequence hasher."""

from __future__ import annotations

__all__ = ["SequenceHasher"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class SequenceHasher(InlineDisplayMixin, BaseHasher[Sequence[Any]]):
    r"""Hasher for sequence types.

    This hasher computes the hash of each item in the sequence
    recursively using the registry, concatenates the intermediate hash
    strings, and then hashes the concatenated result under a type tag.
    Item order matters.

    The tag identifies the sequence family, so sequences of different
    families with the same items (e.g. a ``list`` and a ``tuple``) hash
    differently. Register one instance per family, e.g.
    ``SequenceHasher("list")`` for ``list``. A named tuple is tagged
    with its class name and field names, so it differs from a plain
    ``tuple`` and from other named tuples.

    Args:
        tag: The type tag mixed into the hash. Defaults to
            ``"sequence"``.

    Example:
        ```pycon
        >>> from coola.hashing import SequenceHasher, StrHasher, HasherRegistry
        >>> registry = HasherRegistry({object: StrHasher(), Sequence: SequenceHasher()})
        >>> hasher = SequenceHasher()
        >>> hasher
        SequenceHasher(tag='sequence')
        >>> len(hasher.hash([1, 2, 3], registry=registry))
        64

        ```
    """

    def __init__(self, tag: str = "sequence") -> None:
        self._tag = tag

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"tag": self._tag}

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
        parts = [
            registry.hash(item, length=length, ignore_unhashable=ignore_unhashable) for item in data
        ]
        fields = getattr(type(data), "_fields", None)
        if isinstance(data, tuple) and isinstance(fields, tuple):
            # Named tuple: the class and field names are part of the type.
            tag = f"{self._tag}:{type(data).__qualname__}({','.join(map(str, fields))})"
        else:
            tag = self._tag
        return hash_tagged(tag, "".join(parts), length=length)
