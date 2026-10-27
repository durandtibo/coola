r"""Define the mapping hasher."""

from __future__ import annotations

__all__ = ["MappingHasher"]

from collections.abc import Mapping
from itertools import chain
from typing import Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node


class MappingHasher(InlineDisplayMixin, BaseHasher[Mapping[Any, Any]]):
    r"""Hasher for mapping types.

    This hasher describes a mapping as a ``Node`` whose children are
    the keys and the values, which the engine hashes through the
    registry. The entries (a key and its value) are sorted by digest,
    not by key, so mappings with the same key-value pairs but different
    insertion orders produce the same hash, without requiring keys to be
    orderable. This matches
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
        >>> from coola.hashing import MappingHasher
        >>> hasher = MappingHasher()
        >>> hasher
        MappingHasher(tag='mapping', ordered=False)
        >>> hasher.encode({"a": 1}).tag
        'mapping'

        ```
    """

    def __init__(self, tag: str = "mapping", ordered: bool = False) -> None:
        self._tag = tag
        self._ordered = ordered

    def _display_fields(self) -> dict[str, Any]:
        return {"tag": self._tag, "ordered": self._ordered}

    def encode(self, data: Mapping[Any, Any]) -> Encoding:
        r"""Describe a mapping.

        Args:
            data: The mapping to describe.

        Returns:
            The ``Node`` of the keys and values (key, value, key, value,
            ...) with ``group=2``. It is unordered unless the hasher was
            built with ``ordered=True``.
        """
        return Node(self._tag, chain.from_iterable(data.items()), self._ordered, 2)
