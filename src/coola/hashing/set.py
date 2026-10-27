r"""Define the set hasher."""

from __future__ import annotations

__all__ = ["SetHasher"]

from collections.abc import Set as AbstractSet
from typing import Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node


class SetHasher(InlineDisplayMixin, BaseHasher[AbstractSet[Any]]):
    r"""Hasher for set types.

    This hasher describes a set as an unordered ``Node`` whose children
    are the items, which the engine hashes through the registry. The
    digests are sorted, not the items, so the result is independent of
    iteration order without requiring items to be orderable.

    The tag identifies the set family, so a ``set`` and a ``frozenset``
    with the same items hash differently, and neither collides with a
    ``list`` of the same items.

    Args:
        tag: The type tag mixed into the hash. Defaults to ``"set"``.

    Example:
        ```pycon
        >>> from coola.hashing import SetHasher
        >>> hasher = SetHasher()
        >>> hasher
        SetHasher(tag='set')
        >>> hasher.encode({1})
        Node(tag='set', children={1}, ordered=False, group=1)

        ```
    """

    def __init__(self, tag: str = "set") -> None:
        self._tag = tag

    def _display_fields(self) -> dict[str, Any]:
        return {"tag": self._tag}

    def encode(self, data: AbstractSet[Any]) -> Encoding:
        r"""Describe a set.

        Args:
            data: The set to describe.

        Returns:
            The unordered ``Node`` of the items.
        """
        return Node(self._tag, data, ordered=False)
