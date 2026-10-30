r"""Define the sequence hasher."""

from __future__ import annotations

__all__ = ["SequenceHasher"]

from collections.abc import Sequence
from typing import Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node


class SequenceHasher(InlineDisplayMixin, BaseHasher[Sequence[Any]]):
    r"""Hasher for sequence types.

    This hasher describes a sequence as an ordered ``Node`` whose
    children are the items, which the engine hashes through the
    registry. Item order matters.

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
        >>> from coola.hashing import SequenceHasher
        >>> hasher = SequenceHasher()
        >>> hasher
        SequenceHasher(tag='sequence')
        >>> hasher.encode([1, 2, 3])
        Node(tag='sequence', children=[1, 2, 3], ordered=True, group=1)

        ```
    """

    def __init__(self, tag: str = "sequence") -> None:
        self._tag = tag

    def _display_fields(self) -> dict[str, Any]:
        return {"tag": self._tag}

    def encode(self, data: Sequence[Any]) -> Encoding:
        r"""Describe a sequence.

        Args:
            data: The sequence to describe.

        Returns:
            The ordered ``Node`` of the items. A named tuple has the tag
            ``"<tag>:Class(field1,field2)"``.
        """
        fields = getattr(type(data), "_fields", None) if isinstance(data, tuple) else None
        if isinstance(fields, tuple):
            # Named tuple: the class and field names are part of the type.
            tag = f"{self._tag}:{type(data).__qualname__}({','.join(map(str, fields))})"
        else:
            tag = self._tag
        return Node(tag, data)
