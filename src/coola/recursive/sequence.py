r"""Define the default transformer for sequences data (list, tuple)."""

from __future__ import annotations

__all__ = ["SequenceTransformer"]

from collections.abc import Sequence
from typing import Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer, Expansion
from coola.recursive.rebuild import rebuild_sequence


class SequenceTransformer(InlineDisplayMixin, BaseTransformer[Sequence[Any]]):
    r"""Transformer for sequence types that recursively transforms
    elements.

    This transformer handles sequence structures (list, tuple, namedtuple, etc.)
    by recursively transforming all elements while preserving the original
    sequence type and structure. After transformation, it reconstructs the
    sequence using its original type, with special handling for named tuples
    to preserve their field structure.

    Notes:
        - All elements are transformed recursively by the engine
        - The original sequence type is preserved (list remains list, tuple remains tuple)
        - Named tuples receive special handling to preserve field names
        - Nested sequences and other containers are handled recursively
        - Empty sequences are preserved as empty sequences of the same type
        - String sequences (str) should typically use a different transformer
          as they are often treated as atomic values

    Example:
        ```pycon
        >>> from coola.recursive import SequenceTransformer, TransformerRegistry, DefaultTransformer
        >>> transformer = SequenceTransformer()
        >>> transformer
        SequenceTransformer()
        >>> children, rebuild = transformer.expand((1, 2, 3))
        >>> children
        [1, 2, 3]
        >>> rebuild(["1", "2", "3"])
        ('1', '2', '3')
        >>> registry = TransformerRegistry({object: DefaultTransformer(), list: transformer})
        >>> # Transform list elements
        >>> registry.transform([1, 2, 3], str)
        ['1', '2', '3']
        >>> # Nested sequences are handled recursively
        >>> registry.transform([[1, 2], [3, 4]], str)
        [['1', '2'], ['3', '4']]
        >>> # Empty sequences are preserved
        >>> registry.transform([], str)
        []
        >>> # Named tuples preserve their structure
        >>> from collections import namedtuple
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> TransformerRegistry({object: DefaultTransformer(), Point: transformer}).transform(
        ...     Point(1, 2), str
        ... )
        Point(x='1', y='2')

        ```
    """

    def expand(self, data: Sequence[Any]) -> Expansion | None:
        return list(data), lambda items: rebuild_sequence(data, items)
