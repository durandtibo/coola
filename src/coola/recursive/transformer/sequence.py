r"""Define the default transformer for sequences data (list, tuple)."""

from __future__ import annotations

__all__ = ["SequenceTransformer"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from coola.recursive.transformer.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive.registry import TransformerRegistry


class SequenceTransformer(BaseTransformer[Sequence[Any]]):
    r"""Transformer for sequence types that recursively transforms
    elements.

    This transformer handles sequence structures (list, tuple, namedtuple, etc.)
    by recursively transforming all elements while preserving the original
    sequence type and structure. After transformation, it reconstructs the
    sequence using its original type, with special handling for named tuples
    to preserve their field structure.

    Type Parameters:
        Handles Sequence[Any], supporting any sequence type with arbitrary
        element types.

    Notes:
        - All elements are transformed recursively through the registry
        - The original sequence type is preserved (list remains list, tuple remains tuple)
        - Named tuples receive special handling to preserve field names
        - Nested sequences and other containers are handled recursively
        - Empty sequences are preserved as empty sequences of the same type
        - String sequences (str) should typically use a different transformer
          as they are often treated as atomic values

    Example usage:

    ```pycon
    >>> from coola.recursive.transformer import SequenceTransformer
    >>> from coola.recursive import TransformerRegistry
    >>> transformer = SequenceTransformer()
    >>> transformer
    SequenceTransformer()
    >>> registry = TransformerRegistry({list: transformer})
    >>> # Transform list elements
    >>> transformer.transform([1, 2, 3], func=str, registry=registry)
    ['1', '2', '3']
    >>> # Tuple type is preserved
    >>> transformer.transform((1, 2, 3), func=lambda x: x * 2, registry=registry)
    (2, 4, 6)
    >>> # Nested sequences are handled recursively
    >>> transformer.transform([[1, 2], [3, 4]], func=str, registry=registry)
    [['1', '2'], ['3', '4']]
    >>> # Empty sequences are preserved
    >>> transformer.transform([], func=str, registry=registry)
    []
    >>> # Named tuples preserve their structure
    >>> from collections import namedtuple
    >>> Point = namedtuple("Point", ["x", "y"])
    >>> transformer.transform(Point(1, 2), func=str, registry=registry)
    Point(x='1', y='2')

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        data: Sequence[Any],
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Sequence[Any]:
        # Transform all elements recursively using the registry
        transformed = [registry.transform(item, func) for item in data]

        # Rebuild with original type
        if isinstance(data, tuple):
            # Handle named tuples - preserve field structure
            if hasattr(data, "_fields"):
                return type(data)(*transformed)
            # Regular tuple
            return tuple(transformed)

        # For lists and other sequence types
        return type(data)(transformed)
