r"""Define a DFS iterator class for handling mapping data in
traversal."""

from __future__ import annotations

__all__ = ["MappingIterator"]

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.base import BaseIterator

if TYPE_CHECKING:
    from coola.iterator.dfs.registry import IteratorRegistry


class MappingIterator(BaseIterator[Mapping[Any, Any]]):
    r"""Iterator for depth-first traversal of mapping data structures.

    This iterator handles dict-like objects by recursively iterating over
    their values. Keys are not yielded during iteration - only the values
    are traversed. If values contain nested structures (lists, dicts, etc.),
    those are recursively iterated as well.

    Example:
        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, MappingIterator, IterableIterator
        >>> iterator = MappingIterator()
        >>> registry = IteratorRegistry({dict: iterator, list: IterableIterator()})
        >>> # Simple dictionary with scalar values
        >>> list(iterator.iterate({"a": 1, "b": 2}, registry))
        [1, 2]
        >>> # Nested dictionary with scalar values
        >>> list(iterator.iterate({"a": {"b": 1, "c": 2}, "d": 3}, registry))
        [1, 2, 3]
        >>> # Dictionary with list values
        >>> list(iterator.iterate({"x": [1, 2], "y": [3, 4]}, registry))
        [1, 2, 3, 4]

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(self, data: Mapping[Any, Any], registry: IteratorRegistry) -> Iterator[Any]:
        for value in data.values():
            yield from registry.iterate(value)
