r"""Define an iterator class for handling iterable data in depth-first
traversal."""

from __future__ import annotations

__all__ = ["IterableIterator"]

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.base import BaseIterator

if TYPE_CHECKING:
    from coola.iterator.dfs.registry import IteratorRegistry


class IterableIterator(BaseIterator[Iterable[Any]]):
    r"""Iterator for performing a depth-first traversal over iterable
    data structures.

    This iterator recursively traverses through iterable structures
    such as lists, tuples, or other collections that implement the
    `Iterable` interface, yielding elements one by one.

    Example:
        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator
        >>> iterator = IterableIterator()
        >>> registry = IteratorRegistry({list: iterator})
        >>> # Iterating over a simple list
        >>> list(iterator.iterate([1, 2, 3], registry))
        [1, 2, 3]
        >>> # Iterating over a string (iterable of characters)
        >>> list(iterator.iterate("hello", registry))
        ['h', 'e', 'l', 'l', 'o']
        >>> # Iterating over nested iterables (lists within lists)
        >>> list(iterator.iterate([[1, 2, 3], [4, 5, 6]], registry))
        [1, 2, 3, 4, 5, 6]

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(self, data: Iterable[Any], registry: IteratorRegistry) -> Iterator[Any]:
        for item in data:
            yield from registry.iterate(item)
