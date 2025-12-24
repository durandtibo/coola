r"""Define a DFS iterator class for handling iterable data in
traversal."""

from __future__ import annotations

__all__ = ["IterableIterator"]

from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.base import BaseIterator

if TYPE_CHECKING:
    from coola.iterator.dfs.registry import IteratorRegistry


class IterableIterator(BaseIterator[Iterable[Any]]):
    r"""Iterable iterator for depth-first search traversal.

    Examples:
    ```pycon
    >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator
    >>> iterator = IterableIterator()
    >>> registry = IteratorRegistry({list: iterator})
    >>> list(iterator.iterate([1, 2, 3], registry))
    [1, 2, 3]
    >>> list(iterator.iterate("hello", registry))
    ['h', 'e', 'l', 'l', 'o']
    >>> list(iterator.iterate([[1, 2, 3], [4, 5, 6]], registry))
    [1, 2, 3, 4, 5, 6]

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(self, data: Iterable[Any], registry: IteratorRegistry) -> Generator[Any]:
        for item in data:
            yield from registry.iterate(item)
