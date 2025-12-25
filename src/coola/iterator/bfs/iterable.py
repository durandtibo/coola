r"""Define the BFS child finder class for handling iterable data."""

from __future__ import annotations

__all__ = ["IterableChildFinder"]

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.base import BaseChildFinder

if TYPE_CHECKING:
    from collections.abc import Iterator


class IterableChildFinder(BaseChildFinder[Iterable[Any]]):
    r"""Iterable child finder for breadth-first search traversal.

    This child finder handles iterable objects during breadth-first search
    traversal by yielding each element of the iterable as a direct child.
    It allows the BFS algorithm to explore each item in the iterable at
    the current depth level before moving deeper into the tree structure.

    The IterableChildFinder is used for:
    - Lists, tuples, and other sequence types
    - Sets and other collection types
    - Any object implementing the Iterable protocol
    - Strings (which yield individual characters)

    Examples:
    ```pycon
    >>> from coola.iterator.bfs import IterableChildFinder
    >>> iterator = IterableChildFinder()
    >>> list(iterator.find_children((4, 2, 1)))
    [4, 2, 1]
    >>> list(iterator.find_children("hello"))
    ['h', 'e', 'l', 'l', 'o']

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def find_children(self, data: Iterable[Any]) -> Iterator[Any]:
        yield from data
