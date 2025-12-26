r"""Define the BFS child finder class for handling iterable data."""

from __future__ import annotations

__all__ = ["IterableChildFinder"]

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.base import BaseChildFinder

if TYPE_CHECKING:
    from collections.abc import Iterator


class IterableChildFinder(BaseChildFinder[Iterable[Any]]):
    r"""Child finder for iterable objects.

    This child finder handles iterable objects by yielding each element
    of the iterable. It works with lists, tuples, sets, strings, and any
    object implementing the Iterable protocol.

    Example:
        ```pycon
        >>> from coola.iterator.bfs import IterableChildFinder
        >>> child_finder = IterableChildFinder()
        >>> list(child_finder.find_children((4, 2, 1)))
        [4, 2, 1]
        >>> list(child_finder.find_children("hello"))
        ['h', 'e', 'l', 'l', 'o']

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def find_children(self, data: Iterable[Any]) -> Iterator[Any]:
        yield from data
