r"""Define the BFS child finder class for handling iterable data."""

from __future__ import annotations

__all__ = ["MappingChildFinder"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.base import BaseChildFinder

if TYPE_CHECKING:
    from collections.abc import Iterator


class MappingChildFinder(BaseChildFinder[Mapping[Any, Any]]):
    r"""Child finder for iterable objects.

    This child finder handles iterable objects by yielding each element
    of the iterable. It works with lists, tuples, sets, strings, and any
    object implementing the Mapping protocol.

    Example:
        ```pycon
        >>> from coola.iterator.bfs import MappingChildFinder
        >>> child_finder = MappingChildFinder()
        >>> list(child_finder.find_children({"a": 1, "b": 2}))
        [1, 2]
        >>> list(child_finder.find_children({"a": {"b": 1, "c": 2}, "d": 3}))
        [{'b': 1, 'c': 2}, 3]

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def find_children(self, data: Mapping[Any, Any]) -> Iterator[Any]:
        yield from data.values()
