r"""Define the child finder class for iterable data."""

from __future__ import annotations

__all__ = ["IterableChildFinder"]

from collections.abc import Iterable
from typing import Any

from coola.display import InlineDisplayMixin
from coola.iterator.base import BaseChildFinder


class IterableChildFinder(InlineDisplayMixin, BaseChildFinder[Iterable[Any]]):
    r"""Child finder for the iterable containers.

    The container is its own iterable of children: its elements are
    visited in the order of its iteration. It works with lists, tuples,
    sets and any object implementing the ``Iterable`` protocol.

    Example:
        ```pycon
        >>> from coola.iterator import IterableChildFinder
        >>> child_finder = IterableChildFinder()
        >>> list(child_finder.find_children((4, 2, 1)))
        [4, 2, 1]
        >>> list(child_finder.find_children([]))
        []

        ```
    """

    def find_children(self, data: Iterable[Any]) -> Iterable[Any]:
        return data
