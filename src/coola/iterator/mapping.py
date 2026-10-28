r"""Define the child finder class for mapping data."""

from __future__ import annotations

__all__ = ["MappingChildFinder"]

from collections.abc import Iterable, Mapping
from typing import Any

from coola.display import InlineDisplayMixin
from coola.iterator.base import BaseChildFinder


class MappingChildFinder(InlineDisplayMixin, BaseChildFinder[Mapping[Any, Any]]):
    r"""Child finder for the mappings.

    The children of a mapping are its values, in the order of its
    iteration. The keys are never visited. It works with dictionaries
    and any object implementing the ``Mapping`` protocol.

    Example:
        ```pycon
        >>> from coola.iterator import MappingChildFinder
        >>> child_finder = MappingChildFinder()
        >>> list(child_finder.find_children({"a": 1, "b": 2}))
        [1, 2]
        >>> list(child_finder.find_children({"a": {"b": 1, "c": 2}, "d": 3}))
        [{'b': 1, 'c': 2}, 3]

        ```
    """

    def find_children(self, data: Mapping[Any, Any]) -> Iterable[Any]:
        return data.values()

    def find_items(self, data: Mapping[Any, Any]) -> Iterable[tuple[Any, Any]]:
        return data.items()
