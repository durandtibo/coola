r"""Define the default child finder class."""

from __future__ import annotations

__all__ = ["DefaultChildFinder"]

from typing import Any

from coola.display import InlineDisplayMixin
from coola.iterator.base import BaseChildFinder


class DefaultChildFinder(InlineDisplayMixin, BaseChildFinder[Any]):
    r"""Default child finder: the data is a leaf.

    This child finder is the fallback for the objects that do not have
    a specialized child finder registered. It is used for the
    primitive types (``int``, ``float``, ``str``, ``bool``, ...), the
    iterators and every object without a structure to traverse.

    Example:
        ```pycon
        >>> from coola.iterator import DefaultChildFinder
        >>> child_finder = DefaultChildFinder()
        >>> print(child_finder.find_children(42))
        None
        >>> print(child_finder.find_children("hello"))
        None

        ```
    """

    def find_children(self, data: Any) -> None:
        r"""Return the children of an object.

        Args:
            data: The object.

        Returns:
            ``None``: the object is a leaf.
        """
        del data  # a leaf: no children
