r"""Implement a summarizer for sets."""

from __future__ import annotations

__all__ = ["SetSummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.base import BaseSummarizer, Node
from coola.utils.selection import smallest_items

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from coola.summary.config import SummaryConfig


class SetSummarizer(BaseSummarizer["AbstractSet[Any]"]):
    r"""Implement a summarizer for sets (``set``, ``frozenset``).

    The children are the ``max_items`` smallest items, in order, so the
    summary does not depend on the iteration order of the set (and on
    ``PYTHONHASHSEED``). The items that cannot be compared (for example
    an ``int`` and a ``str``) are shown in iteration order.

    Example:
        ```pycon
        >>> from coola.summary import SetSummarizer, SummaryConfig
        >>> node = SetSummarizer().summarize({3, 1, 2}, SummaryConfig(max_items=2), compact=False)
        >>> node.header
        "<class 'set'> (length=3)"
        >>> list(node.children)
        [(0, 1), (1, 2)]
        >>> node.total
        3

        ```
    """

    def summarize(self, data: AbstractSet[Any], config: SummaryConfig, *, compact: bool) -> Node:  # noqa: ARG002
        length = len(data)
        children = enumerate(smallest_items(data, limit=config.max_items))
        return Node(f"{type(data)} (length={length:,})", children, length)
