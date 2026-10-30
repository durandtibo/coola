r"""Implement a summarizer for mappings."""

from __future__ import annotations

__all__ = ["MappingSummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.base import BaseSummarizer, Node

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.summary.config import SummaryConfig


class MappingSummarizer(BaseSummarizer["Mapping[Any, Any]"]):
    r"""Implement a summarizer for mappings (``dict``, ``Mapping``,
    ``MappingProxyType``, ...).

    The children are the ``(key, value)`` pairs, in the order of the
    mapping. The pairs are taken lazily, so only ``max_items`` of them
    are read.

    Example:
        ```pycon
        >>> from coola.summary import MappingSummarizer, SummaryConfig
        >>> node = MappingSummarizer().summarize({"a": 1, "b": 2}, SummaryConfig(), compact=False)
        >>> node.header
        "<class 'dict'> (length=2)"
        >>> list(node.children)
        [('a', 1), ('b', 2)]
        >>> node.total
        2

        ```
    """

    def summarize(self, data: Mapping[Any, Any], config: SummaryConfig, *, compact: bool) -> Node:  # noqa: ARG002
        length = len(data)
        return Node(f"{type(data)} (length={length:,})", data.items(), length)
