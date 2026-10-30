r"""Implement a summarizer for sequences."""

from __future__ import annotations

__all__ = ["SequenceSummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.base import BaseSummarizer, Node

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.summary.config import SummaryConfig


class SequenceSummarizer(BaseSummarizer["Sequence[Any]"]):
    r"""Implement a summarizer for sequences (``list``, ``tuple``,
    ``deque``, ...).

    The children are the items, labeled by their position, or by their
    field name for a named tuple. The items are taken lazily, so only
    ``max_items`` of them are read.

    Example:
        ```pycon
        >>> from coola.summary import SequenceSummarizer, SummaryConfig
        >>> node = SequenceSummarizer().summarize([1, 2, 3], SummaryConfig(), compact=False)
        >>> node.header
        "<class 'list'> (length=3)"
        >>> list(node.children)
        [(0, 1), (1, 2), (2, 3)]
        >>> node.total
        3

        ```
    """

    def summarize(self, data: Sequence[Any], config: SummaryConfig, *, compact: bool) -> Node:  # noqa: ARG002
        r"""Summarize a sequence.

        Args:
            data: The data to summarize.
            config: The options of the summary.
            compact: If ``True``, the summary is a single line, as the
                node is not expanded.

        Returns:
            The node whose children are the items, numbered by their
                index (named by their field for a named tuple).
        """
        length = len(data)
        fields = getattr(data, "_fields", None) if isinstance(data, tuple) else None
        children = enumerate(data) if fields is None else zip(fields, data, strict=True)
        return Node(f"{type(data)} (length={length:,})", children, length)
