r"""Implement summarizers for sequence data."""

from __future__ import annotations

__all__ = ["SequenceSummarizer"]

from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any

from coola.summary.collection import BaseCollectionSummarizer
from coola.utils.format import str_sequence

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry


class SequenceSummarizer(BaseCollectionSummarizer[Sequence[Any]]):
    r"""Summarizer for sequence-based data structures like lists and
    tuples.

    This class formats sequence types (list, tuple, etc.) into readable
    summaries that display the type, length, and indexed items with proper
    indentation. It respects the max_items limit and handles nested structures
    through the registry system.

    This class creates a multi-line summary showing the sequence's type,
    length, and contents. It handles depth limiting to prevent excessively
    deep nested summaries and truncates the output when the number of items
    exceeds max_items.

    Args:
        max_items: The maximum number of items to display.
            If negative, shows all items. Defaults to 5.
        num_spaces: The number of spaces for indenting each level.
            Defaults to 2.

    Example:
        ```pycon
        >>> from coola.summary import SummarizerRegistry, SequenceSummarizer, DefaultSummarizer
        >>> registry = SummarizerRegistry({object: DefaultSummarizer()})
        >>> summarizer = SequenceSummarizer()
        >>> output = summarizer.summarize([1, 2, 3], registry)
        >>> print(output)
        <class 'list'> (length=3)
          (0): 1
          (1): 2
          (2): 3

        ```
    """

    def _get_preview(self, data: Sequence[Any]) -> list:
        return list(islice(data, self._max_items))

    def _format_items(
        self,
        data: Sequence[Any],
        registry: SummarizerRegistry,
        depth: int,
        max_depth: int,
    ) -> str:
        items = data
        if self._max_items > 0:
            items = islice(items, self._max_items)
        return str_sequence(
            [registry.summarize(value, depth=depth + 1, max_depth=max_depth) for value in items],
            num_spaces=self._num_spaces,
        )
