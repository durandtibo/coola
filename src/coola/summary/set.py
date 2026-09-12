r"""Implement summarizers for set data."""

from __future__ import annotations

__all__ = ["SetSummarizer"]

from collections.abc import Set as AbstractSet
from itertools import islice
from typing import TYPE_CHECKING, Any

from coola.summary.collection import BaseCollectionSummarizer
from coola.utils.format import str_sequence

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry


class SetSummarizer(BaseCollectionSummarizer[AbstractSet[Any]]):
    r"""Summarizer for set-based data structures.

    This class formats set types (set, frozenset, etc.) into readable
    summaries that display the type, length, and items with proper
    indentation. It respects the max_items limit and handles nested structures
    through the registry system.

    This class creates a multi-line summary showing the set's type,
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
        >>> from coola.summary import SummarizerRegistry, SetSummarizer, DefaultSummarizer
        >>> registry = SummarizerRegistry({object: DefaultSummarizer()})
        >>> summarizer = SetSummarizer()
        >>> output = summarizer.summarize({1}, registry)
        >>> print(output)
        <class 'set'> (length=1)
          (0): 1

        ```
    """

    def _get_preview(self, data: AbstractSet[Any], limit: int) -> list:
        return list(islice(iter(data), limit))

    def _format_items(
        self,
        data: AbstractSet[Any],
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
