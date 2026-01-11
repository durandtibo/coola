r"""Implement summarizers for set data."""

from __future__ import annotations

__all__ = ["SetSummarizer"]

from collections.abc import Set as AbstractSet
from itertools import islice
from typing import TYPE_CHECKING, Any

from coola.summary.collection import BaseCollectionSummarizer
from coola.utils.format import str_indent, str_sequence

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

    def summarize(
        self,
        data: AbstractSet[Any],
        registry: SummarizerRegistry,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        if depth >= max_depth:
            return registry.summarize(str(data), depth=depth + 1, max_depth=max_depth)
        typ = type(data)
        length = len(data)
        if length == 0:
            return str_indent(f"{typ} {data}", num_spaces=self._num_spaces)
        if self._max_items == 0:
            return str_indent(f"{typ} (length={length:,}) ...", num_spaces=self._num_spaces)

        if self._max_items > 0:
            data = islice(data, self._max_items)
        data = str_sequence(
            [registry.summarize(value, depth=depth + 1, max_depth=max_depth) for value in data],
            num_spaces=self._num_spaces,
        )
        if length > self._max_items and self._max_items > 0:
            data = f"{data}\n..."
        return str_indent(f"{typ} (length={length:,})\n{data}", num_spaces=self._num_spaces)
