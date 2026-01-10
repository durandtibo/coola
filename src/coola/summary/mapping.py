r"""Implement summarizers for mapping data."""

from __future__ import annotations

__all__ = ["MappingSummarizer"]

from collections.abc import Mapping
from itertools import islice
from typing import TYPE_CHECKING, Any

from coola.summary.collection import BaseCollectionSummarizer
from coola.utils import str_indent, str_mapping

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry


class MappingSummarizer(BaseCollectionSummarizer[Mapping[Any, Any]]):
    r"""Summarizer for mapping-based data structures like dictionaries.

    This class formats mapping types (dict, OrderedDict, etc.) into readable
    summaries that display the type, length, and key-value pairs with proper
    indentation. It respects the max_items limit and handles nested structures
    through the registry system.
    This class creates a multi-line summary showing the mapping's type,
    length, and contents. It handles depth limiting to prevent excessively
    deep nested summaries and truncates the output when the number of items
    exceeds max_items.


    Args:
        max_items: The maximum number of key-value pairs to display.
            If negative, shows all pairs. Defaults to 5.
        num_spaces: The number of spaces for indenting each level.
            Defaults to 2.

    Example:
        ```pycon
        >>> from coola.summary import SummarizerRegistry, MappingSummarizer, DefaultSummarizer
        >>> registry = SummarizerRegistry({object: DefaultSummarizer()})
        >>> summarizer = MappingSummarizer()
        >>> output = summarizer.summary({"key1": 1.2, "key2": "abc", "key3": 42}, registry)
        >>> print(output)
        <class 'dict'> (length=3)
          (key1): 1.2
          (key2): abc
          (key3): 42

        ```
    """

    def summary(
        self,
        data: Mapping[Any, Any],
        registry: SummarizerRegistry,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        if depth >= max_depth:
            return registry.summary(str(data), depth=depth + 1, max_depth=max_depth)
        typ = type(data)
        length = len(data)
        if length == 0:
            return str_indent(f"{typ} {data}", num_spaces=self._num_spaces)
        if self._max_items == 0:
            return str_indent(f"{typ} (length={length:,}) ...", num_spaces=self._num_spaces)

        items = data.items()
        if self._max_items > 0:
            items = islice(items, self._max_items)
        data = str_mapping(
            {
                key: registry.summary(val, depth=depth + 1, max_depth=max_depth)
                for key, val in items
            },
            num_spaces=self._num_spaces,
        )
        if length > self._max_items and self._max_items > 0:
            data = f"{data}\n..."
        return str_indent(f"{typ} (length={length:,})\n{data}", num_spaces=self._num_spaces)
