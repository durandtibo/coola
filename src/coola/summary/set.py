r"""Implement summarizers for set data."""

from __future__ import annotations

__all__ = ["SetSummarizer"]

import heapq
from collections.abc import Set as AbstractSet
from itertools import islice
from typing import TYPE_CHECKING, Any

from coola.summary.collection import BaseCollectionSummarizer
from coola.utils.format import str_sequence

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry


def _first_items(data: AbstractSet[Any], limit: int) -> list[Any]:
    r"""Return the first ``limit`` items of a set (all the items if
    ``limit`` is not positive).

    The items are the smallest ones when they can be compared, so the
    output does not depend on the iteration order of the set (which
    depends on ``PYTHONHASHSEED`` for the strings). Otherwise, the items
    are returned in iteration order.
    """
    try:
        return heapq.nsmallest(limit, data) if limit > 0 else sorted(data)
    except TypeError:
        return list(islice(data, limit)) if limit > 0 else list(data)


class SetSummarizer(BaseCollectionSummarizer[AbstractSet[Any]]):
    r"""Summarizer for set-based data structures.

    This class formats set types (set, frozenset, etc.) into readable
    summaries that display the type, length, and items with proper
    indentation. It respects the max_items limit and handles nested structures
    through the registry system.

    When the items can be compared, the items that are displayed are
    the smallest ones, in order, so the summary is the same from one
    run to the next (the iteration order of a set of strings depends on
    ``PYTHONHASHSEED``). Otherwise, the items are displayed in
    iteration order.

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

    def _get_preview(self, data: AbstractSet[Any], limit: int) -> Any:
        items = _first_items(data, limit)
        # The type is kept for the builtin sets, any other set is shown as a list.
        if type(data) is set:
            return set(items)
        if type(data) is frozenset:
            return frozenset(items)
        return list(items)

    def _format_items(
        self,
        data: AbstractSet[Any],
        registry: SummarizerRegistry,
        depth: int,
        max_depth: int,
    ) -> str:
        items = _first_items(data, self._max_items)
        return str_sequence(
            [registry.summarize(value, depth=depth + 1, max_depth=max_depth) for value in items],
            num_spaces=self._num_spaces,
        )
