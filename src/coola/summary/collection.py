r"""Implement summarizers for ``Collection`` data."""

from __future__ import annotations

__all__ = ["BaseCollectionSummarizer"]

from typing import TypeVar

from coola.summary.base import BaseSummarizer

T = TypeVar("T")


class BaseCollectionSummarizer(BaseSummarizer[T]):
    r"""Base class for summarizing collection-based data structures.

    This class provides the foundation for summarizing various collection types with
    configurable formatting options. It handles item limiting and indentation for
    readable output.

    Args:
        max_items: The maximum number of items to display in the summary.
            If set to a negative value (e.g., -1), all items in the collection
            will be shown without truncation. Defaults to 5.
        num_spaces: The number of spaces to use for indentation in the formatted
            output. This affects the visual structure of nested summaries.
            Defaults to 2.

    Attributes:
        _max_items: Stores the maximum number of items to display.
        _num_spaces: Stores the number of spaces for indentation.

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

    def __init__(self, max_items: int = 5, num_spaces: int = 2) -> None:
        self._max_items = max_items
        self._num_spaces = num_spaces

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(max_items={self._max_items:,}, "
            f"num_spaces={self._num_spaces})"
        )

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self._max_items == other._max_items and self._num_spaces == other._num_spaces
