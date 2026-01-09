r"""Implement summarizers for ``Collection`` data."""

from __future__ import annotations

__all__ = ["BaseCollectionSummarizer", "MappingSummarizer"]

from collections.abc import Mapping
from itertools import islice
from typing import TYPE_CHECKING, Any, TypeVar

from coola.summary.base import BaseSummarizer
from coola.utils import str_indent, str_mapping

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry

T = TypeVar("T")
TBaseCollectionSummarizer = TypeVar("TBaseCollectionSummarizer", bound="BaseCollectionSummarizer")


class BaseCollectionSummarizer(BaseSummarizer[T]):
    r"""Implement the default summarizer.

    Args:
        max_items: The maximum number of items to show.
            If a negative value is provided, all the items are shown.
        num_spaces: The number of spaces used for the indentation.

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

    def clone(self) -> TBaseCollectionSummarizer:
        return self.__class__(max_items=self._max_items, num_spaces=self._num_spaces)

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self._max_items == other._max_items and self._num_spaces == other._num_spaces


class MappingSummarizer(BaseCollectionSummarizer[Mapping[Any, Any]]):
    r"""Implement a formatter for ``Mapping``.

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
        if length > 0:
            items = data.items()
            if self._max_items >= 0:
                items = islice(data.items(), self._max_items)
            data = str_mapping(
                {
                    key: registry.summary(val, depth=depth + 1, max_depth=max_depth)
                    for key, val in items
                },
                num_spaces=self._num_spaces,
            )
            if length > self._max_items and self._max_items >= 0:
                data = f"{data}\n..."
            data = f"(length={length:,})\n{data}"
        return str_indent(f"{typ} {data}", num_spaces=self._num_spaces)
