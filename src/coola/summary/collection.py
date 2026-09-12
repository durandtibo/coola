r"""Implement summarizers for ``Collection`` data."""

from __future__ import annotations

__all__ = ["BaseCollectionSummarizer"]

from collections.abc import Sized
from typing import TYPE_CHECKING, Any, TypeVar

from coola.summary.base import BaseSummarizer
from coola.utils.format import str_indent

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry

T = TypeVar("T", bound=Sized)


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

    Example:
        ```pycon
        >>> from coola.summary import SummarizerRegistry, MappingSummarizer, DefaultSummarizer
        >>> registry = SummarizerRegistry({object: DefaultSummarizer()})
        >>> summarizer = MappingSummarizer()
        >>> output = summarizer.summarize({"key1": 1.2, "key2": "abc", "key3": 42}, registry)
        >>> print(output)
        <class 'dict'> (length=3)
          (key1): 1.2
          (key2): abc
          (key3): 42

        ```
    """

    #: Number of items to preview when the depth limit is hit and
    #: ``max_items`` is negative (i.e. truncation is otherwise disabled).
    _DEPTH_LIMIT_PREVIEW_ITEMS = 5

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

    def summarize(
        self,
        data: T,
        registry: SummarizerRegistry,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        r"""Summarize a collection following the shared truncation and
        depth-limiting skeleton.

        Subclasses only need to implement ``_get_preview`` and
        ``_format_items`` to control how items are previewed and
        formatted.
        """
        if depth >= max_depth:
            text = str(data)
            preview_limit = (
                self._max_items if self._max_items >= 0 else self._DEPTH_LIMIT_PREVIEW_ITEMS
            )
            if len(data) > preview_limit:
                preview = self._get_preview(data, preview_limit)
                text = f"{preview!r} ..."
            return registry.summarize(text, depth=depth + 1, max_depth=max_depth)
        typ = type(data)
        length = len(data)
        if length == 0:
            return str_indent(f"{typ} {data}", num_spaces=self._num_spaces)
        if self._max_items == 0:
            return str_indent(f"{typ} (length={length:,}) ...", num_spaces=self._num_spaces)

        body = self._format_items(data, registry, depth=depth, max_depth=max_depth)
        if length > self._max_items and self._max_items > 0:
            body = f"{body}\n..."
        return str_indent(f"{typ} (length={length:,})\n{body}", num_spaces=self._num_spaces)

    def _get_preview(self, data: T, limit: int) -> Any:
        r"""Return a preview of ``data`` truncated to ``limit`` items,
        used when the depth limit has been reached and the collection
        exceeds that limit."""
        raise NotImplementedError

    def _format_items(
        self,
        data: T,
        registry: SummarizerRegistry,
        depth: int,
        max_depth: int,
    ) -> str:
        r"""Format the (possibly truncated) items of ``data`` into the
        body of the summary."""
        raise NotImplementedError
