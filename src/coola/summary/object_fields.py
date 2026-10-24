r"""Implement the base class of the summarizers of the objects that are
described by named fields (dataclasses, pydantic models)."""

from __future__ import annotations

__all__ = ["BaseObjectFieldsSummarizer"]

from abc import abstractmethod
from itertools import islice
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import InlineDisplayMixin
from coola.summary.base import BaseSummarizer
from coola.utils.format import str_indent, str_mapping
from coola.utils.introspection import is_same_type
from coola.utils.safe import safe_str

if TYPE_CHECKING:
    from coola.summary.registry import SummarizerRegistry

T = TypeVar("T")


class BaseObjectFieldsSummarizer(InlineDisplayMixin, BaseSummarizer[T]):
    r"""Define the base class of the summarizers of the objects that have
    named fields.

    The fields are summarized with the registry, so a field can be an
    array, a data frame, another dataclass, and so on, and ``max_depth``
    applies as it does for the mappings. When the depth limit is
    reached, the object is shown as a single string.

    Subclasses implement ``_get_fields``.

    Args:
        max_items: The maximum number of fields to display. If
            negative, all the fields are shown. Defaults to 5.
        num_spaces: The number of spaces for indenting each level.
            Defaults to 2.
    """

    def __init__(self, max_items: int = 5, num_spaces: int = 2) -> None:
        self._max_items = max_items
        self._num_spaces = num_spaces

    def _display_fields(self) -> dict[str, Any]:
        return {"max_items": self._max_items, "num_spaces": self._num_spaces}

    def equal(self, other: object) -> bool:
        if not is_same_type(other, self):
            return False
        return self._max_items == other._max_items and self._num_spaces == other._num_spaces

    @abstractmethod
    def _get_fields(self, data: T) -> dict[str, Any]:
        r"""Return the fields to summarize, by name."""

    def summarize(
        self,
        data: T,
        registry: SummarizerRegistry,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        if depth >= max_depth:
            return registry.summarize(safe_str(data), depth=depth + 1, max_depth=max_depth)
        typ = type(data)
        fields = self._get_fields(data)
        length = len(fields)
        if length == 0:
            return str_indent(f"{typ} (fields=0)", num_spaces=self._num_spaces)
        if self._max_items == 0:
            return str_indent(f"{typ} (fields={length:,}) ...", num_spaces=self._num_spaces)
        items = fields.items()
        if self._max_items > 0:
            items = islice(items, self._max_items)
        body = str_mapping(
            {
                name: registry.summarize(value, depth=depth + 1, max_depth=max_depth)
                for name, value in items
            },
            num_spaces=self._num_spaces,
        )
        if length > self._max_items and self._max_items > 0:
            body = f"{body}\n..."
        return str_indent(f"{typ} (fields={length:,})\n{body}", num_spaces=self._num_spaces)
