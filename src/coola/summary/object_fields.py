r"""Implement the base class of the summarizers of objects with
fields."""

from __future__ import annotations

__all__ = ["BaseObjectFieldsSummarizer"]

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

from coola.summary.base import BaseSummarizer, Node

if TYPE_CHECKING:
    from coola.summary.config import SummaryConfig

T = TypeVar("T")


class BaseObjectFieldsSummarizer(BaseSummarizer[T]):
    r"""Define the base class of the summarizers of objects that are
    described by their fields (dataclasses, models, ...).

    A subclass implements ``_get_names``, which returns the names of the
    fields to show. The values are read lazily with ``getattr``, so only
    the fields that are shown are read.
    """

    @abstractmethod
    def _get_names(self, data: T) -> list[str]:
        r"""Return the names of the fields to show.

        Args:
            data: The object.

        Returns:
            The names of the fields, in display order.
        """

    def summarize(self, data: T, config: SummaryConfig, *, compact: bool) -> Node:  # noqa: ARG002
        r"""Summarize an object by its fields.

        Args:
            data: The data to summarize.
            config: The options of the summary.
            compact: If ``True``, the summary is a single line, as the
                node is not expanded.

        Returns:
            The node whose children are the fields.
        """
        names = self._get_names(data)
        children = ((name, getattr(data, name)) for name in names)
        return Node(f"{type(data)} (fields={len(names):,})", children, len(names))
