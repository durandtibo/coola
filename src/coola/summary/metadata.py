r"""Implement a base class for summarizers that describe the metadata of
an object (shape, dtype...) instead of its values."""

from __future__ import annotations

__all__ = ["BaseMetadataSummarizer", "format_fields"]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import InlineDisplayMixin
from coola.summary.base import BaseSummarizer
from coola.utils.introspection import is_same_type

if TYPE_CHECKING:
    from collections.abc import Iterable

    from coola.summary.registry import SummarizerRegistry

T = TypeVar("T")

MAX_FIELDS = 10
r"""The maximum number of columns/variables listed in a summary."""


def format_fields(fields: Iterable[tuple[Any, Any]], max_fields: int = MAX_FIELDS) -> str:
    r"""Format the fields of a table-like object (e.g. the columns of a
    data frame) as ``name: type`` pairs.

    Args:
        fields: The ``(name, type)`` pairs.
        max_fields: The maximum number of pairs to show. The other
            ones are replaced by ``... (+N more)``.

    Returns:
        The formatted fields.

    Example:
        ```pycon
        >>> from coola.summary.metadata import format_fields
        >>> format_fields([("a", "int64"), ("b", "float64")])
        'a: int64, b: float64'
        >>> format_fields([("a", "int64"), ("b", "float64"), ("c", "str")], max_fields=2)
        'a: int64, b: float64, ... (+1 more)'

        ```
    """
    items = [f"{name}: {dtype}" for name, dtype in fields]
    if len(items) <= max_fields:
        return ", ".join(items)
    hidden = len(items) - max_fields
    return ", ".join([*items[:max_fields], f"... (+{hidden} more)"])


class BaseMetadataSummarizer(InlineDisplayMixin, BaseSummarizer[T]):
    r"""Define the base class for the summarizers that describe the
    metadata of an object, instead of its values.

    The summary is ``<class> | <field 1> | <field 2> | ...``, where the
    fields are returned by ``_describe``. A child class needs to
    implement ``_describe``, and can override ``_check_dependencies``
    to raise an error if a required optional package is not installed.

    Args:
        show_data: If ``True``, the summary is ``repr(data)``, which
            shows the values. If ``False`` (default), only the
            metadata is shown.

    Example:
        ```pycon
        >>> from coola.summary import SummarizerRegistry
        >>> from coola.summary.metadata import BaseMetadataSummarizer
        >>> class ListSummarizer(BaseMetadataSummarizer[list]):
        ...     def _describe(self, data):
        ...         return [f"length={len(data)}"]
        ...
        >>> summarizer = ListSummarizer()
        >>> print(summarizer.summarize([1, 2, 3], SummarizerRegistry()))
        <class 'list'> | length=3
        >>> summarizer = ListSummarizer(show_data=True)
        >>> print(summarizer.summarize([1, 2, 3], SummarizerRegistry()))
        [1, 2, 3]

        ```
    """

    def __init__(self, show_data: bool = False) -> None:
        self._check_dependencies()
        self._show_data = bool(show_data)

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"show_data": self._show_data}

    def _check_dependencies(self) -> None:
        r"""Check that the packages required by the summarizer are
        installed.

        By default, the summarizer has no requirement.

        Raises:
            MissingPackageError: if a required package is not installed.
        """

    @abstractmethod
    def _describe(self, data: T) -> list[str]:
        r"""Describe the metadata of an object.

        Args:
            data: The object to describe.

        Returns:
            The fields of the summary, e.g. ``["shape=(2, 3)", "dtype=int64"]``.
        """

    def equal(self, other: object) -> bool:
        if not is_same_type(other, self):
            return False
        return self._show_data == other._show_data

    def summarize(
        self,
        data: T,
        registry: SummarizerRegistry,  # noqa: ARG002
        depth: int = 0,  # noqa: ARG002
        max_depth: int = 1,  # noqa: ARG002
    ) -> str:
        if self._show_data:
            return repr(data)
        return " | ".join([f"{type(data)}", *self._describe(data)])
