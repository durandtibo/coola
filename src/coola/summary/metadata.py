r"""Implement a base class for summarizers that describe the metadata of
an object (shape, dtype...) instead of its values."""

from __future__ import annotations

__all__ = ["MAX_FIELDS", "BaseFieldsSummarizer", "BaseMetadataSummarizer", "format_fields"]

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import InlineDisplayMixin
from coola.summary.base import BaseSummarizer
from coola.utils.introspection import is_same_type
from coola.validation import validate_non_negative

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

    def _display_fields(self) -> dict[str, Any]:
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


class BaseFieldsSummarizer(BaseMetadataSummarizer[T]):
    r"""Define the base class for the metadata summarizers that list
    the fields of an object (the columns of a data frame, the variables
    of a dataset).

    The listed fields are limited to ``max_fields``, to keep the
    summary short for an object with hundreds of columns. A child class
    uses ``self._format_fields`` to format its fields.

    Args:
        show_data: If ``True``, the summary is ``repr(data)``, which
            shows the values. If ``False`` (default), only the
            metadata is shown.
        max_fields: The maximum number of fields to list. The other
            fields are replaced by ``... (+N more)``. Defaults to
            ``MAX_FIELDS`` (10).

    Example:
        ```pycon
        >>> from coola.summary import SummarizerRegistry
        >>> from coola.summary.metadata import BaseFieldsSummarizer
        >>> class ColumnsSummarizer(BaseFieldsSummarizer[dict]):
        ...     def _describe(self, data):
        ...         return [f"columns={self._format_fields(data.items())}"]
        ...
        >>> data = {"a": "int", "b": "float", "c": "str"}
        >>> print(ColumnsSummarizer().summarize(data, SummarizerRegistry()))
        <class 'dict'> | columns=a: int, b: float, c: str
        >>> print(ColumnsSummarizer(max_fields=2).summarize(data, SummarizerRegistry()))
        <class 'dict'> | columns=a: int, b: float, ... (+1 more)

        ```
    """

    def __init__(self, show_data: bool = False, max_fields: int = MAX_FIELDS) -> None:
        super().__init__(show_data=show_data)
        validate_non_negative(max_fields, name="max_fields")
        self._max_fields = max_fields

    def _display_fields(self) -> dict[str, Any]:
        kwargs = super()._display_fields()
        # Shown only if it is not the default, so the usual summarizers have a short repr.
        if self._max_fields != MAX_FIELDS:
            kwargs["max_fields"] = self._max_fields
        return kwargs

    def equal(self, other: object) -> bool:
        if not is_same_type(other, self):
            return False
        return self._show_data == other._show_data and self._max_fields == other._max_fields

    def _format_fields(self, fields: Iterable[tuple[Any, Any]]) -> str:
        r"""Format the fields, limited to ``max_fields``.

        Args:
            fields: The ``(name, type)`` pairs to format.

        Returns:
            The formatted fields.
        """
        return format_fields(fields, max_fields=self._max_fields)
