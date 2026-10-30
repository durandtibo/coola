r"""Implement summarizers for pyarrow objects.

The summarizers show metadata (shape, type, column names) instead of the
values by default.
"""

from __future__ import annotations

__all__ = ["PyarrowArraySummarizer", "PyarrowTableSummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.metadata import BaseMetadataSummarizer, Field
from coola.utils.imports import check_pyarrow, is_pyarrow_available

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa


if TYPE_CHECKING:
    from coola.summary.base import Summarizer


class PyarrowArraySummarizer(BaseMetadataSummarizer["pa.Array | pa.ChunkedArray"]):
    r"""Implement a summarizer for ``pyarrow.Array`` and
    ``pyarrow.ChunkedArray`` objects.

    By default, it displays the metadata (type, length, arrow type)
    rather than the values.

    Raises:
        MissingPackageError: If pyarrow is not installed.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.summary import PyarrowArraySummarizer, SummaryConfig
        >>> summarizer = PyarrowArraySummarizer()
        >>> print(summarizer.summarize(pa.array([1, 2, 3]), SummaryConfig(), compact=False))  # doctest: +ELLIPSIS
        <class '...'> | length=3 | type=int64

        ```
    """

    def _check_dependencies(self) -> None:
        check_pyarrow()

    def _describe(self, data: pa.Array | pa.ChunkedArray) -> list[Field]:
        return [f"length={len(data)}", f"type={data.type}"]


class PyarrowTableSummarizer(BaseMetadataSummarizer["pa.Table | pa.RecordBatch"]):
    r"""Implement a summarizer for ``pyarrow.Table`` and
    ``pyarrow.RecordBatch`` objects.

    By default, it displays the metadata (type, shape, columns with their
    arrow type) rather than the values. At most `max_items` columns are listed.

    Raises:
        MissingPackageError: If pyarrow is not installed.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.summary import PyarrowTableSummarizer, SummaryConfig
        >>> summarizer = PyarrowTableSummarizer()
        >>> table = pa.table({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        >>> print(summarizer.summarize(table, SummaryConfig(), compact=False))  # doctest: +ELLIPSIS
        <class '...'> | shape=(3, 2) | columns=a: int64, b: double

        ```
    """

    def _check_dependencies(self) -> None:
        check_pyarrow()

    def _describe(self, data: pa.Table | pa.RecordBatch) -> list[Field]:
        fields = [(field.name, field.type) for field in data.schema]
        return [f"shape={data.shape}", ("columns", fields)]


def get_default_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the default summarizers of the pyarrow objects.

    Returns:
        The summarizers of the pyarrow objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the summarizers of the pyarrow objects.
    """
    if not is_pyarrow_available():
        return {}

    array_summarizer = PyarrowArraySummarizer()
    table_summarizer = PyarrowTableSummarizer()
    return {
        pa.Array: array_summarizer,
        pa.ChunkedArray: array_summarizer,
        pa.Table: table_summarizer,
        pa.RecordBatch: table_summarizer,
    }
