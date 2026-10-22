r"""Implement summarizers for pyarrow objects.

The summarizers show metadata (shape, type, column names) instead of the
values by default.
"""

from __future__ import annotations

__all__ = ["PyarrowArraySummarizer", "PyarrowTableSummarizer"]

from typing import TYPE_CHECKING

from coola.summary.metadata import BaseFieldsSummarizer, BaseMetadataSummarizer
from coola.utils.imports import check_pyarrow, is_pyarrow_available

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa


class PyarrowArraySummarizer(BaseMetadataSummarizer["pa.Array | pa.ChunkedArray"]):
    r"""Implement a summarizer for ``pyarrow.Array`` and
    ``pyarrow.ChunkedArray`` objects.

    By default, it displays the metadata (type, length, arrow type)
    rather than the values.

    Args:
        show_data: If ``True``, returns ``repr(array)``, displaying
            the values. If ``False`` (default), returns only the
            metadata: ``<class> | length=<length> | type=<arrow type>``.

    Raises:
        MissingPackageError: If pyarrow is not installed.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.summary import SummarizerRegistry, PyarrowArraySummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = PyarrowArraySummarizer()
        >>> print(summarizer.summarize(pa.array([1, 2, 3]), registry))  # doctest: +ELLIPSIS
        <class '...'> | length=3 | type=int64

        ```
    """

    def _check_dependencies(self) -> None:
        check_pyarrow()

    def _describe(self, data: pa.Array | pa.ChunkedArray) -> list[str]:
        return [f"length={len(data)}", f"type={data.type}"]


class PyarrowTableSummarizer(BaseFieldsSummarizer[pa.Table]):
    r"""Implement a summarizer for ``pyarrow.Table`` objects.

    By default, it displays the metadata (type, shape, columns with their
    arrow type) rather than the values. At most 10 columns are listed.

    Args:
        show_data: If ``True``, returns ``repr(table)``, displaying the
            schema and the values. If ``False`` (default), returns only
            the metadata: ``<class> | shape=<shape> | columns=<name: type, ...>``.

    Raises:
        MissingPackageError: If pyarrow is not installed.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.summary import SummarizerRegistry, PyarrowTableSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = PyarrowTableSummarizer()
        >>> table = pa.table({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        >>> print(summarizer.summarize(table, registry))  # doctest: +ELLIPSIS
        <class '...'> | shape=(3, 2) | columns=a: int64, b: double

        ```
    """

    def _check_dependencies(self) -> None:
        check_pyarrow()

    def _describe(self, data: pa.Table) -> list[str]:
        fields = [(field.name, field.type) for field in data.schema]
        return [f"shape={data.shape}", f"columns={self._format_fields(fields)}"]
