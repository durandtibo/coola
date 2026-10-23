r"""Implement summarizers for polars objects.

The summarizers show metadata (shape, dtype, column names) instead of
the values by default.
"""

from __future__ import annotations

__all__ = ["PolarsDataFrameSummarizer", "PolarsLazyFrameSummarizer", "PolarsSeriesSummarizer"]

from typing import TYPE_CHECKING

from coola.summary.metadata import BaseFieldsSummarizer, BaseMetadataSummarizer
from coola.utils.imports import check_polars, is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl


class PolarsDataFrameSummarizer(BaseFieldsSummarizer[pl.DataFrame]):
    r"""Implement a summarizer for ``polars.DataFrame`` objects.

    By default, it displays the metadata (type, shape, columns with their
    dtype) rather than the values. At most 10 columns are listed.

    Args:
        show_data: If ``True``, returns ``repr(dataframe)``, displaying
            the values. If ``False`` (default), returns only the
            metadata: ``<class> | shape=<shape> | columns=<name: dtype, ...>``.

    Raises:
        MissingPackageError: If polars is not installed.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.summary import SummarizerRegistry, PolarsDataFrameSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = PolarsDataFrameSummarizer()
        >>> frame = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        >>> print(summarizer.summarize(frame, registry))  # doctest: +ELLIPSIS
        <class '...'> | shape=(3, 2) | columns=a: Int64, b: Float64

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _describe(self, data: pl.DataFrame) -> list[str]:
        return [f"shape={data.shape}", f"columns={self._format_fields(data.schema.items())}"]


class PolarsLazyFrameSummarizer(BaseFieldsSummarizer[pl.LazyFrame]):
    r"""Implement a summarizer for ``polars.LazyFrame`` objects.

    The lazy frame is **not** collected: only its schema is read, so the
    number of rows is not shown. At most 10 columns are listed.

    Args:
        show_data: If ``True``, returns ``repr(lazyframe)``, which shows
            the query plan. If ``False`` (default), returns only the
            columns: ``<class> | columns=<name: dtype, ...>``.

    ``collect_schema()`` resolves the whole query plan, which can read
    file metadata (``scan_parquet``, ``scan_csv``, possibly remote). If
    the plan is invalid, the columns are replaced by
    ``<unresolved schema: ExceptionType>`` instead of raising.

    Raises:
        MissingPackageError: If polars is not installed.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.summary import SummarizerRegistry, PolarsLazyFrameSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = PolarsLazyFrameSummarizer()
        >>> frame = pl.LazyFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        >>> print(summarizer.summarize(frame, registry))  # doctest: +ELLIPSIS
        <class '...'> | columns=a: Int64, b: Float64

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _describe(self, data: pl.LazyFrame) -> list[str]:
        try:
            schema = data.collect_schema()
        except Exception as exc:
            return [f"columns=<unresolved schema: {type(exc).__name__}>"]
        return [f"columns={self._format_fields(schema.items())}"]


class PolarsSeriesSummarizer(BaseMetadataSummarizer[pl.Series]):
    r"""Implement a summarizer for ``polars.Series`` objects.

    By default, it displays the metadata (type, shape, dtype, name)
    rather than the values.

    Args:
        show_data: If ``True``, returns ``repr(series)``, displaying
            the values. If ``False`` (default), returns only the
            metadata: ``<class> | shape=<shape> | dtype=<dtype> | name=<name>``.

    Raises:
        MissingPackageError: If polars is not installed.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.summary import SummarizerRegistry, PolarsSeriesSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = PolarsSeriesSummarizer()
        >>> print(summarizer.summarize(pl.Series("a", [1, 2, 3]), registry))  # doctest: +ELLIPSIS
        <class '...'> | shape=(3,) | dtype=Int64 | name='a'

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _describe(self, data: pl.Series) -> list[str]:
        return [f"shape={data.shape}", f"dtype={data.dtype}", f"name={data.name!r}"]
