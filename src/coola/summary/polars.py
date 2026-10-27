r"""Implement summarizers for polars objects.

The summarizers show metadata (shape, dtype, column names) instead of
the values by default.
"""

from __future__ import annotations

__all__ = ["PolarsDataFrameSummarizer", "PolarsLazyFrameSummarizer", "PolarsSeriesSummarizer"]

from typing import TYPE_CHECKING, Any

from coola.summary.metadata import BaseMetadataSummarizer, Field
from coola.utils.imports import check_polars, is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl


if TYPE_CHECKING:
    from coola.summary.base import Summarizer


class PolarsDataFrameSummarizer(BaseMetadataSummarizer[pl.DataFrame]):
    r"""Implement a summarizer for ``polars.DataFrame`` objects.

    By default, it displays the metadata (type, shape, columns with their
    dtype) rather than the values. At most `max_items` columns are listed.

    Raises:
        MissingPackageError: If polars is not installed.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.summary import PolarsDataFrameSummarizer, SummaryConfig
        >>> summarizer = PolarsDataFrameSummarizer()
        >>> frame = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        >>> print(summarizer.summarize(frame, SummaryConfig(), compact=False))  # doctest: +ELLIPSIS
        <class '...'> | shape=(3, 2) | columns=a: Int64, b: Float64

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _describe(self, data: pl.DataFrame) -> list[Field]:
        return [
            f"shape={data.shape}",
            ("columns", data.schema.items()),
        ]


class PolarsLazyFrameSummarizer(BaseMetadataSummarizer[pl.LazyFrame]):
    r"""Implement a summarizer for ``polars.LazyFrame`` objects.

    The lazy frame is **not** collected: only its schema is read, so the
    number of rows is not shown. At most `max_items` columns are listed.

    Raises:
        MissingPackageError: If polars is not installed.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.summary import PolarsLazyFrameSummarizer, SummaryConfig
        >>> summarizer = PolarsLazyFrameSummarizer()
        >>> frame = pl.LazyFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        >>> print(summarizer.summarize(frame, SummaryConfig(), compact=False))  # doctest: +ELLIPSIS
        <class '...'> | columns=a: Int64, b: Float64

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _describe(self, data: pl.LazyFrame) -> list[Field]:
        try:
            schema = data.collect_schema()
        except pl.exceptions.PolarsError as exc:
            return [f"columns=<unresolved schema: {type(exc).__name__}>"]
        return [("columns", schema.items())]


class PolarsSeriesSummarizer(BaseMetadataSummarizer[pl.Series]):
    r"""Implement a summarizer for ``polars.Series`` objects.

    By default, it displays the metadata (type, shape, dtype, name)
    rather than the values.

    Raises:
        MissingPackageError: If polars is not installed.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.summary import PolarsSeriesSummarizer, SummaryConfig
        >>> summarizer = PolarsSeriesSummarizer()
        >>> print(summarizer.summarize(pl.Series("a", [1, 2, 3]), SummaryConfig(), compact=False))  # doctest: +ELLIPSIS
        <class '...'> | shape=(3,) | dtype=Int64 | name='a'

        ```
    """

    def _check_dependencies(self) -> None:
        check_polars()

    def _describe(self, data: pl.Series) -> list[Field]:
        return [f"shape={data.shape}", f"dtype={data.dtype}", f"name={data.name!r}"]


def get_default_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the default summarizers of the polars objects.

    Returns:
        The summarizers of the polars objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the summarizers of the polars objects.
    """
    if not is_polars_available():
        return {}

    return {
        pl.DataFrame: PolarsDataFrameSummarizer(),
        pl.LazyFrame: PolarsLazyFrameSummarizer(),
        pl.Series: PolarsSeriesSummarizer(),
    }
