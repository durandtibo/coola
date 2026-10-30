r"""Implement the adapters of the polars objects.

This module provides the adapters that convert a ``polars.DataFrame``,
``polars.LazyFrame`` or ``polars.Series`` to a table, which is compared
by the ``TableEqualityTester``.
"""

from __future__ import annotations

__all__ = [
    "PolarsFrameTable",
    "PolarsLazyFrameTable",
    "PolarsSeriesTable",
    "polars_frame",
    "polars_lazy_frame",
    "polars_series",
]

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from coola.equality.tester.table import Column, TableEqualityTester
from coola.utils.imports import is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine
    from coola.equality.tester.base import BaseEqualityTester


class PolarsFrameTable:
    r"""Adapt a ``polars.DataFrame`` to a table.

    The structure is the shape and the schema (names, order and data
    types of the columns). The columns are compared one by one, by the
    kind of their data type.

    Args:
        frame: The data frame.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.tester.polars import PolarsFrameTable
        >>> table = PolarsFrameTable(pl.DataFrame({"a": [1, 2]}))
        >>> table.kind_name
        'tables'

        ```
    """

    kind_name = "tables"

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def same_structure(self, other: PolarsFrameTable, ctx: Engine) -> bool:  # noqa: ARG002
        r"""Indicate if two data frames have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        return self._frame.shape == other._frame.shape and self._frame.schema == other._frame.schema

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of each data frame column.
        """
        for name in self._frame.columns:
            yield _column(name, self._frame[name])


class PolarsLazyFrameTable:
    r"""Adapt a ``polars.LazyFrame`` to a table.

    The schemas are compared first, without collecting the data. The
    frames are then collected and compared as data frames. The query
    plan is ignored.

    Args:
        frame: The lazy frame.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.tester.polars import PolarsLazyFrameTable
        >>> table = PolarsLazyFrameTable(pl.LazyFrame({"a": [1, 2]}))
        >>> table.kind_name
        'tables'

        ```
    """

    kind_name = "tables"

    def __init__(self, frame: pl.LazyFrame) -> None:
        self._frame = frame
        self._collected: pl.DataFrame | None = None

    def same_structure(self, other: PolarsLazyFrameTable, ctx: Engine) -> bool:
        r"""Indicate if two lazy frames have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        if self._frame.collect_schema() != other._frame.collect_schema():
            return False
        return PolarsFrameTable(self.collect()).same_structure(
            PolarsFrameTable(other.collect()), ctx
        )

    def collect(self) -> pl.DataFrame:
        r"""Collect the lazy frame (once).

        Returns:
            The collected data frame.
        """
        if self._collected is None:
            self._collected = self._frame.collect()
        return self._collected

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of each column of the collected frame.
        """
        return PolarsFrameTable(self.collect()).columns()


class PolarsSeriesTable:
    r"""Adapt a ``polars.Series`` to a table with one column.

    The structure is the shape, the name and the data type.

    Args:
        series: The series.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.tester.polars import PolarsSeriesTable
        >>> table = PolarsSeriesTable(pl.Series([1, 2]))
        >>> table.kind_name
        'series'

        ```
    """

    kind_name = "series"

    def __init__(self, series: pl.Series) -> None:
        self._series = series

    def same_structure(self, other: PolarsSeriesTable, ctx: Engine) -> bool:  # noqa: ARG002
        r"""Indicate if two series have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        s1, s2 = self._series, other._series
        return s1.shape == s2.shape and s1.name == s2.name and s1.dtype == s2.dtype

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of the series.
        """
        yield _column(None, self._series)


def polars_frame(frame: pl.DataFrame) -> PolarsFrameTable:
    r"""Adapt a ``polars.DataFrame`` to a table.

    Args:
        frame: The data frame to adapt.

    Returns:
        The table.
    """
    return PolarsFrameTable(frame)


def polars_lazy_frame(frame: pl.LazyFrame) -> PolarsLazyFrameTable:
    r"""Adapt a ``polars.LazyFrame`` to a table.

    Args:
        frame: The lazy frame to adapt.

    Returns:
        The table.
    """
    return PolarsLazyFrameTable(frame)


def polars_series(series: pl.Series) -> PolarsSeriesTable:
    r"""Adapt a ``polars.Series`` to a table.

    Args:
        series: The series to adapt.

    Returns:
        The table.
    """
    return PolarsSeriesTable(series)


def _column(name: object, series: pl.Series) -> Column:
    r"""Convert a series to a column, by the kind of its data type.

    - numeric (integers and floats): the nulls are at the same
      positions, then the values are compared with the shared numeric
      rule (tolerance, ``equal_nan``);
    - duration: same, on the durations;
    - other (booleans, strings, categories, instants, decimals, nested
      values, ...): the ``equals`` of polars. It never sees the
      tolerance or ``equal_nan``, and a null is equal to a null.

    Args:
        name: The name of the column.
        series: The series of the column.

    Returns:
        The column.
    """
    dtype = series.dtype
    if dtype.is_integer() or dtype.is_float():
        return Column(name, "numeric", series.fill_null(0).to_numpy(), series.is_null().to_numpy())
    if isinstance(dtype, pl.Duration):
        return Column(
            name,
            "duration",
            series.fill_null(timedelta(0)).to_numpy(),
            series.is_null().to_numpy(),
        )
    return Column(name, "exact", series, equals=_series_equals)


def _series_equals(series1: pl.Series, series2: pl.Series) -> bool:
    r"""Indicate if two series are equal.

    Args:
        series1: The first series.
        series2: The second series.

    Returns:
        ``True`` if the dtypes, the names and the values are equal,
            otherwise ``False``. A null is equal to a null.
    """
    return bool(series1.equals(series2, check_dtypes=True, check_names=True, null_equal=True))


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the polars objects.

    Returns:
        The equality testers of the polars objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the polars objects.
    """
    if not is_polars_available():
        return {}

    return {
        pl.DataFrame: TableEqualityTester(polars_frame),
        pl.LazyFrame: TableEqualityTester(polars_lazy_frame),
        pl.Series: TableEqualityTester(polars_series),
    }
