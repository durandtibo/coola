r"""Implement handlers for ``polars.DataFrame``s, ``polars.LazyFrame``s,
and ``polars.Series``es."""

from __future__ import annotations

__all__ = ["PolarsDataFrameEqualHandler", "PolarsLazyFrameEqualHandler", "PolarsSeriesEqualHandler"]

from datetime import timedelta
from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import report_difference
from coola.utils.imports import is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class PolarsDataFrameEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``polars.DataFrame``s are equal.

    This handler returns ``True`` if the two ``polars.DataFrame``s are
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PolarsDataFrameEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PolarsDataFrameEqualHandler()
        >>> handler.handle(
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> handler.handle(
        ...     pl.DataFrame({"col": [1, 2, 3]}),
        ...     pl.DataFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def handle(
        self,
        actual: pl.DataFrame,
        expected: pl.DataFrame,
        config: EqualityConfig,
    ) -> bool:
        object_equal = frame_equal(actual, expected, config)
        if config.explaining and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="polars.DataFrames"),
            )
        return object_equal


class PolarsLazyFrameEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``polars.LazyFrame``s are equal.

    This handler returns ``True`` if the two ``polars.LazyFrame``s are
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PolarsLazyFrameEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PolarsLazyFrameEqualHandler()
        >>> handler.handle(
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> handler.handle(
        ...     pl.LazyFrame({"col": [1, 2, 3]}),
        ...     pl.LazyFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def handle(
        self,
        actual: pl.LazyFrame,
        expected: pl.LazyFrame,
        config: EqualityConfig,
    ) -> bool:
        # Compare the schemas first to fail fast without collecting the data.
        object_equal = actual.collect_schema() == expected.collect_schema() and frame_equal(
            actual.collect(), expected.collect(), config
        )
        if config.explaining and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="polars.LazyFrames"),
            )
        return object_equal


class PolarsSeriesEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``polars.Series``es are equal.

    This handler returns ``True`` if the two ``polars.Series``es are
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PolarsSeriesEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PolarsSeriesEqualHandler()
        >>> handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config)
        True
        >>> handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), config)
        False

        ```
    """

    def handle(
        self,
        actual: pl.Series,
        expected: pl.Series,
        config: EqualityConfig,
    ) -> bool:
        object_equal = series_equal(actual, expected, config)
        if config.explaining and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="polars.Series"),
            )
        return object_equal


def has_nan(df_or_series: pl.DataFrame | pl.Series) -> bool:
    r"""Indicate if a DataFrame or Series has NaN values.

    Only the float columns are checked: the values nested in a
    ``List``, ``Array`` or ``Struct`` column are left to polars (see
    ``series_equal``).

    Args:
        df_or_series: The DataFrame or series to check.

    Returns:
        ``True`` if the DataFrame or Series has a float NaN value,
            otherwise ``False``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.handler.polars import has_nan
        >>> has_nan(pl.Series([1.0, float("nan")]))
        True
        >>> has_nan(pl.Series([1.0, None]))
        False

        ```
    """
    if isinstance(df_or_series, pl.Series):
        return _series_has_nan(df_or_series)
    return any(_series_has_nan(df_or_series[name]) for name in df_or_series.columns)


def _series_has_nan(series: pl.Series) -> bool:
    # Only the float dtypes have NaN values, and ``is_nan`` is not defined for the
    # other numeric dtypes (``is_numeric`` also includes ``Decimal``, which raises).
    return series.dtype.is_float() and bool(series.is_nan().any())


def frame_equal(df1: pl.DataFrame, df2: pl.DataFrame, config: EqualityConfig) -> bool:
    r"""Indicate if the two DataFrames are equal or not.

    The structure is compared first (shape, names, order and dtypes of
    the columns), then the columns one by one, by the kind of their
    dtype (see ``series_equal``). The comparison stops at the first
    column that differs.

    Args:
        df1: The first DataFrame to compare.
        df2: The second DataFrame to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two DataFrames are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.polars import frame_equal
        >>> config = EqualityConfig()
        >>> frame_equal(pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 3]}), config)
        True
        >>> frame_equal(pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 4]}), config)
        False

        ```
    """
    if df1.shape != df2.shape or df1.schema != df2.schema:
        return False
    return all(_values_equal(df1[name], df2[name], config) for name in df1.columns)


def series_equal(series1: pl.Series, series2: pl.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the two series are equal or not.

    The shape, the name and the dtype are compared first, then the
    values, by the kind of the dtype:

    - numeric (integers and floats): the nulls must be at the same
      positions, then the values are compared with the shared numeric
      rule (tolerance, ``equal_nan``);
    - duration: same, on the durations (``atol`` is a number of
      seconds);
    - other (booleans, strings, categories, instants, decimals, nested
      values, ...): the ``equals`` of polars. It never sees the
      tolerance or ``equal_nan``, and a null is equal to a null.

    Args:
        series1: The first series to compare.
        series2: The second series to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two series are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.polars import series_equal
        >>> config = EqualityConfig()
        >>> series_equal(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config)
        True
        >>> series_equal(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), config)
        False

        ```
    """
    if (
        series1.shape != series2.shape
        or series1.name != series2.name
        or series1.dtype != series2.dtype
    ):
        return False
    return _values_equal(series1, series2, config)


def _values_equal(series1: pl.Series, series2: pl.Series, config: EqualityConfig) -> bool:
    # The two series have the same name and dtype.
    dtype = series1.dtype
    if dtype.is_float() and not config.equal_nan and _series_has_nan(series1):
        return False
    if dtype.is_integer() or dtype.is_float():
        return _numeric_series_close(series1, series2, config)
    if isinstance(dtype, pl.Duration):
        return _duration_series_close(series1, series2, config)
    return bool(series1.equals(series2, check_dtypes=True, check_names=True, null_equal=True))


def _numeric_series_close(series1: pl.Series, series2: pl.Series, config: EqualityConfig) -> bool:
    r"""Indicate if two numeric series are equal within a tolerance.

    The nulls must be at the same positions, then the values are
    compared with the NumPy rule (``|a - b| <= atol + rtol * max(|a|,
    |b|)``), like the other libraries.
    """
    from coola.equality.handler.numpy import array_equal  # noqa: PLC0415

    if not series1.is_null().equals(series2.is_null()):
        return False
    return array_equal(series1.fill_null(0).to_numpy(), series2.fill_null(0).to_numpy(), config)


def _duration_series_close(series1: pl.Series, series2: pl.Series, config: EqualityConfig) -> bool:
    r"""Indicate if two duration series are equal within a tolerance
    (``atol`` is a number of seconds).

    The nulls must be at the same positions.
    """
    from coola.equality.handler.numpy import array_equal  # noqa: PLC0415

    if not series1.is_null().equals(series2.is_null()):
        return False
    zero = timedelta(0)
    return array_equal(
        series1.fill_null(zero).to_numpy(), series2.fill_null(zero).to_numpy(), config
    )
