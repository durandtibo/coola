r"""Implement handlers for ``polars.DataFrame``s, ``polars.LazyFrame``s,
and ``polars.Series``es."""

from __future__ import annotations

__all__ = ["PolarsDataFrameEqualHandler", "PolarsLazyFrameEqualHandler", "PolarsSeriesEqualHandler"]

import inspect
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import report_difference
from coola.utils.imports import is_numpy_available, is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
    import polars.selectors as cs
    from polars import testing
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
        if config.show_difference and not object_equal:
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
        if config.show_difference and not object_equal:
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
        if config.show_difference and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="polars.Series"),
            )
        return object_equal


def has_nan(df_or_series: pl.DataFrame | pl.Series) -> bool:
    r"""Indicate if a DataFrame or Series has NaN values.

    Args:
        df_or_series: The DataFrame or series to check.

    Returns:
        ``True`` if the DataFrame or Series has NaN values,
            otherwise ``False``. The floats nested in ``List``,
            ``Array`` and ``Struct`` columns are checked too.
    """
    if isinstance(df_or_series, pl.Series):
        return _series_has_nan(df_or_series)
    frame = df_or_series.select(cs.float())
    if not frame.is_empty() and frame.select(pl.any_horizontal(pl.all().is_nan().any())).item():
        return True
    # The nested columns (List, Array, Struct) can hold floats
    return any(
        _series_has_nan(df_or_series[name])
        for name in df_or_series.select(cs.by_dtype(pl.List, pl.Array, pl.Struct)).columns
    )


def _series_has_nan(series: pl.Series) -> bool:
    # Only the float dtypes have NaN values, and ``is_nan`` is not defined for the
    # other numeric dtypes (``is_numeric`` also includes ``Decimal``, which raises).
    dtype = series.dtype
    if dtype.is_float():
        return bool(series.is_nan().any())
    if isinstance(dtype, (pl.List, pl.Array)):
        return _series_has_nan(series.explode())
    if isinstance(dtype, pl.Struct):
        return any(_series_has_nan(series.struct.field(field.name)) for field in dtype.fields)
    return False


def frame_equal(df1: pl.DataFrame, df2: pl.DataFrame, config: EqualityConfig) -> bool:
    r"""Indicate if the two DataFrames are equal or not.

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
    if df1.shape != df2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and has_nan(df1):
        return False
    if df1.columns != df2.columns:
        return False
    if _tolerance_applies(config):
        # The integer and float columns follow the same rule as the other libraries.
        numeric = [c for c in df1.columns if _is_numeric_pair(df1[c], df2[c])]
        if not all(_numeric_series_close(df1[c], df2[c], config) for c in numeric):
            return False
        df1, df2 = df1.drop(numeric), df2.drop(numeric)
    try:
        assert_frame_equal(df1, df2, atol=config.atol, rtol=config.rtol)
    except AssertionError:
        return False
    return True


def series_equal(series1: pl.Series, series2: pl.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the two series are equal or not.

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
    if series1.shape != series2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and has_nan(series1):
        return False
    if _tolerance_applies(config) and _is_numeric_pair(series1, series2):
        return series1.name == series2.name and _numeric_series_close(series1, series2, config)
    try:
        assert_series_equal(
            series1,
            series2,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True


def _tolerance_applies(config: EqualityConfig) -> bool:
    # The tolerance of the numeric columns is applied with NumPy (the shared rule).
    return (config.atol > 0 or config.rtol > 0) and is_numpy_available()


def _is_numeric_pair(series1: pl.Series, series2: pl.Series) -> bool:
    # ``is_numeric`` also includes ``Decimal``, which is compared by polars.
    dtype = series1.dtype
    return dtype == series2.dtype and (dtype.is_integer() or dtype.is_float())


def _numeric_series_close(series1: pl.Series, series2: pl.Series, config: EqualityConfig) -> bool:
    r"""Indicate if two numeric series are equal within a tolerance.

    ``polars.testing`` applies the tolerance to the floats only and with
    the rule of ``math.isclose``. The nulls must be at the same
    positions, then the values are compared with the NumPy rule
    (``|a - b| <= atol + rtol * max(|a|, |b|)``).
    """
    from coola.equality.handler.numpy import array_equal  # noqa: PLC0415

    if not series1.is_null().equals(series2.is_null()):
        return False
    return array_equal(series1.fill_null(0).to_numpy(), series2.fill_null(0).to_numpy(), config)


def assert_frame_equal(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    r"""Compare two polars DataFrames with tolerance.

    This function supports multiple polars versions.

    Args:
        df1: The first DataFrame to compare.
        df2: The second DataFrame to compare.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.
    """
    check_exact = atol == 0 and rtol == 0
    if is_new_naming():
        testing.assert_frame_equal(df1, df2, abs_tol=atol, rel_tol=rtol, check_exact=check_exact)
    else:
        old_tol: dict[str, Any] = {"atol": atol, "rtol": rtol}
        testing.assert_frame_equal(df1, df2, check_exact=check_exact, **old_tol)


def assert_series_equal(
    series1: pl.Series,
    series2: pl.Series,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    r"""Compare two polars Series with tolerance.

    This function supports multiple polars versions.

    Args:
        series1: The first series to compare.
        series2: The second series to compare.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.
    """
    check_exact = atol == 0 and rtol == 0
    if is_new_naming():
        testing.assert_series_equal(
            series1, series2, abs_tol=atol, rel_tol=rtol, check_exact=check_exact
        )
    else:
        old_tol: dict[str, Any] = {"atol": atol, "rtol": rtol}
        testing.assert_series_equal(series1, series2, check_exact=check_exact, **old_tol)


@lru_cache
def is_new_naming() -> bool:
    r"""Indicate if polars uses the new naming for ``abs_tol`` and
    ``rel_tol``.

    Returns:
        ``True`` if ``abs_tol`` and ``rel_tol`` are used,
            otherwise ``atol`` and ``rtol`` are used.
    """
    return "abs_tol" in inspect.signature(testing.assert_frame_equal).parameters
