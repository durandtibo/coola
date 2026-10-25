r"""Implement handlers for ``pandas.DataFrame``s and
``pandas.Series``s."""

from __future__ import annotations

__all__ = [
    "PandasDataFrameEqualHandler",
    "PandasExtensionArrayEqualHandler",
    "PandasIndexEqualHandler",
    "PandasSeriesEqualHandler",
]

from typing import TYPE_CHECKING, Any, cast

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import report_difference
from coola.utils.imports import is_numpy_available, is_pandas_available

if TYPE_CHECKING or is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class PandasDataFrameEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``pandas.DataFrame``s are equal.

    This handler returns ``True`` if the two ``pandas.DataFrame``s are
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import pandas
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PandasDataFrameEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PandasDataFrameEqualHandler()
        >>> handler.handle(
        ...     pandas.DataFrame({"col": [1, 2, 3]}),
        ...     pandas.DataFrame({"col": [1, 2, 3]}),
        ...     config,
        ... )
        True
        >>> handler.handle(
        ...     pandas.DataFrame({"col": [1, 2, 3]}),
        ...     pandas.DataFrame({"col": [1, 2, 4]}),
        ...     config,
        ... )
        False

        ```
    """

    def handle(
        self,
        actual: pd.DataFrame,
        expected: pd.DataFrame,
        config: EqualityConfig,
    ) -> bool:
        object_equal = frame_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas.DataFrames"),
            )
        return object_equal


class PandasSeriesEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``pandas.Series``es are equal.

    This handler returns ``True`` if the two ``pandas.Series``es are
    equal, otherwise ``False``. This handler is designed to be used
    at the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import pandas
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PandasSeriesEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PandasSeriesEqualHandler()
        >>> handler.handle(pandas.Series([1, 2, 3]), pandas.Series([1, 2, 3]), config)
        True
        >>> handler.handle(pandas.Series([1, 2, 3]), pandas.Series([1, 2, 4]), config)
        False

        ```
    """

    def handle(
        self,
        actual: pd.Series,
        expected: pd.Series,
        config: EqualityConfig,
    ) -> bool:
        object_equal = series_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas.Series"),
            )
        return object_equal


class PandasIndexEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two ``pandas.Index``es are equal.

    This handler returns ``True`` if the two ``pandas.Index``es are
    equal (values, dtype and names), otherwise ``False``. This handler
    is designed to be used at the end of the chain of responsibility.
    This handler does not call the next handler.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PandasIndexEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PandasIndexEqualHandler()
        >>> handler.handle(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]), config)
        True
        >>> handler.handle(pd.Index([1, 2, 3]), pd.Index([1, 2, 4]), config)
        False

        ```
    """

    def handle(
        self,
        actual: pd.Index,
        expected: pd.Index,
        config: EqualityConfig,
    ) -> bool:
        object_equal = index_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas.Index"),
            )
        return object_equal


class PandasExtensionArrayEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two pandas extension arrays are equal.

    The extension arrays include ``pandas.Categorical`` and the arrays
    behind the nullable, string, datetime-with-timezone, interval and
    period dtypes. This handler returns ``True`` if the two arrays are
    equal (values and dtype), otherwise ``False``. This handler is
    designed to be used at the end of the chain of responsibility. This
    handler does not call the next handler.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import PandasExtensionArrayEqualHandler
        >>> config = EqualityConfig()
        >>> handler = PandasExtensionArrayEqualHandler()
        >>> handler.handle(pd.Categorical(["a", "b"]), pd.Categorical(["a", "b"]), config)
        True
        >>> handler.handle(pd.Categorical(["a", "b"]), pd.Categorical(["a", "c"]), config)
        False

        ```
    """

    def handle(
        self,
        actual: Any,
        expected: Any,
        config: EqualityConfig,
    ) -> bool:
        object_equal = extension_array_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas arrays"),
            )
        return object_equal


def index_equal(index1: pd.Index, index2: pd.Index, config: EqualityConfig) -> bool:
    r"""Indicate if the two indexes are equal or not.

    Args:
        index1: The first index to compare.
        index2: The second index to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two indexes are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pandas import index_equal
        >>> config = EqualityConfig()
        >>> index_equal(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]), config)
        True
        >>> index_equal(pd.Index([1, 2, 3]), pd.Index([1, 2, 4]), config)
        False

        ```
    """
    if index1.shape != index2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and _index_has_nan(index1):
        return False
    try:
        cast("Any", pd.testing).assert_index_equal(
            index1,
            index2,
            exact=True,
            check_exact=_check_exact(config, index1.dtype),
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True


def _is_same_duration_dtype(series1: pd.Series, series2: pd.Series) -> bool:
    return series1.dtype == series2.dtype and series1.dtype.kind == "m"


def _to_seconds(series: pd.Series) -> pd.Series:
    return cast("Any", series).dt.total_seconds()


def _durations_to_seconds_frames(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[Any, Any]:
    r"""Convert the duration columns that have the same dtype to float
    seconds, so the tolerance applies to them (a duration is a quantity,
    unlike an instant).

    The other columns are not changed, and the dtypes are still compared
    by pandas.
    """
    positions = [
        i
        for i, (dtype1, dtype2) in enumerate(zip(df1.dtypes, df2.dtypes, strict=True))
        if dtype1 == dtype2 and dtype1.kind == "m"
    ]
    if not positions:
        return df1, df2
    df1, df2 = df1.copy(), df2.copy()
    for i in positions:
        cast("Any", df1).isetitem(i, _to_seconds(df1.iloc[:, i]))
        cast("Any", df2).isetitem(i, _to_seconds(df2.iloc[:, i]))
    return df1, df2


def _tolerance_applies(config: EqualityConfig) -> bool:
    # The tolerance of the numeric columns is applied with NumPy (the shared rule).
    return (config.atol > 0 or config.rtol > 0) and is_numpy_available()


def _check_exact(config: EqualityConfig, dtype: Any) -> bool:
    # A boolean is a label, not a quantity: the tolerance never applies to it.
    return (config.atol == 0 and config.rtol == 0) or bool(pd.api.types.is_bool_dtype(dtype))


def _is_boolean_pair(series1: pd.Series, series2: pd.Series) -> bool:
    return series1.dtype == series2.dtype and bool(pd.api.types.is_bool_dtype(series1.dtype))


def _align_boolean_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame | None:
    r"""Compare the boolean columns exactly and copy the values of
    ``df1`` in ``df2``, so ``pandas.testing`` does not apply the
    tolerance to them.

    Returns:
        The data frame ``df2`` with the values of the boolean columns
            of ``df1``, or ``None`` if one boolean column differs.
    """
    positions = [i for i in range(df1.shape[1]) if _is_boolean_pair(df1.iloc[:, i], df2.iloc[:, i])]
    if not positions:
        return df2
    df2 = df2.copy()
    for i in positions:
        if not pd.Series(df1.iloc[:, i].array).equals(pd.Series(df2.iloc[:, i].array)):
            return None
        cast("Any", df2).isetitem(i, df1.iloc[:, i].array)
    return df2


def _is_numeric_pair(series1: pd.Series, series2: pd.Series) -> bool:
    dtype = series1.dtype
    return dtype == series2.dtype and bool(
        pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)
    )


def _numeric_values_close(series1: pd.Series, series2: pd.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the values of two numeric series are equal within a
    tolerance.

    ``pandas.testing`` applies its own tolerance rule. The missing
    values must be at the same positions, then the values are compared
    with the NumPy rule (``|a - b| <= atol + rtol * max(|a|, |b|)``),
    like the other libraries. The index and the name are not compared.
    """
    from coola.equality.handler.numpy import array_equal  # noqa: PLC0415

    if (
        not pd.api.types.is_float_dtype(series1.dtype)
        and not (series1.isna().to_numpy() == series2.isna().to_numpy()).all()
    ):
        return False
    return array_equal(_to_numpy(series1), _to_numpy(series2), config)


def _to_numpy(series: pd.Series) -> Any:
    # A float NaN stays a NaN (governed by ``equal_nan``). The other missing values are
    # replaced by zero: their positions are compared apart.
    if pd.api.types.is_float_dtype(series.dtype):
        return series.to_numpy(dtype="float64", na_value=float("nan"))
    if series.isna().any():
        return series.fillna(0).to_numpy(dtype="float64")
    return series.to_numpy()


def _with_values(series: pd.Series, values: pd.Series) -> pd.Series:
    r"""Return a series with the index and the name of ``series`` and
    the values of ``values``."""
    return pd.Series(values.array, index=series.index, name=series.name)


def _align_numeric_columns(
    df1: pd.DataFrame, df2: pd.DataFrame, config: EqualityConfig
) -> pd.DataFrame | None:
    r"""Compare the numeric columns with the shared rule and copy the
    values of ``df1`` in ``df2``, so the other checks of pandas (index,
    columns, dtypes, other columns) do not see a difference.

    Returns:
        The data frame ``df2`` with the values of the numeric columns
            of ``df1``, or ``None`` if one numeric column is not close.
    """
    positions = [i for i in range(df1.shape[1]) if _is_numeric_pair(df1.iloc[:, i], df2.iloc[:, i])]
    if not positions:
        return df2
    df2 = df2.copy()
    for i in positions:
        if not _numeric_values_close(df1.iloc[:, i], df2.iloc[:, i], config):
            return None
        cast("Any", df2).isetitem(i, df1.iloc[:, i].array)
    return df2


def _index_has_nan(index: pd.Index) -> bool:
    if isinstance(index, pd.MultiIndex):  # ``isna`` is not defined for MultiIndex
        # A level that holds only missing values has the dtype ``object``, whatever the values
        # were: it is conservatively considered as ``NaN``.
        return any(
            bool((codes == -1).any()) and (len(level) == 0 or _has_nan_dtype(level.dtype))
            for level, codes in zip(index.levels, index.codes, strict=True)
        )
    return _has_nan(index)


def _has_nan(values: Any) -> bool:
    r"""Indicate if a series, an index or an array has a ``NaN``.

    There are two kinds of missing values. A ``NaN`` (in a float or
    complex dtype) is governed by ``equal_nan``. A null (``None``,
    ``pandas.NA`` or ``NaT`` in the other dtypes) is equal to a null,
    whatever ``equal_nan``: a copy of a data frame must be equal to the
    data frame.
    """
    return _has_nan_dtype(values.dtype) and bool(values.isna().any())


def _has_nan_dtype(dtype: Any) -> bool:
    return bool(pd.api.types.is_float_dtype(dtype) or pd.api.types.is_complex_dtype(dtype))


def _frame_has_nan(df: pd.DataFrame) -> bool:
    return any(_has_nan(df.iloc[:, i]) for i in range(df.shape[1]))


def extension_array_equal(array1: Any, array2: Any, config: EqualityConfig) -> bool:
    r"""Indicate if the two pandas extension arrays are equal or not.

    Args:
        array1: The first array to compare.
        array2: The second array to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two arrays are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pandas import extension_array_equal
        >>> config = EqualityConfig()
        >>> extension_array_equal(pd.array([1, 2], dtype="Int64"), pd.array([1, 2], dtype="Int64"), config)
        True
        >>> extension_array_equal(pd.array([1, 2], dtype="Int64"), pd.array([1, 3], dtype="Int64"), config)
        False

        ```
    """
    if array1.shape != array2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and _has_nan(array1):
        return False
    try:
        cast("Any", pd.testing).assert_extension_array_equal(
            array1,
            array2,
            check_exact=_check_exact(config, array1.dtype),
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True


def frame_equal(df1: pd.DataFrame, df2: pd.DataFrame, config: EqualityConfig) -> bool:
    r"""Indicate if the two DataFrames are equal or not.

    Args:
        df1: The first DataFrame to compare.
        df2: The second DataFrame to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two DataFrames are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import pandas
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pandas import frame_equal
        >>> config = EqualityConfig()
        >>> frame_equal(
        ...     pandas.DataFrame({"col": [1, 2, 3]}), pandas.DataFrame({"col": [1, 2, 3]}), config
        ... )
        True
        >>> frame_equal(
        ...     pandas.DataFrame({"col": [1, 2, 3]}), pandas.DataFrame({"col": [1, 2, 4]}), config
        ... )
        False

        ```
    """
    if df1.shape != df2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and _frame_has_nan(df1):
        return False
    if config.atol > 0 or config.rtol > 0:
        df1, df2 = _durations_to_seconds_frames(df1, df2)
    if config.atol > 0 or config.rtol > 0:
        aligned = _align_boolean_columns(df1, df2)
        if aligned is None:
            return False
        df2 = aligned
    if _tolerance_applies(config):
        aligned = _align_numeric_columns(df1, df2, config)
        if aligned is None:
            return False
        df2 = aligned
    try:
        cast("Any", pd.testing).assert_frame_equal(
            df1,
            df2,
            check_exact=config.atol == 0 and config.rtol == 0,
            check_index_type=True,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True


def series_equal(series1: pd.Series, series2: pd.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the two series are equal or not.

    Args:
        series1: The first series to compare.
        series2: The second series to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two series are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> import pandas
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.pandas import series_equal
        >>> config = EqualityConfig()
        >>> series_equal(pandas.Series([1, 2, 3]), pandas.Series([1, 2, 3]), config)
        True
        >>> series_equal(pandas.Series([1, 2, 3]), pandas.Series([1, 2, 4]), config)
        False

        ```
    """
    if series1.shape != series2.shape:  # cheap check before scanning for NaN
        return False
    if not config.equal_nan and _has_nan(series1):
        return False
    if (config.atol > 0 or config.rtol > 0) and _is_same_duration_dtype(series1, series2):
        series1, series2 = _to_seconds(series1), _to_seconds(series2)
    if _tolerance_applies(config) and _is_numeric_pair(series1, series2):
        if not _numeric_values_close(series1, series2, config):
            return False
        series2 = _with_values(series2, series1)
    try:
        cast("Any", pd.testing).assert_series_equal(
            series1,
            series2,
            check_exact=_check_exact(config, series1.dtype),
            check_index_type=True,
            atol=config.atol,
            rtol=config.rtol,
        )
    except AssertionError:
        return False
    return True
