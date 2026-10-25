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
from coola.utils.imports import is_pandas_available

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
        if config.explaining and not object_equal:
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
        if config.explaining and not object_equal:
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
        if config.explaining and not object_equal:
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
        if config.explaining and not object_equal:
            report_difference(
                config,
                format_value_difference(actual=actual, expected=expected, name="pandas arrays"),
            )
        return object_equal


def index_equal(index1: pd.Index, index2: pd.Index, config: EqualityConfig) -> bool:
    r"""Indicate if the two indexes are equal or not.

    The indexes are equal if they have the same class, shape, dtype,
    names and values. An index is a label, not a quantity: the
    tolerance does not apply to it.

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
    if type(index1) is not type(index2) or index1.shape != index2.shape:
        return False
    if index1.names != index2.names or not _same_dtypes(index1, index2):
        return False
    if not config.equal_nan and _index_has_nan(index1):
        return False
    return bool(index1.equals(index2))


def _same_dtypes(obj1: pd.Index | pd.DataFrame, obj2: pd.Index | pd.DataFrame) -> bool:
    if isinstance(obj1, pd.MultiIndex):
        return list(obj1.dtypes) == list(cast("pd.MultiIndex", obj2).dtypes)
    if isinstance(obj1, pd.DataFrame):
        return list(obj1.dtypes) == list(cast("pd.DataFrame", obj2).dtypes)
    return bool(obj1.dtype == obj2.dtype)


def _is_numeric_dtype(dtype: Any) -> bool:
    return bool(
        pd.api.types.is_integer_dtype(dtype)
        or pd.api.types.is_float_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
    )


def _is_duration_dtype(dtype: Any) -> bool:
    return getattr(dtype, "kind", None) == "m"


def _values_equal(series1: pd.Series, series2: pd.Series, config: EqualityConfig) -> bool:
    r"""Compare the values of two series that have the same dtype and the
    same index, by the kind of their dtype.

    - numeric (integers, floats, complex, also the nullable ones): the
      null values must be at the same positions, then the values are
      compared with the shared numeric rule (tolerance, ``equal_nan``).
    - duration (``timedelta64``): same, on the durations.
    - other (booleans, strings, categories, instants, periods,
      intervals, decimals, nested values and ``object``): the
      ``equals`` of pandas. It never sees the tolerance or
      ``equal_nan``, and a null is equal to a null.
    """
    dtype = series1.dtype
    if _is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        return _numeric_values_close(series1, series2, config)
    if _is_duration_dtype(dtype):
        return _duration_values_close(series1, series2, config)
    return bool(series1.equals(series2.set_axis(series1.index)))


def _numeric_values_close(series1: pd.Series, series2: pd.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the values of two numeric series are equal within a
    tolerance.

    The missing values must be at the same positions (a float ``NaN``
    is governed by ``equal_nan``), then the values are compared with the
    NumPy rule (``|a - b| <= atol + rtol * max(|a|, |b|)``), like the
    other libraries. The index and the name are not compared.
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
        # The integers stay integers: a conversion to ``float64`` rounds the large ones.
        return series.fillna(0).to_numpy()
    return series.to_numpy()


def _duration_values_close(series1: pd.Series, series2: pd.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the values of two ``timedelta64`` series are equal
    within a tolerance (``atol`` is a number of seconds).

    ``NaT`` is a null: it must be at the same positions, whatever
    ``equal_nan``.
    """
    from coola.equality.handler.numpy import array_equal  # noqa: PLC0415

    nat1, nat2 = series1.isna().to_numpy(), series2.isna().to_numpy()
    if not (nat1 == nat2).all():
        return False
    zero = pd.Timedelta(0)
    return array_equal(series1.fillna(zero).to_numpy(), series2.fillna(zero).to_numpy(), config)


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


def _same_label(label1: Any, label2: Any) -> bool:
    # A name can be ``NaN`` or a tuple: compare them as the labels of an index.
    return bool(
        pd.Index([label1], tupleize_cols=False).equals(pd.Index([label2], tupleize_cols=False))
    )


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
    if array1.shape != array2.shape or array1.dtype != array2.dtype:
        return False
    if not config.equal_nan and _has_nan(array1):
        return False
    return _values_equal(pd.Series(array1), pd.Series(array2), config)


def frame_equal(df1: pd.DataFrame, df2: pd.DataFrame, config: EqualityConfig) -> bool:
    r"""Indicate if the two DataFrames are equal or not.

    The structure is compared first (shape, index, columns and dtypes),
    then the columns one by one, by the kind of their dtype (see
    ``_values_equal``). The comparison stops at the first column that
    differs.

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
    if df1.shape != df2.shape or not _same_dtypes(df1, df2):
        return False
    if not index_equal(df1.index, df2.index, config) or not index_equal(
        df1.columns, df2.columns, config
    ):
        return False
    for i in range(df1.shape[1]):
        column1, column2 = df1.iloc[:, i], df2.iloc[:, i]
        if not config.equal_nan and _has_nan(column1):
            return False
        if not _values_equal(column1, column2, config):
            return False
    return True


def series_equal(series1: pd.Series, series2: pd.Series, config: EqualityConfig) -> bool:
    r"""Indicate if the two series are equal or not.

    The shape, the name, the dtype and the index are compared first,
    then the values, by the kind of the dtype (see ``_values_equal``).

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
    if (
        series1.shape != series2.shape
        or series1.dtype != series2.dtype
        or not _same_label(series1.name, series2.name)
        or not index_equal(series1.index, series2.index, config)
    ):
        return False
    if not config.equal_nan and _has_nan(series1):
        return False
    return _values_equal(series1, series2, config)
