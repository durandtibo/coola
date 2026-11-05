r"""Implement the adapters of the pandas objects.

This module provides the adapters that convert a ``pandas.DataFrame``,
``pandas.Series``, ``pandas.Index`` or pandas extension array to a
table, which is compared by the ``TableEqualityTester``.
"""

from __future__ import annotations

__all__ = [
    "PandasArrayTable",
    "PandasFrameTable",
    "PandasIndexTable",
    "PandasSeriesTable",
    "pandas_array",
    "pandas_frame",
    "pandas_index",
    "pandas_series",
]

from typing import TYPE_CHECKING, Any, cast

from coola.equality.tester.table import Column, TableEqualityTester
from coola.utils.imports import is_pandas_available

if TYPE_CHECKING or is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine
    from coola.equality.tester.base import BaseEqualityTester


class PandasFrameTable:
    r"""Adapt a ``pandas.DataFrame`` to a table.

    The structure is the shape, the data types, the index and the
    columns. The columns are compared one by one, by the kind of their
    data type.

    Args:
        frame: The data frame.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.tester.pandas import PandasFrameTable
        >>> table = PandasFrameTable(pd.DataFrame({"a": [1, 2]}))
        >>> table.kind_name
        'tables'

        ```
    """

    kind_name = "tables"

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def same_structure(self, other: PandasFrameTable, ctx: Engine) -> bool:
        r"""Indicate if two data frames have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        df1, df2 = self._frame, other._frame
        return (
            df1.shape == df2.shape
            and _same_dtypes(df1, df2)
            and _same_index(df1.index, df2.index, ctx)
            and _same_index(df1.columns, df2.columns, ctx)
        )

    def structure_difference(self, other: PandasFrameTable, ctx: Engine) -> str:
        r"""Say what differs in the structure of two data frames.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            A short text, for example ``"shapes differ: (3, 2) vs (2, 2)"``.
        """
        df1, df2 = self._frame, other._frame
        if df1.shape != df2.shape:
            return f"shapes differ: {df1.shape} vs {df2.shape}"
        if not _same_dtypes(df1, df2):
            return _dtypes_difference(df1.columns, df1.dtypes, df2.dtypes)
        if not _same_index(df1.index, df2.index, ctx):
            return _index_difference("index", df1.index, df2.index, ctx)
        return _index_difference("columns", df1.columns, df2.columns, ctx)

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of each data frame column.
        """
        for i, name in enumerate(self._frame.columns):
            yield _column(name, self._frame.iloc[:, i])


class PandasSeriesTable:
    r"""Adapt a ``pandas.Series`` to a table with one column.

    The structure is the shape, the name, the data type and the index.

    Args:
        series: The series.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.tester.pandas import PandasSeriesTable
        >>> table = PandasSeriesTable(pd.Series([1, 2]))
        >>> table.kind_name
        'series'

        ```
    """

    kind_name = "series"

    def __init__(self, series: pd.Series) -> None:
        self._series = series

    def same_structure(self, other: PandasSeriesTable, ctx: Engine) -> bool:
        r"""Indicate if two series have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        s1, s2 = self._series, other._series
        return (
            s1.shape == s2.shape
            and s1.dtype == s2.dtype
            and _same_label(s1.name, s2.name)
            and _same_index(s1.index, s2.index, ctx)
        )

    def structure_difference(self, other: PandasSeriesTable, ctx: Engine) -> str:
        r"""Say what differs in the structure of two series.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            A short text, for example ``"names differ: 'a' vs 'b'"``.
        """
        s1, s2 = self._series, other._series
        if s1.shape != s2.shape:
            return f"shapes differ: {s1.shape} vs {s2.shape}"
        if s1.dtype != s2.dtype:
            return f"dtypes differ: {s1.dtype} vs {s2.dtype}"
        if not _same_label(s1.name, s2.name):
            return f"names differ: {s1.name!r} vs {s2.name!r}"
        return _index_difference("index", s1.index, s2.index, ctx)

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of the series.
        """
        yield _column(None, self._series)


class PandasArrayTable:
    r"""Adapt a pandas extension array to a table with one column.

    The extension arrays include ``pandas.Categorical`` and the arrays
    behind the nullable, string, interval and period data types.

    Args:
        array: The extension array.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.tester.pandas import PandasArrayTable
        >>> table = PandasArrayTable(pd.array([1, 2], dtype="Int64"))
        >>> table.kind_name
        'arrays'

        ```
    """

    kind_name = "arrays"

    def __init__(self, array: Any) -> None:
        self._array = array

    def same_structure(self, other: PandasArrayTable, ctx: Engine) -> bool:  # noqa: ARG002
        r"""Indicate if two extension arrays have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        return bool(
            self._array.shape == other._array.shape and self._array.dtype == other._array.dtype
        )

    def structure_difference(self, other: PandasArrayTable, ctx: Engine) -> str:  # noqa: ARG002
        r"""Say what differs in the structure of two extension arrays.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            A short text, for example ``"dtypes differ: Int64 vs Float64"``.
        """
        a1, a2 = self._array, other._array
        if a1.shape != a2.shape:
            return f"shapes differ: {a1.shape} vs {a2.shape}"
        return f"dtypes differ: {a1.dtype} vs {a2.dtype}"

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of the array.
        """
        yield _column(None, pd.Series(self._array))


class PandasIndexTable:
    r"""Adapt a ``pandas.Index`` to a table with one exact column.

    An index is a label, not a quantity: the tolerance does not apply to
    it. It has the same class, shape, data type and names, and equal
    values.

    Args:
        index: The index.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality.tester.pandas import PandasIndexTable
        >>> table = PandasIndexTable(pd.Index([1, 2]))
        >>> table.kind_name
        'indexes'

        ```
    """

    kind_name = "indexes"

    def __init__(self, index: pd.Index) -> None:
        self._index = index

    def same_structure(self, other: PandasIndexTable, ctx: Engine) -> bool:
        r"""Indicate if two indexes have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        i1, i2 = self._index, other._index
        return (
            type(i1) is type(i2)
            and i1.shape == i2.shape
            and i1.names == i2.names
            and _same_dtypes(i1, i2)
            and (ctx.equal_nan or not _index_has_nan(i1))
        )

    def structure_difference(self, other: PandasIndexTable, ctx: Engine) -> str:
        r"""Say what differs in the structure of two indexes.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            A short text, for example ``"index names differ: ['i'] vs [None]"``.
        """
        return _index_difference("index", self._index, other._index, ctx)

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of the index, compared exactly.
        """
        yield Column(
            None, "exact", self._index, equals=_index_equals, difference=_index_difference_row
        )


def pandas_frame(frame: pd.DataFrame) -> PandasFrameTable:
    r"""Adapt a ``pandas.DataFrame`` to a table.

    Args:
        frame: The data frame to adapt.

    Returns:
        The table.
    """
    return PandasFrameTable(frame)


def pandas_series(series: pd.Series) -> PandasSeriesTable:
    r"""Adapt a ``pandas.Series`` to a table.

    Args:
        series: The series to adapt.

    Returns:
        The table.
    """
    return PandasSeriesTable(series)


def pandas_array(array: Any) -> PandasArrayTable:
    r"""Adapt a pandas extension array to a table.

    Args:
        array: The extension array to adapt.

    Returns:
        The table.
    """
    return PandasArrayTable(array)


def pandas_index(index: pd.Index) -> PandasIndexTable:
    r"""Adapt a ``pandas.Index`` to a table.

    Args:
        index: The index to adapt.

    Returns:
        The table.
    """
    return PandasIndexTable(index)


def _same_index(index1: pd.Index, index2: pd.Index, ctx: Engine) -> bool:
    r"""Indicate if two indexes have the same structure and labels.

    Args:
        index1: The first index.
        index2: The second index.
        ctx: The ``Engine`` of the comparison.

    Returns:
        ``True`` if the indexes are equal, otherwise ``False``.
    """
    return PandasIndexTable(index1).same_structure(PandasIndexTable(index2), ctx) and bool(
        index1.equals(index2)
    )


def _index_difference_row(index1: pd.Index, index2: pd.Index) -> tuple[int, Any, Any] | None:
    r"""Find the first position where two indexes differ.

    Args:
        index1: The first index.
        index2: The second index, with the same type and length.

    Returns:
        ``(position, label, other label)`` for the first position that
            differs, or ``None`` if it cannot be found.
    """
    return _series_difference(pd.Series(index1.to_numpy()), pd.Series(index2.to_numpy()))


def _dtypes_difference(names: Any, dtypes1: Any, dtypes2: Any) -> str:
    r"""Describe the columns that have different dtypes.

    Args:
        names: The names of the columns.
        dtypes1: The dtypes of the first columns.
        dtypes2: The dtypes of the second columns.

    Returns:
        A text that lists the first columns with different dtypes.
    """
    pairs = [
        f"{name}: {dtype1} vs {dtype2}"
        for name, dtype1, dtype2 in zip(names, dtypes1, dtypes2, strict=True)
        if dtype1 != dtype2
    ]
    shown = ", ".join(pairs[:3])
    more = f" (+{len(pairs) - 3} more)" if len(pairs) > 3 else ""
    return f"dtypes differ: {shown}{more}"


def _index_difference(label: str, index1: pd.Index, index2: pd.Index, ctx: Engine) -> str:
    r"""Say what differs between two indexes.

    Args:
        label: The word that names the indexes in the text, for example
            ``"index"`` or ``"columns"``.
        index1: The first index.
        index2: The second index.
        ctx: The ``Engine`` of the comparison.

    Returns:
        A short text.
    """
    if type(index1) is not type(index2):
        return f"{label} classes differ: {type(index1).__name__} vs {type(index2).__name__}"
    if index1.shape != index2.shape:
        return f"{label} shapes differ: {index1.shape} vs {index2.shape}"
    if index1.names != index2.names:
        return f"{label} names differ: {list(index1.names)} vs {list(index2.names)}"
    if not _same_dtypes(index1, index2):
        return f"{label} dtypes differ: {index1.dtype} vs {index2.dtype}"
    if not ctx.equal_nan and _index_has_nan(index1):
        return f"{label} has a NaN, which is not equal to NaN (use equal_nan=True)"
    verb = "differ" if label == "columns" else "values differ"
    return f"{label} {verb}: {_short_list(index1)} vs {_short_list(index2)}"


def _short_list(index: pd.Index, limit: int = 5) -> str:
    r"""Return the first labels of an index as a text.

    Args:
        index: The index.
        limit: The maximum number of labels shown.

    Returns:
        The labels as a list, followed by ``...`` if some are not shown.
    """
    labels = list(index[:limit])
    if len(index) > limit:
        return f"{str(labels)[:-1]}, ...]"
    return str(labels)


def _index_equals(index1: pd.Index, index2: pd.Index) -> bool:
    r"""Indicate if two indexes are equal.

    Args:
        index1: The first index.
        index2: The second index.

    Returns:
        ``True`` if the indexes are equal, otherwise ``False``.
    """
    return bool(index1.equals(index2))


def _same_dtypes(obj1: pd.Index | pd.DataFrame, obj2: pd.Index | pd.DataFrame) -> bool:
    r"""Indicate if two indexes or data frames have the same dtypes.

    Args:
        obj1: The first index or data frame.
        obj2: The second index or data frame, with the same type.

    Returns:
        ``True`` if the dtypes are the same, otherwise ``False``.
    """
    if isinstance(obj1, pd.MultiIndex):
        return list(obj1.dtypes) == list(cast("pd.MultiIndex", obj2).dtypes)
    if isinstance(obj1, pd.DataFrame):
        return list(obj1.dtypes) == list(cast("pd.DataFrame", obj2).dtypes)
    return bool(obj1.dtype == obj2.dtype)


def _column(name: object, series: pd.Series) -> Column:
    r"""Convert a series to a column, by the kind of its data type.

    - numeric (integers, floats, complex, also the nullable ones): the
      null values are at the same positions, then the values are
      compared with the shared numeric rule. A float ``NaN`` (also the
      missing value of a nullable float) is governed by ``equal_nan``.
    - duration (``timedelta64``): same, on the durations.
    - other (booleans, strings, categories, instants, periods,
      intervals, decimals, nested values and ``object``): the
      ``equals`` of pandas. It never sees the tolerance or
      ``equal_nan``, and a null is equal to a null.

    Args:
        name: The name of the column.
        series: The series of the column.

    Returns:
        The column.
    """
    dtype = series.dtype
    if _is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        if pd.api.types.is_float_dtype(dtype):
            return Column(name, "numeric", series.to_numpy(dtype="float64", na_value=float("nan")))
        nulls = series.isna().to_numpy()
        # The integers stay integers: a conversion to ``float64`` rounds the large ones.
        # pandas < 2.2 converts a nullable integer array to ``object`` unless the dtype is given.
        numpy_dtype = getattr(dtype, "numpy_dtype", None)
        values = series.fillna(0).to_numpy(dtype=numpy_dtype)
        return Column(name, "numeric", values, nulls)
    if getattr(dtype, "kind", None) == "m":
        nulls = series.isna().to_numpy()
        return Column(name, "duration", series.fillna(pd.Timedelta(0)).to_numpy(), nulls)
    return Column(name, "exact", series, equals=_series_equals, difference=_series_difference)


def _series_equals(series1: pd.Series, series2: pd.Series) -> bool:
    r"""Indicate if two series are equal, ignoring their index.

    Args:
        series1: The first series.
        series2: The second series, which takes the index of
            ``series1``.

    Returns:
        ``True`` if the series are equal, otherwise ``False``.
    """
    return bool(series1.equals(series2.set_axis(series1.index)))


def _series_difference(series1: pd.Series, series2: pd.Series) -> tuple[int, Any, Any] | None:
    r"""Find the first row where two series differ, ignoring their index.

    Args:
        series1: The first series.
        series2: The second series, with the same dtype and length.

    Returns:
        ``(row, value, other value)`` for the first row that differs, or
            ``None`` if the row cannot be found.
    """
    values2 = series2.set_axis(series1.index)
    different = ~(series1.eq(values2) | (series1.isna() & values2.isna()))
    rows = different.to_numpy().nonzero()[0]
    if len(rows) == 0:
        return None
    row = int(rows[0])
    return row, series1.iloc[row], series2.iloc[row]


def _is_numeric_dtype(dtype: Any) -> bool:
    r"""Indicate if a dtype is an integer, float or complex dtype.

    Args:
        dtype: The dtype.

    Returns:
        ``True`` if the dtype is numeric, otherwise ``False``.
    """
    return bool(
        pd.api.types.is_integer_dtype(dtype)
        or pd.api.types.is_float_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
    )


def _index_has_nan(index: pd.Index) -> bool:
    r"""Indicate if an index can hold a ``NaN`` label.

    Args:
        index: The index.

    Returns:
        ``True`` if a label is ``NaN``, otherwise ``False``.
    """
    if isinstance(index, pd.MultiIndex):  # ``isna`` is not defined for MultiIndex
        # A level that holds only missing values has the dtype ``object``, whatever the values
        # were: it is conservatively considered as ``NaN``.
        return any(
            bool((codes == -1).any()) and (len(level) == 0 or _has_nan_dtype(level.dtype))
            for level, codes in zip(index.levels, index.codes, strict=True)
        )
    return _has_nan_dtype(index.dtype) and bool(index.isna().any())


def _has_nan_dtype(dtype: Any) -> bool:
    r"""Indicate if a dtype can hold a ``NaN``.

    Args:
        dtype: The dtype.

    Returns:
        ``True`` for a float or complex dtype, otherwise ``False``.
    """
    return bool(pd.api.types.is_float_dtype(dtype) or pd.api.types.is_complex_dtype(dtype))


def _same_label(label1: Any, label2: Any) -> bool:
    r"""Indicate if two labels are equal, as labels of an index.

    Args:
        label1: The first label.
        label2: The second label.

    Returns:
        ``True`` if the labels are equal, otherwise ``False``. A ``NaN``
            is equal to a ``NaN``.
    """
    # A name can be ``NaN`` or a tuple: compare them as the labels of an index.
    return bool(
        pd.Index([label1], tupleize_cols=False).equals(pd.Index([label2], tupleize_cols=False))
    )


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the pandas objects.

    Returns:
        The equality testers of the pandas objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the pandas objects.
    """
    if not is_pandas_available():
        return {}

    return {
        pd.DataFrame: TableEqualityTester(pandas_frame),
        pd.Series: TableEqualityTester(pandas_series),
        pd.Index: TableEqualityTester(pandas_index),
        pd.api.extensions.ExtensionArray: TableEqualityTester(pandas_array),
    }
