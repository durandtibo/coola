r"""Provide deterministic hashing for pandas objects."""

from __future__ import annotations

__all__ = [
    "PandasDataFrameHasher",
    "PandasIntervalHasher",
    "PandasMissingHasher",
    "PandasPeriodHasher",
    "PandasSeriesHasher",
]

import math
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node, tagged
from coola.utils.imports import is_pandas_available

if TYPE_CHECKING or is_pandas_available():
    import numpy as np
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

if TYPE_CHECKING:
    from coola.hashing.base import Hasher


def _dtype_key(dtype: Any) -> Any:
    r"""Return the dtype in a form that can be hashed: the categories and
    the ``ordered`` flag are added to a categorical dtype.

    Args:
        dtype: The data type.

    Returns:
        The data type, in a form that can be hashed.
    """
    if isinstance(dtype, pd.CategoricalDtype):
        return [str(dtype), dtype.ordered, dtype.categories.to_numpy()]
    return str(dtype)


def _normalize_name(name: Any) -> Any:
    r"""Return ``None`` for a missing name (``NaN``), and ``name``
    otherwise.

    A series named ``None`` and a series named ``NaN`` are equal for
    ``objects_are_equal``, so they must have the same hash.

    Args:
        name: The name of a series or an index.

    Returns:
        ``None`` for a ``NaN``, otherwise ``name``.
    """
    return None if isinstance(name, float) and math.isnan(name) else name


def _column_values(column: Any) -> Any:
    r"""Return the values of a column in a form that can be hashed.

    The NumPy dtypes, the categorical and the sparse dtypes are hashed
    as a NumPy array. The other extension dtypes (``Int64``,
    ``boolean``, ``string``, tz-aware datetimes, periods, intervals,
    ...) are converted to an object array where a missing value is
    ``None``, because ``to_numpy()`` would put ``pandas.NA`` or
    ``pandas.NaT`` in the array.

    Args:
        column: The column.

    Returns:
        The values, in a form that can be hashed.
    """
    dtype = column.dtype
    if isinstance(dtype, (np.dtype, pd.CategoricalDtype, pd.SparseDtype)):
        return column.to_numpy()
    return column.to_numpy(dtype=object, na_value=None)


def _index_key(index: pd.Index) -> list[Any]:
    r"""Return an index in a form that can be hashed: its names, its
    dtype(s) and its values.

    The values of a ``MultiIndex`` are described level by level, not as
    an array of tuples.

    Args:
        index: The index.

    Returns:
        The names, dtypes and values of the index, in a form that can be
            hashed.
    """
    levels = (
        [index.get_level_values(i) for i in range(index.nlevels)]
        if isinstance(index, pd.MultiIndex)
        else [index]
    )
    key: list[Any] = [[_normalize_name(name) for name in index.names]]
    for level in levels:
        key.extend([_dtype_key(level.dtype), _column_values(level)])
    return key


class PandasSeriesHasher(InlineDisplayMixin, BaseHasher["pd.Series"]):
    r"""Hasher for ``pandas.Series`` objects.

    The hash depends on the name, the dtype, the index (its names, its
    dtype and its values) and the values. For a categorical dtype, the
    categories and the ``ordered`` flag are part of the dtype.
    The values are hashed as a NumPy array, so the elements of an
    ``object`` column are hashed through the registry. The values of
    the other extension dtypes are converted to an ``object`` array in
    which a missing value is ``None``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.hashing import PandasSeriesHasher
        >>> hasher = PandasSeriesHasher()
        >>> hasher
        PandasSeriesHasher()
        >>> hasher.encode(pd.Series([1, 2, 3])).tag
        'pandas.Series'

        ```
    """

    def encode(self, data: pd.Series) -> Encoding:
        r"""Describe a pandas series.

        Args:
            data: The series to describe.

        Returns:
            The ``Node`` of the name, the dtype, the index and the
            values.
        """
        return Node(
            "pandas.Series",
            [
                _normalize_name(data.name),
                _dtype_key(data.dtype),
                _index_key(data.index),
                _column_values(data),
            ],
        )


class PandasDataFrameHasher(InlineDisplayMixin, BaseHasher["pd.DataFrame"]):
    r"""Hasher for ``pandas.DataFrame`` objects.

    The hash depends on the column labels (and the name of the columns
    index), the index (its names, its dtype and its values) and, for
    each column, its dtype and its values (see ``PandasSeriesHasher``).

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.hashing import PandasDataFrameHasher
        >>> hasher = PandasDataFrameHasher()
        >>> hasher
        PandasDataFrameHasher()
        >>> hasher.encode(pd.DataFrame({"a": [1, 2]})).tag
        'pandas.DataFrame'

        ```
    """

    def encode(self, data: pd.DataFrame) -> Encoding:
        r"""Describe a pandas data frame.

        Args:
            data: The data frame to describe.

        Returns:
            The ``Node`` of the columns index, the index and, for each
            column, its dtype and its values.
        """
        children: list[Any] = [_index_key(data.columns), _index_key(data.index)]
        for i in range(data.shape[1]):
            column = data.iloc[:, i]
            children.extend([_dtype_key(column.dtype), _column_values(column)])
        return Node("pandas.DataFrame", children)


class PandasMissingHasher(InlineDisplayMixin, BaseHasher[Any]):
    r"""Hasher for the pandas missing value markers ``pandas.NA`` and
    ``pandas.NaT``.

    ``pandas.NaT`` is a ``datetime`` subclass, so this hasher has to be
    registered for its type to be found before ``DatetimeHasher``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.hashing import get_default_registry
        >>> from coola.hashing.pandas import PandasMissingHasher
        >>> hasher = PandasMissingHasher()
        >>> hasher
        PandasMissingHasher()
        >>> hasher.encode(pd.NaT).startswith(b'pandas.')
        True

        ```
    """

    def encode(self, data: Any) -> Encoding:
        r"""Describe a missing value.

        Args:
            data: The missing value (``pd.NA`` or ``pd.NaT``).

        Returns:
            The leaf with the tag ``pandas.NA`` or ``pandas.NaTType``.
        """
        return tagged(f"pandas.{type(data).__name__}")


class PandasPeriodHasher(InlineDisplayMixin, BaseHasher["pd.Period"]):
    r"""Hasher for ``pandas.Period`` objects.

    The hash depends on the frequency and the ordinal.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.hashing import get_default_registry
        >>> from coola.hashing.pandas import PandasPeriodHasher
        >>> hasher = PandasPeriodHasher()
        >>> hasher
        PandasPeriodHasher()
        >>> hasher.encode(pd.Period("2020")).startswith(b'pandas.Period')
        True

        ```
    """

    def encode(self, data: pd.Period) -> Encoding:
        r"""Describe a period.

        Args:
            data: The period to describe.

        Returns:
            The leaf of the frequency and the ordinal.
        """
        return tagged("pandas.Period", f"{data.freqstr}\0{data.ordinal}")


class PandasIntervalHasher(InlineDisplayMixin, BaseHasher["pd.Interval"]):
    r"""Hasher for ``pandas.Interval`` objects.

    The hash depends on the bounds (hashed through the registry) and on
    the side(s) on which the interval is closed.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.hashing import get_default_registry
        >>> from coola.hashing.pandas import PandasIntervalHasher
        >>> hasher = PandasIntervalHasher()
        >>> hasher
        PandasIntervalHasher()
        >>> hasher.encode(pd.Interval(0, 1)).tag
        'pandas.Interval'

        ```
    """

    def encode(self, data: pd.Interval) -> Encoding:
        r"""Describe an interval.

        Args:
            data: The interval to describe.

        Returns:
            The ``Node`` of the bounds and the closed side.
        """
        return Node("pandas.Interval", [data.left, data.right, data.closed])


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the pandas objects.

    Returns:
        The hashers of the pandas objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the pandas objects.
    """
    if not is_pandas_available():
        return {}
    missing = PandasMissingHasher()
    return {
        pd.Series: PandasSeriesHasher(),
        pd.DataFrame: PandasDataFrameHasher(),
        type(pd.NA): missing,
        type(pd.NaT): missing,
        pd.Period: PandasPeriodHasher(),
        pd.Interval: PandasIntervalHasher(),
    }
