r"""Implement the rounding of the pandas Series and DataFrames."""

from __future__ import annotations

__all__ = ["get_default_rounders", "round_pandas"]

from typing import TYPE_CHECKING, Any

from coola.rounding.common import UnsupportedTypeError
from coola.rounding.numpy import round_numpy
from coola.utils.imports import is_pandas_available

if TYPE_CHECKING:
    from coola.rounding.common import Rounder, RoundMode

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd


def round_pandas(data: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round a pandas Series or DataFrame.

    Only the floating-point columns are rounded. The integer columns
    are not changed, and neither are the columns that are not numerical
    (strings, booleans, dates, ...). The missing values are kept.

    Args:
        data: The ``pandas.Series`` or ``pandas.DataFrame`` to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded Series or DataFrame. It never shares memory with
            ``data``.

    Raises:
        UnsupportedTypeError: If ``data`` is not a Series or a DataFrame, or if
            ``data`` is a Series that is not numerical.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.rounding.pandas import round_pandas
        >>> round_pandas(pd.DataFrame({"a": [1.234, 5.678], "b": ["x", "y"]}), "round", 1)
             a  b
        0  1.2  x
        1  5.7  y

        ```
    """
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for position, dtype in enumerate(data.dtypes):
            if pd.api.types.is_float_dtype(dtype):
                result.isetitem(position, _round_series(data.iloc[:, position], mode, decimals))
        return result
    if isinstance(data, pd.Series):
        if not _is_supported(data.dtype):
            msg = f"Unsupported dtype: {data.dtype}"
            raise UnsupportedTypeError(msg)
        return _round_series(data, mode, decimals)
    msg = f"Unsupported type: {type(data).__qualname__}"
    raise UnsupportedTypeError(msg)


def _is_supported(dtype: Any) -> bool:
    r"""Indicate if a dtype can be rounded.

    Args:
        dtype: The dtype.

    Returns:
        ``True`` for a float or an integer (not boolean) dtype,
            otherwise ``False``.
    """
    return pd.api.types.is_float_dtype(dtype) or (
        pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
    )


def _round_series(series: pd.Series, mode: RoundMode, decimals: int) -> pd.Series:
    r"""Round a series.

    Args:
        series: The series.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        A new series. A series that is not a float one is copied, and
            the missing values are kept.
    """
    if not pd.api.types.is_float_dtype(series.dtype):
        return series.copy()
    values = series.to_numpy(dtype="float64", na_value=float("nan"))
    result = pd.Series(
        round_numpy(values, mode, decimals), index=series.index, name=series.name
    ).astype(series.dtype)
    # A missing value (``NaN`` or ``NA``) is kept as it is.
    return result.where(series.notna(), series)


def get_default_rounders() -> dict[type, Rounder]:
    r"""Get the default rounders of the pandas Series and DataFrames.

    Returns:
        The rounder of the pandas Series and DataFrames and their
            associated types.

    Notes:
        This function is called by the default registry the first time
        it needs the rounders of pandas.
    """
    if not is_pandas_available():
        return {}
    return {pd.Series: round_pandas, pd.DataFrame: round_pandas}
