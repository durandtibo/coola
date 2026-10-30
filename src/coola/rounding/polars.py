r"""Implement the rounding of the polars Series and DataFrames."""

from __future__ import annotations

__all__ = ["get_default_rounders", "round_polars"]

from typing import TYPE_CHECKING, Any

from coola.rounding.common import UnsupportedTypeError
from coola.rounding.numpy import round_numpy
from coola.utils.imports import check_numpy, is_polars_available

if TYPE_CHECKING:
    from coola.rounding.common import Rounder, RoundMode

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl


def round_polars(data: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round a polars Series or DataFrame.

    Only the floating-point columns are rounded. The integer columns
    are not changed, and neither are the columns that are not numerical
    (strings, booleans, dates, decimals, ...). The null values are
    kept. The values go through the NumPy kernel of
    ``coola.rounding.numpy``, so NumPy is required.

    Args:
        data: The ``polars.Series`` or ``polars.DataFrame`` to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded Series or DataFrame.

    Raises:
        MissingPackageError: If ``numpy`` is not installed.
        UnsupportedTypeError: If ``data`` is not a Series or a DataFrame, or if
            ``data`` is a Series that is not numerical.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.rounding.polars import round_polars
        >>> round_polars(pl.DataFrame({"a": [1.234, 5.678], "b": ["x", "y"]}), "round", 1)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ f64 ┆ str │
        ╞═════╪═════╡
        │ 1.2 ┆ x   │
        │ 5.7 ┆ y   │
        └─────┴─────┘

        ```
    """
    check_numpy()
    if isinstance(data, pl.DataFrame):
        return data.with_columns(
            _round_series(series, mode, decimals) for series in data if series.dtype.is_float()
        )
    if isinstance(data, pl.Series):
        if not (data.dtype.is_float() or data.dtype.is_integer()):
            msg = f"Unsupported dtype: {data.dtype}"
            raise UnsupportedTypeError(msg)
        return _round_series(data, mode, decimals)
    msg = f"Unsupported type: {type(data).__qualname__}"
    raise UnsupportedTypeError(msg)


def _round_series(series: pl.Series, mode: RoundMode, decimals: int) -> pl.Series:
    r"""Round a series.

    Args:
        series: The series.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        A new series. A series that is not a float one is cloned, and
            the null values are kept.
    """
    if not series.dtype.is_float():
        return series.clone()
    # ``to_numpy`` converts a null value to ``NaN``: the null values are restored below.
    result = pl.Series(
        series.name, round_numpy(series.to_numpy(), mode, decimals), dtype=series.dtype
    )
    if series.null_count():
        return result.zip_with(series.is_not_null(), series)
    return result


def get_default_rounders() -> dict[type, Rounder]:
    r"""Get the default rounders of the polars Series and DataFrames.

    Returns:
        The rounder of the polars Series and DataFrames and their
            associated types.

    Notes:
        This function is called by the default registry the first time
        it needs the rounders of polars.
    """
    if not is_polars_available():
        return {}
    return {pl.Series: round_polars, pl.DataFrame: round_polars}
