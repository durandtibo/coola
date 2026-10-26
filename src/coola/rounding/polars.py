r"""Implement the rounding of the polars Series and DataFrames."""

from __future__ import annotations

__all__ = ["round_polars"]

from typing import Any

from coola.rounding.common import MAX_EXACT, RoundMode, get_scale
from coola.utils.imports import is_polars_available

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl


def round_polars(data: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round a polars Series or DataFrame.

    Only the floating-point columns are rounded. The integer columns
    are not changed, and neither are the columns that are not numerical
    (strings, booleans, dates, decimals, ...). The null values are
    kept. The computation is made in ``Float64``.

    Args:
        data: The ``polars.Series`` or ``polars.DataFrame`` to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded Series or DataFrame.

    Raises:
        TypeError: If ``data`` is not a Series or a DataFrame, or if
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
    if isinstance(data, pl.DataFrame):
        return data.with_columns(
            _round_series(series, mode, decimals) for series in data if series.dtype.is_float()
        )
    if isinstance(data, pl.Series):
        if not (data.dtype.is_float() or data.dtype.is_integer()):
            msg = f"Unsupported dtype: {data.dtype}"
            raise TypeError(msg)
        return _round_series(data, mode, decimals)
    msg = f"Unsupported type: {type(data).__qualname__}"
    raise TypeError(msg)


def _round_series(series: pl.Series, mode: RoundMode, decimals: int) -> pl.Series:
    scale = get_scale(decimals)
    if not series.dtype.is_float() or scale is None:
        return series.clone()
    # The computation uses an expression: ``Series / scalar`` multiplies by the reciprocal of
    # the scalar, which changes the last digit (``12 / 10`` is ``1.2000000000000002``).
    value = pl.col("value")
    scaled = value * scale
    if mode == "round":
        result = scaled.round(0, mode="half_to_even") / scale
    else:
        step = 1 if mode == "floor" else -1
        index = scaled.floor() if mode == "floor" else scaled.ceil()
        # ``0.29 * 100`` is below 29: take the grid point if it is not above the value. The grid
        # point is compared in the dtype of the series, as the value is.
        grid = ((index + step) / scale).cast(series.dtype).cast(pl.Float64)
        in_grid = grid <= value if mode == "floor" else grid >= value
        result = pl.when(in_grid).then(index + step).otherwise(index) / scale
    # A null value stays null, and ``NaN`` and the infinite values are kept.
    rounded = pl.when(scaled.abs() < MAX_EXACT).then(result).otherwise(value).cast(series.dtype)
    frame = series.cast(pl.Float64).alias("value").to_frame()
    return frame.select(rounded.alias(series.name)).to_series()
