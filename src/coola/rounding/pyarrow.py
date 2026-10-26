r"""Implement the rounding of the pyarrow arrays and tables."""

from __future__ import annotations

__all__ = ["round_pyarrow"]

from typing import Any

from coola.rounding.common import MAX_EXACT, RoundMode, get_scale
from coola.utils.imports import is_pyarrow_available

if is_pyarrow_available():
    import pyarrow as pa
    import pyarrow.compute as pc
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa, pyarrow as pc


def round_pyarrow(data: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round a pyarrow Table, Array or ChunkedArray.

    Only the floating-point columns of a table are rounded. The integer
    columns are not changed, and neither are the columns that are not
    numerical (strings, booleans, dates, decimals, ...). The null
    values are kept. The computation is made in ``float64``.

    Args:
        data: The ``pyarrow.Table``, ``pyarrow.Array`` or
            ``pyarrow.ChunkedArray`` to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded table, array or chunked array.

    Raises:
        TypeError: If ``data`` is not a table, an array or a chunked
            array, or if ``data`` is an array that is not numerical.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.rounding.pyarrow import round_pyarrow
        >>> round_pyarrow(pa.array([1.234, 5.678]), "round", 1)
        <pyarrow.lib.DoubleArray object at 0x...>
        [
          1.2,
          5.7
        ]

        ```
    """
    if isinstance(data, pa.Table):
        result = data
        for position, field in enumerate(data.schema):
            if pa.types.is_floating(field.type):
                column = _round_array(data.column(position), mode, decimals)
                result = result.set_column(position, field, column)
        return result
    if isinstance(data, (pa.Array, pa.ChunkedArray)):
        if not (pa.types.is_floating(data.type) or pa.types.is_integer(data.type)):
            msg = f"Unsupported type: {data.type}"
            raise TypeError(msg)
        return _round_array(data, mode, decimals)
    msg = f"Unsupported type: {type(data).__qualname__}"
    raise TypeError(msg)


def _round_array(array: Any, mode: RoundMode, decimals: int) -> Any:
    scale = get_scale(decimals)
    if not pa.types.is_floating(array.type) or scale is None:
        return array
    values = array.cast(pa.float64())
    scaled = pc.multiply(values, scale)
    if mode == "round":
        result = pc.divide(pc.round(scaled, 0, "half_to_even"), scale)
    else:
        step = 1 if mode == "floor" else -1
        index = pc.floor(scaled) if mode == "floor" else pc.ceil(scaled)
        moved = pc.add(index, step)
        # ``0.29 * 100`` is below 29: take the grid point if it is not above the value. The grid
        # point is compared in the type of the array, as the value is.
        grid = pc.divide(moved, scale).cast(array.type).cast(pa.float64())
        in_grid = pc.less_equal(grid, values) if mode == "floor" else pc.greater_equal(grid, values)
        result = pc.divide(pc.if_else(in_grid, moved, index), scale)
    # A null value stays null, and ``NaN`` and the infinite values are kept.
    in_range = pc.less(pc.abs(scaled), MAX_EXACT)
    return pc.if_else(in_range, result, values).cast(array.type)
