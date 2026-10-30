r"""Implement the rounding of the pyarrow arrays and tables."""

from __future__ import annotations

__all__ = ["get_default_rounders", "round_pyarrow"]

from typing import TYPE_CHECKING, Any

from coola.rounding.common import UnsupportedTypeError
from coola.rounding.numpy import round_numpy
from coola.utils.imports import check_numpy, is_pyarrow_available

if TYPE_CHECKING:
    from coola.rounding.common import Rounder, RoundMode

if is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa


def round_pyarrow(data: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round a pyarrow Table, RecordBatch, Array or ChunkedArray.

    Only the floating-point columns of a table or a record
    batch are rounded. The integer
    columns are not changed, and neither are the columns that are not
    numerical (strings, booleans, dates, decimals, ...). The null
    values are kept. The values go through the NumPy kernel of
    ``coola.rounding.numpy``, so NumPy is required.

    Args:
        data: The ``pyarrow.Table``, ``pyarrow.RecordBatch``,
            ``pyarrow.Array`` or ``pyarrow.ChunkedArray`` to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded table, record batch, array or chunked array.

    Raises:
        MissingPackageError: If ``numpy`` is not installed.
        UnsupportedTypeError: If ``data`` is not a table, a record batch, an
            array or a chunked array, or if ``data`` is an array that
            is not numerical.

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
    check_numpy()
    if isinstance(data, (pa.Table, pa.RecordBatch)):
        columns = [_round_array(column, mode, decimals) for column in data.columns]
        return type(data).from_arrays(columns, schema=data.schema)
    if isinstance(data, (pa.Array, pa.ChunkedArray)):
        if not (pa.types.is_floating(data.type) or pa.types.is_integer(data.type)):
            msg = f"Unsupported type: {data.type}"
            raise UnsupportedTypeError(msg)
        return _round_array(data, mode, decimals)
    msg = f"Unsupported type: {type(data).__qualname__}"
    raise UnsupportedTypeError(msg)


def _round_array(array: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round an array.

    Args:
        array: The array or chunked array.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded array. An array that is not a floating-point one is
            returned as it is, and the null values are kept.
    """
    if not pa.types.is_floating(array.type):
        return array
    if isinstance(array, pa.ChunkedArray):
        chunks = [_round_array(chunk, mode, decimals) for chunk in array.chunks]
        return pa.chunked_array(chunks, type=array.type)
    # ``to_numpy`` converts a null value to ``NaN``: the null values are restored with the mask.
    values = round_numpy(array.to_numpy(zero_copy_only=False), mode, decimals)
    mask = array.is_null().to_numpy(zero_copy_only=False) if array.null_count else None
    return pa.array(values, type=array.type, mask=mask)


def get_default_rounders() -> dict[type, Rounder]:
    r"""Get the default rounders of the pyarrow arrays and tables.

    Returns:
        The rounder of the pyarrow arrays, chunked arrays, tables and
            record batches and their associated types.

    Notes:
        This function is called by the default registry the first time
        it needs the rounders of pyarrow.
    """
    if not is_pyarrow_available():
        return {}
    return {
        pa.Array: round_pyarrow,
        pa.ChunkedArray: round_pyarrow,
        pa.Table: round_pyarrow,
        pa.RecordBatch: round_pyarrow,
    }
