r"""Implement the rounding of the NumPy arrays and scalars."""

from __future__ import annotations

__all__ = ["round_numpy"]

from typing import Any

from coola.rounding.common import MAX_EXACT, RoundMode, get_scale
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


def round_numpy(data: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round a NumPy array or scalar.

    Integer and boolean values are not changed. The computation is made
    in ``float64`` and the result has the dtype of ``data``. The mask
    of a masked array is kept.

    Args:
        data: The ``numpy.ndarray`` or NumPy scalar to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded array or scalar. It never shares memory with
            ``data``.

    Raises:
        TypeError: If ``data`` is not an array or a scalar of NumPy, or
            if its dtype is not an integer, boolean or floating-point
            type.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.rounding.numpy import round_numpy
        >>> round_numpy(np.array([1.234, 5.678]), "ceil", 1)
        array([1.3, 5.7])

        ```
    """
    if isinstance(data, np.generic):
        return _round_array(np.asarray(data), mode, decimals)[()]
    if not isinstance(data, np.ndarray):
        msg = f"Unsupported type: {type(data).__qualname__}"
        raise TypeError(msg)
    if isinstance(data, np.ma.MaskedArray):
        result = _round_array(np.asarray(data.data), mode, decimals)
        return np.ma.array(result, mask=np.ma.getmask(data), fill_value=data.fill_value)
    return _round_array(data, mode, decimals)


def _round_array(data: np.ndarray, mode: RoundMode, decimals: int) -> np.ndarray:
    kind = data.dtype.kind
    if kind in "iub":
        return data.copy()
    if kind != "f":
        msg = f"Unsupported dtype: {data.dtype}"
        raise TypeError(msg)
    scale = get_scale(decimals)
    if scale is None:
        return data.copy()
    values = data.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = values * scale
        if mode == "round":
            result = np.rint(scaled) / scale
        elif mode == "floor":
            index = np.floor(scaled)
            # ``0.29 * 100`` is below 29: take the grid point if it is not above the value. The grid
            # point is compared in the dtype of the array, as the value is.
            index = np.where(_grid(index + 1, scale, data.dtype) <= values, index + 1, index)
            result = index / scale
        else:
            index = np.ceil(scaled)
            index = np.where(_grid(index - 1, scale, data.dtype) >= values, index - 1, index)
            result = index / scale
        result = np.where(np.abs(scaled) < MAX_EXACT, result, values)
    return result.astype(data.dtype)


def _grid(index: np.ndarray, scale: float, dtype: np.dtype) -> np.ndarray:
    return (index / scale).astype(dtype).astype(np.float64)
