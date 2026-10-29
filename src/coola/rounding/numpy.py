r"""Implement the rounding of the NumPy arrays and scalars."""

from __future__ import annotations

__all__ = ["get_default_rounders", "round_numpy"]

from typing import Any

from coola.rounding.common import MAX_EXACT, Rounder, RoundMode, UnsupportedTypeError, get_scale
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np


def round_numpy(data: Any, mode: RoundMode, decimals: int) -> Any:
    r"""Round a NumPy array or scalar.

    Integer and boolean values are not changed. The computation is made
    in ``float64`` and the result has the type and the dtype of
    ``data`` (a ``matrix``, a subclass of ``ndarray`` and a masked
    array with its mask and fill value are kept).

    Args:
        data: The ``numpy.ndarray`` or NumPy scalar to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded array or scalar. It never shares memory with
            ``data``.

    Raises:
        UnsupportedTypeError: If ``data`` is not an array or a scalar of NumPy, or
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
        raise UnsupportedTypeError(msg)
    return _round_array(data, mode, decimals)


def _round_array(data: np.ndarray, mode: RoundMode, decimals: int) -> np.ndarray:
    kind = data.dtype.kind
    if kind in "iub":
        return data.copy()
    if kind != "f":
        msg = f"Unsupported dtype: {data.dtype}"
        raise UnsupportedTypeError(msg)
    # ``copy`` keeps the type of the array (``matrix``, subclasses, mask and fill value of a
    # ``MaskedArray``): the rounded values are assigned to the copy.
    result = data.copy()
    scale = get_scale(decimals)
    if scale is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            np.copyto(result, _round_float_array(np.asarray(data), mode, scale))
    return result


def _round_float_array(data: np.ndarray, mode: RoundMode, scale: float) -> np.ndarray:
    # The computation is made in float64. The grid points are compared with the value after a
    # cast to the dtype of the array, as the value is.
    x = data.astype(np.float64)
    low = data.dtype != np.float64

    def grid(index: np.ndarray) -> np.ndarray:
        point = index / scale
        return point.astype(data.dtype).astype(np.float64) if low else point

    scaled = x * scale
    index = np.floor(scaled)
    # ``scaled`` is rounded, so the index is wrong by at most one in either direction.
    index = index + (grid(index + 1) <= x)
    index = index - (grid(index) > x)
    if mode == "ceil":
        index = index + (grid(index) < x)
    elif mode == "round":
        middle = (2 * index + 1) / (2 * scale)
        if low:
            middle = middle.astype(data.dtype).astype(np.float64)
        # ``rint`` sends a halfway case to the even index.
        index = np.where(x > middle, index + 1, np.where(x == middle, np.rint(index + 0.5), index))
    # ``NaN``, the infinite values and the values past the grid are not changed.
    result = np.where(np.abs(scaled) < MAX_EXACT, index / scale, x)
    return np.copysign(result, x).astype(data.dtype)


def get_default_rounders() -> dict[type, Rounder]:
    r"""Get the default rounders of the NumPy arrays and scalars.

    Returns:
        The rounder of the NumPy arrays and scalars and their
            associated types.

    Notes:
        This function is called by the default registry the first time
        it needs the rounders of NumPy.
    """
    if not is_numpy_available():
        return {}
    return {np.ndarray: round_numpy, np.generic: round_numpy}
