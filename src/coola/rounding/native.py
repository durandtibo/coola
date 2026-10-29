r"""Implement the rounding of the native Python numbers."""

from __future__ import annotations

__all__ = ["round_native"]

import math
from typing import Any

from coola.rounding.common import MAX_EXACT, RoundMode, get_scale


def round_native(data: Any, mode: RoundMode, decimals: int) -> int | float:
    r"""Round a native Python number.

    Args:
        data: The ``int`` or ``float`` to round. A ``bool`` is not
            changed.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded number. An ``int`` is returned as it is, and a
            ``float`` is rounded to a ``float``.

    Raises:
        TypeError: If ``data`` is not an ``int`` or a ``float``.

    Example:
        ```pycon
        >>> from coola.rounding.native import round_native
        >>> round_native(2.567, "floor", 2)
        2.56

        ```
    """
    if isinstance(data, int):
        return data
    if isinstance(data, float):
        return _round_float(data, mode, decimals)
    msg = f"Unsupported type: {type(data).__qualname__}"
    raise TypeError(msg)


def _round_float(value: float, mode: RoundMode, decimals: int) -> float:
    scale = get_scale(decimals)
    if scale is None:
        return value
    scaled = value * scale
    if not abs(scaled) < MAX_EXACT:  # also true for NaN and infinite values
        return value
    index = math.floor(scaled)
    # ``scaled`` is rounded, so the index is wrong by at most one in either direction.
    if (index + 1) / scale <= value:
        index += 1
    if index / scale > value:
        index -= 1
    if mode == "ceil":
        if index / scale < value:
            index += 1
    elif mode == "round":
        middle = (2 * index + 1) / (2 * scale)
        if value > middle:
            index += 1
        elif value == middle:
            index += index & 1  # a halfway case goes to the even index
    return math.copysign(index / scale, value)
