r"""Implement the rounding of the native Python numbers."""

from __future__ import annotations

__all__ = ["get_default_rounders", "round_float", "round_int"]

import math
from typing import Any

from coola.rounding.common import MAX_EXACT, Rounder, RoundMode, UnsupportedTypeError, get_scale


def round_int(data: Any, mode: RoundMode, decimals: int) -> int:
    r"""Round a native Python ``int``.

    Args:
        data: The ``int`` (or ``bool``) to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        ``data``, as an integer is never changed.

    Raises:
        UnsupportedTypeError: If ``data`` is not an ``int``.

    Example:
        ```pycon
        >>> from coola.rounding.native import round_int
        >>> round_int(3, "floor", 2)
        3

        ```
    """
    del mode, decimals  # an integer is never changed
    if not isinstance(data, int):
        msg = f"Unsupported type: {type(data).__qualname__}"
        raise UnsupportedTypeError(msg)
    return data


def round_float(data: Any, mode: RoundMode, decimals: int) -> float:
    r"""Round a native Python ``float``.

    The same formula as the NumPy kernel is used, so the result is the
    same bit for bit. A subclass of ``float`` is returned as a
    ``float``.

    Args:
        data: The ``float`` to round.
        mode: The rounding mode.
        decimals: The number of decimals.

    Returns:
        The rounded ``float``.

    Raises:
        UnsupportedTypeError: If ``data`` is not a ``float``.

    Example:
        ```pycon
        >>> from coola.rounding.native import round_float
        >>> round_float(2.567, "floor", 2)
        2.56

        ```
    """
    if not isinstance(data, float):
        msg = f"Unsupported type: {type(data).__qualname__}"
        raise UnsupportedTypeError(msg)
    data = float(data)
    scale = get_scale(decimals)
    if scale is None:
        return data
    scaled = data * scale
    if not abs(scaled) < MAX_EXACT:  # also true for NaN and infinite values
        return data
    index = math.floor(scaled)
    # ``scaled`` is rounded, so the index is wrong by at most one in either direction.
    if (index + 1) / scale <= data:
        index += 1
    if index / scale > data:
        index -= 1
    return math.copysign(_move_index(index, data, scale, mode) / scale, data)


def _move_index(index: int, value: float, scale: float, mode: RoundMode) -> int:
    r"""Move the index of the grid point that is not above ``value`` to
    the index that the mode selects.

    Args:
        index: The index of the grid point that is not above ``value``.
        value: The value.
        scale: The scale of the grid: ``10 ** decimals``.
        mode: The rounding mode.

    Returns:
        The index selected by the mode.
    """
    if mode == "ceil":
        return index + (index / scale < value)
    if mode == "round":
        middle = (2 * index + 1) / (2 * scale)
        if value > middle:
            return index + 1
        if value == middle:
            return index + (index & 1)  # a halfway case goes to the even index
    return index


def get_default_rounders() -> dict[type, Rounder]:
    r"""Get the default rounders of the native numbers.

    Returns:
        The rounders of ``int`` (and so ``bool``) and ``float``.
    """
    return {int: round_int, float: round_float}
