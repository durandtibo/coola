r"""Define the helpers shared by the backends of the rounding
functions."""

from __future__ import annotations

__all__ = ["MAX_EXACT", "MODES", "RoundMode", "get_scale", "validate_arguments"]

from typing import Literal, get_args

from coola.validation import validate_ge, validate_in, validate_int

RoundMode = Literal["round", "floor", "ceil"]

MODES: tuple[str, ...] = get_args(RoundMode)

# Above this magnitude, a scaled float64 is an integer and the decimal grid is finer than the
# spacing of the floats, so the value is already on the grid.
MAX_EXACT = 2.0**53


def validate_arguments(mode: str, decimals: int) -> None:
    r"""Validate the arguments of ``round_values``.

    Args:
        mode: The rounding mode.
        decimals: The number of decimals.

    Raises:
        ValueError: If ``mode`` is not one of ``"round"``, ``"floor"``
            and ``"ceil"``, or if ``decimals`` is negative.
        TypeError: If ``decimals`` is not an ``int``.

    Example:
        ```pycon
        >>> from coola.rounding.common import validate_arguments
        >>> validate_arguments("floor", 2)

        ```
    """
    validate_in(mode, MODES, name="mode")
    validate_int(decimals, name="decimals")
    validate_ge(decimals, 0, name="decimals")


def get_scale(decimals: int) -> float | None:
    r"""Return the factor ``10**decimals`` as a float.

    Args:
        decimals: The number of decimals.

    Returns:
        The factor, or ``None`` if it is too large for a float. In
            that case, the decimal grid is finer than any float and the
            values are not changed.

    Example:
        ```pycon
        >>> from coola.rounding.common import get_scale
        >>> get_scale(2)
        100.0
        >>> get_scale(400) is None
        True

        ```
    """
    try:
        return 10.0**decimals
    except OverflowError:
        return None
