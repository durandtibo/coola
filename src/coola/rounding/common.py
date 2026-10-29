r"""Define the helpers shared by the backends of the rounding
functions."""

from __future__ import annotations

__all__ = [
    "MAX_EXACT",
    "MISSING_PACKAGE_POLICIES",
    "MODES",
    "MissingPackagePolicy",
    "RoundMode",
    "Rounder",
    "UnsupportedTypeError",
    "get_scale",
    "validate_arguments",
]

import numbers
from collections.abc import Callable
from typing import Any, Literal, get_args

from coola.validation import validate_ge, validate_in, validate_int

RoundMode = Literal["round", "floor", "ceil"]

MODES: tuple[str, ...] = get_args(RoundMode)

MissingPackagePolicy = Literal["raise", "warn", "ignore"]

MISSING_PACKAGE_POLICIES: tuple[str, ...] = get_args(MissingPackagePolicy)

#: A function ``(data, mode, decimals) -> result`` that rounds the objects of one type. ``mode``
#: and ``decimals`` are validated by ``round_values``. It returns a new object and raises
#: ``UnsupportedTypeError`` for a type or a dtype that it does not support.
Rounder = Callable[[Any, RoundMode, int], Any]

# Above this magnitude, a scaled float64 is an integer and the decimal grid is finer than the
# spacing of the floats, so the value is already on the grid.
MAX_EXACT = 2.0**53


class UnsupportedTypeError(TypeError):
    r"""Raised when an object, or its dtype, cannot be rounded.

    Example:
        ```pycon
        >>> from coola.rounding import UnsupportedTypeError, round_values
        >>> try:
        ...     round_values("abc")
        ... except UnsupportedTypeError as error:
        ...     print(type(error).__name__)
        ...
        UnsupportedTypeError

        ```
    """


def validate_arguments(mode: str, decimals: int, on_missing_package: str = "raise") -> int:
    r"""Validate the arguments of ``round_values``.

    Args:
        mode: The rounding mode.
        decimals: The number of decimals. Any integer type is
            accepted (``int``, NumPy integers, ...), but not a ``bool``.
        on_missing_package: What to do when a rounder needs a package
            that is not installed.

    Returns:
        The number of decimals as a Python ``int``.

    Raises:
        ValueError: If ``mode`` is not one of ``"round"``, ``"floor"``
            and ``"ceil"``, if ``decimals`` is negative, or if
            ``on_missing_package`` is not one of ``"raise"``,
            ``"warn"`` and ``"ignore"``.
        TypeError: If ``decimals`` is not an integer.

    Example:
        ```pycon
        >>> from coola.rounding.common import validate_arguments
        >>> validate_arguments("floor", 2)
        2

        ```
    """
    validate_in(mode, MODES, name="mode")
    if isinstance(decimals, numbers.Integral) and not isinstance(decimals, bool):
        decimals = int(decimals)
    validate_int(decimals, name="decimals")
    validate_ge(decimals, 0, name="decimals")
    validate_in(on_missing_package, MISSING_PACKAGE_POLICIES, name="on_missing_package")
    return decimals


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
