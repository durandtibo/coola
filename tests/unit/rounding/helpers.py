r"""Define the oracle shared by the tests of ``coola.rounding``.

The oracle is the specification of the package: a float is read as the
decimal that ``repr`` shows, and this decimal is quantized with the
mode of the rounding. It is independent of the implementation, which
computes the same result with float arithmetic.
"""

from __future__ import annotations

__all__ = ["EXACT_BOUND", "HARD_VALUES", "oracle", "oracle_many"]

import math
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    localcontext,
)

# The result equals the oracle while ``abs(x) * 10**decimals`` is below this bound.
EXACT_BOUND = 1e13

_ROUNDING = {"round": ROUND_HALF_EVEN, "floor": ROUND_FLOOR, "ceil": ROUND_CEILING}

# Ties, values one ulp around a grid point, the smallest subnormal and a value past the grid.
HARD_VALUES = (
    2.675,
    0.125,
    8.995,
    -9.995,
    1.005,
    0.29,
    0.28999999999999998,
    0.29000000000000004,
    0.7,
    0.6999999999999999,
    0.7000000000000001,
    5.52,
    5.5200000000000005,
    3.3,
    3.3000000000000003,
    5e-324,
    1e15 + 0.5,
)


def oracle(value: float, mode: str, decimals: int, *, dtype: type = float) -> float:
    r"""Round a float with ``Decimal``.

    Args:
        value: The value to round.
        mode: The rounding mode.
        decimals: The number of decimals.
        dtype: The type of ``value`` (``float`` or a NumPy float type)
            whose ``repr`` gives the decimal.

    Returns:
        The grid point nearest to the quantized decimal, with the sign
            of ``value``. ``NaN``, infinite values and values on the
            grid past ``2**53`` are returned as they are.
    """
    if not math.isfinite(value) or abs(value) * 10.0**decimals >= 2.0**53:
        return value
    with localcontext(Context(prec=800, Emax=999999, Emin=-999999)):
        decimal = Decimal(repr(dtype(value)))
        result = decimal.quantize(Decimal(1).scaleb(-decimals), rounding=_ROUNDING[mode])
        return math.copysign(float(dtype(str(result))), value)


def oracle_many(values: list[float], mode: str, decimals: int, *, dtype: type = float) -> list:
    r"""Round a list of floats with ``Decimal``."""
    return [oracle(value, mode, decimals, dtype=dtype) for value in values]
