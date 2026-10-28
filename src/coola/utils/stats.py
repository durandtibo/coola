r"""Implement some utility functions to compute statistics."""

from __future__ import annotations

__all__ = ["quantile"]

import math
from typing import TYPE_CHECKING

from coola.validation import validate_range

if TYPE_CHECKING:
    from collections.abc import Sequence


def quantile(values: Sequence[float | int], quantiles: Sequence[float]) -> list[float]:
    r"""Compute the quantiles with the linear method.

    Args:
        values: The values.
        quantiles: The quantile values in the
            range ``[0, 1]``.

    Returns:
        The quantiles.

    Raises:
        ValueError: If a quantile is not in the range ``[0, 1]``, or if
            ``values`` is empty and ``quantiles`` is not.

    Notes:
        If ``values`` contains a ``NaN``, every quantile is ``NaN``, like
        with NumPy: the sort order of the ``NaN`` is undefined, so the
        quantiles would depend on the position of the ``NaN``.

    Example:
        ```pycon
        >>> from coola.utils.stats import quantile
        >>> quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0.2, 0.5, 0.9))
        [2.0, 5.0, 9.0]
        >>> quantile([1.0, float("nan")], (0.5,))
        [nan]

        ```
    """
    for q in quantiles:
        validate_range(q, 0.0, 1.0, name="quantiles")
    if len(values) == 0 and len(quantiles) > 0:
        msg = "values must not be empty when quantiles is not empty"
        raise ValueError(msg)
    if any(math.isnan(value) for value in values):
        return [math.nan] * len(quantiles)
    values = sorted(values)
    n = len(values)
    output: list[float] = []
    for q in quantiles:
        virtual_index = max(0.0, min(n - 1.0, q * (n - 1.0)))
        index = math.floor(virtual_index)
        g = virtual_index - index
        if g == 0.0:
            # Exact value: the interpolation would compute ``0 * inf = nan`` next to an infinite value.
            output.append(float(values[index]))
        else:
            output.append((1.0 - g) * values[index] + g * values[max(0, min(n - 1, index + 1))])
    return output
