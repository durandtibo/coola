from __future__ import annotations

__all__ = ["quantile"]

import math
from collections.abc import Sequence


def quantile(values: Sequence[float | int], quantiles: Sequence[float]) -> list[float]:
    r"""Computes the quantiles with the linear method.

    Args:
    ----
        values (sequence of ints or floats): Specifies the values.
        quantiles (sequence of float): Specifies the quantile
            values in the range ``[0, 1]``.

    Returns:
    -------
        list of floats: The quantiles.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.stats import quantile
        >>> quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0.2, 0.5, 0.9))
        [2.0, 5.0, 9.0]
    """
    values = sorted(values)
    n = len(values)
    output = []
    for q in quantiles:
        virtual_index = max(0.0, min(n - 1.0, q * (n - 1.0)))
        index = math.floor(virtual_index)
        g = virtual_index - index
        output.append((1.0 - g) * values[index] + g * values[max(0, min(n - 1, index + 1))])
    return output
