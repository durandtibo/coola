r"""Implement the elementwise rule that decides if two arrays are
close."""

from __future__ import annotations

__all__ = ["elementwise_close"]

from typing import Any


def elementwise_close(
    array1: Any, array2: Any, *, atol: float, rtol: float, equal_nan: bool, xp: Any
) -> Any:
    r"""Indicate for each element if two arrays are close.

    The elements are close if ``|a - b| <= atol + rtol * max(|a|, |b|)``,
    which is the rule used for the Python numbers. The rule is applied
    to each pair of elements, so it is symmetric: the result is the same
    if the arrays are swapped, whatever the elements. Two equal elements
    are always close (infinities included), two different elements are
    never close if one of them is infinite or NaN, and two NaNs are
    close only if ``equal_nan`` is ``True``. The computation of ``inf -
    inf`` makes NumPy emit a ``RuntimeWarning``: the NumPy caller
    silences it with ``numpy.errstate``.

    The arrays must have the same shape and a floating-point or complex
    dtype (the caller casts the other dtypes, because an unsigned
    subtraction wraps around and a boolean has no subtraction).

    Args:
        array1: The first array (NumPy, PyTorch or JAX).
        array2: The second array, with the same type, shape and dtype.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        equal_nan: If ``True``, two NaNs are close.
        xp: The module that provides ``abs``, ``maximum``, ``isfinite``
            and ``isnan`` for the arrays (``numpy``, ``torch`` or
            ``jax.numpy``).

    Returns:
        A boolean array with the shape of the inputs.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.handler.elementwise import elementwise_close
        >>> x, y = np.array([100.0, 101.005]), np.array([101.005, 100.0])
        >>> elementwise_close(x, y, atol=0.0, rtol=0.01, equal_nan=False, xp=np)
        array([ True,  True])

        ```
    """
    equal = array1 == array2
    finite = xp.isfinite(array1) & xp.isfinite(array2)
    tolerance = atol + rtol * xp.maximum(xp.abs(array1), xp.abs(array2))
    close = equal | (finite & (xp.abs(array1 - array2) <= tolerance))
    if equal_nan:
        close = close | (xp.isnan(array1) & xp.isnan(array2))
    return close
