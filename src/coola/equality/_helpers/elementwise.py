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

    The arrays are compared in double precision: ``float16``,
    ``bfloat16`` and ``float32`` are cast to ``float64`` (and
    ``complex64`` to ``complex128``), so the tolerance cannot overflow
    at the edge of the range of the dtype. The difference of two
    integers is computed exactly in integer arithmetic (a difference
    that wraps around is corrected), and only the comparison to the
    tolerance is made in ``float64``. This comparison can round the
    other way than an exact comparison when the difference is above
    2^53 and equal to the tolerance (a measure-zero case). The arrays
    must have the same shape and dtype. A boolean has no subtraction:
    the caller compares it exactly. If ``jax`` does not enable 64-bit
    values, ``float64`` is ``float32`` for JAX.

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
        >>> from coola.equality._helpers.elementwise import elementwise_close
        >>> x, y = np.array([100.0, 101.005]), np.array([101.005, 100.0])
        >>> elementwise_close(x, y, atol=0.0, rtol=0.01, equal_nan=False, xp=np)
        array([ True,  True])
        >>> x, y = np.array([2**62]), np.array([2**62 + 1])
        >>> elementwise_close(x, y, atol=0.5, rtol=0.0, equal_nan=False, xp=np)
        array([False])

        ```
    """
    if _is_integer(array1, xp):
        return _integer_close(array1, array2, atol=atol, rtol=rtol, xp=xp)
    array1, array2 = _to_double(array1, xp), _to_double(array2, xp)
    equal = array1 == array2
    finite = xp.isfinite(array1) & xp.isfinite(array2)
    tolerance = atol + rtol * xp.maximum(xp.abs(array1), xp.abs(array2))
    close = equal | (finite & (xp.abs(array1 - array2) <= tolerance))
    if equal_nan:
        close = close | (xp.isnan(array1) & xp.isnan(array2))
    return close


def _is_torch(xp: Any) -> bool:
    r"""Indicate if the array module is PyTorch.

    Args:
        xp: The array module (NumPy, PyTorch or JAX).

    Returns:
        ``True`` if ``xp`` is ``torch``, otherwise ``False``.
    """
    return xp.__name__ == "torch"


def _is_integer(array: Any, xp: Any) -> bool:
    r"""Indicate if the dtype of an array is an integer type.

    Args:
        array: The array to check.
        xp: The array module of ``array``.

    Returns:
        ``True`` if the dtype is an integer (not floating-point or
            complex), otherwise ``False``.
    """
    if _is_torch(xp):
        return not (array.is_floating_point() or array.is_complex())
    return array.dtype.kind in {"i", "u"}


def _itemsize(array: Any, xp: Any) -> int:
    r"""Return the size in bytes of one element of an array.

    Args:
        array: The array.
        xp: The array module of ``array``.

    Returns:
        The number of bytes of one element.
    """
    return array.element_size() if _is_torch(xp) else array.dtype.itemsize


def _cast(array: Any, name: str, xp: Any) -> Any:
    r"""Cast ``array`` to the dtype ``name`` of the module ``xp``.

    Args:
        array: The array to cast.
        name: The name of the target dtype, e.g. ``"float64"``.
        xp: The array module of ``array``.

    Returns:
        The cast array.
    """
    dtype = getattr(xp, name)
    if _is_torch(xp):
        return array.to(dtype)
    if xp.__name__.startswith("jax"):
        # Without 64-bit mode, JAX has no ``float64``: use the widest dtype it has.
        import jax  # noqa: PLC0415

        dtype = jax.dtypes.canonicalize_dtype(dtype)
    return array.astype(dtype)


def _to_double(array: Any, xp: Any) -> Any:
    r"""Cast the floating-point and complex arrays to double precision.

    Args:
        array: The array to cast.
        xp: The array module of ``array``.

    Returns:
        The array in double precision (``float64`` or ``complex128``),
            or ``array`` itself if it is already at least that wide.
    """
    is_complex = array.is_complex() if _is_torch(xp) else array.dtype.kind == "c"
    if is_complex:
        return array if _itemsize(array, xp) >= 16 else _cast(array, "complex128", xp)
    return array if _itemsize(array, xp) >= 8 else _cast(array, "float64", xp)


def _integer_close(array1: Any, array2: Any, *, atol: float, rtol: float, xp: Any) -> Any:
    r"""Indicate for each element if two integer arrays are close.

    The difference is exact: the integers narrower than 64 bits are
    widened to ``int64``, so the difference cannot wrap around; the
    difference of two ``int64`` can wrap, and then the modulus is added
    back. The unsigned 64-bit integers do not wrap.

    Args:
        array1: The first integer array.
        array2: The second integer array, with the same dtype and shape.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        xp: The array module of the arrays.

    Returns:
        A boolean array that is ``True`` where the elements are close.
    """
    unsigned_64 = not _is_torch(xp) and array1.dtype.kind == "u"
    if _itemsize(array1, xp) < 8:
        array1, array2 = _cast(array1, "int64", xp), _cast(array2, "int64", xp)
    delta = xp.maximum(array1, array2) - xp.minimum(array1, array2)
    difference = _cast(delta, "float64", xp)
    if not unsigned_64:
        # ``delta`` is in [0, 2**bits): a negative value is a wrapped one.
        difference = difference + (delta < 0) * 2.0 ** (8 * _itemsize(delta, xp))
    scale = xp.maximum(xp.abs(_cast(array1, "float64", xp)), xp.abs(_cast(array2, "float64", xp)))
    return (array1 == array2) | (difference <= atol + rtol * scale)
