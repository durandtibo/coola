r"""Define the public interface to round numerical values."""

from __future__ import annotations

__all__ = ["round_values"]

import pkgutil
from typing import Any

from coola.recursive.interface import recursive_apply
from coola.rounding.common import RoundMode, validate_arguments

# (name of the top-level module of the types, function that rounds them)
_BACKENDS = {
    "builtins": "coola.rounding.native:round_native",
    "numpy": "coola.rounding.numpy:round_numpy",
    "pandas": "coola.rounding.pandas:round_pandas",
    "polars": "coola.rounding.polars:round_polars",
    "pyarrow": "coola.rounding.pyarrow:round_pyarrow",
    "torch": "coola.rounding.torch:round_torch",
}


def round_values(
    data: Any, *, mode: RoundMode = "round", decimals: int = 0, recursive: bool = False
) -> Any:
    r"""Round the numerical values of an object.

    The supported objects are:

    - the native ``int`` and ``float``;
    - the NumPy arrays and scalars;
    - the PyTorch tensors;
    - the pandas Series and DataFrames;
    - the polars Series and DataFrames;
    - the pyarrow Arrays, ChunkedArrays and Tables.

    The integer values are never changed, whatever the mode and the
    number of decimals. A floating-point value stays a floating-point
    value of the same type (``round_values(2.5)`` is ``2.0``, not
    ``2``). ``NaN``, the infinite values and the missing values are
    kept. The input is not modified and the result never shares
    memory with it.

    The tables (pandas, polars and pyarrow) are heterogeneous: only
    their floating-point columns are rounded and the other columns
    (integers, strings, booleans, dates, ...) are left as they are. A
    Series, an array or a tensor must be numerical. The date and time
    types, the complex numbers and the other objects (``Decimal``,
    ``Fraction``, containers, ...) are not supported.

    Args:
        data: The object to round.
        mode: The rounding mode. ``"round"`` rounds to the nearest
            value and the halfway cases to the even value (like the
            Python ``round`` function), ``"floor"`` rounds towards
            negative infinity and ``"ceil"`` rounds towards positive
            infinity.
        decimals: The number of decimals to keep. It must be greater
            than or equal to ``0``.
        recursive: If ``True``, ``data`` can be a nested structure
            (``list``, ``tuple``, ``dict``, ``set``, dataclass, ...,
            as handled by ``coola.recursive.recursive_apply``) and
            every leaf is rounded. A leaf that is not supported (a
            string, ``None``, a date, ...) is kept as it is instead of
            raising an error. Defaults to ``False``.

    Returns:
        The object with the rounded values, with the same type, dtype
            and shape as ``data``.

    Raises:
        ValueError: If ``mode`` is not ``"round"``, ``"floor"`` or
            ``"ceil"``, or if ``decimals`` is negative.
        TypeError: If ``decimals`` is not an ``int``, or if ``data`` is
            not supported (with ``recursive=False``).

    Notes:
        ``"floor"`` and ``"ceil"`` act on the decimal value of the
        float: ``round_values(0.29, mode="floor", decimals=2)`` is
        ``0.29`` and not ``0.28``, even if ``0.29 * 100`` is slightly
        below ``29`` in floating-point arithmetic.

        ``"round"`` uses the function of each library, so the halfway
        cases of a float that cannot be represented exactly can differ
        between the libraries. Python rounds the exact binary value, so
        ``2.675`` is rounded to ``2.67``, while the other libraries
        scale the value, round it and scale it back.

        The computation is made in ``float64`` and the result is cast
        back to the dtype of ``data``.

    Example:
        ```pycon
        >>> from coola.rounding import round_values
        >>> round_values(2.567, decimals=1)
        2.6
        >>> round_values(2.567, mode="floor", decimals=2)
        2.56
        >>> round_values(2.567, mode="ceil")
        3.0
        >>> round_values({"a": [1.234, "x"], "b": (5.678, 3)}, decimals=1, recursive=True)
        {'a': [1.2, 'x'], 'b': (5.7, 3)}
        >>> import numpy as np
        >>> round_values(np.array([1.5, 2.5, -0.5]))
        array([ 2.,  2., -0.])
        >>> import torch
        >>> round_values(torch.tensor([1.234, 5.678]), mode="ceil", decimals=1)
        tensor([1.3000, 5.7000])
        >>> import pandas as pd
        >>> round_values(pd.DataFrame({"a": [1.234, 5.678], "b": [1, 2]}), decimals=2)
              a  b
        0  1.23  1
        1  5.68  2

        ```
    """
    validate_arguments(mode, decimals)
    if recursive:
        return recursive_apply(data, lambda x: _round_leaf(x, mode, decimals))
    return _round_object(data, mode, decimals)


def _round_leaf(data: Any, mode: RoundMode, decimals: int) -> Any:
    try:
        return _round_object(data, mode, decimals)
    except TypeError:
        return data


def _round_object(data: Any, mode: RoundMode, decimals: int) -> Any:
    # The native function is the fallback: it raises the error for the unsupported types.
    modules = (cls.__module__.split(".")[0] for cls in type(data).__mro__)
    target = _BACKENDS[next((name for name in modules if name in _BACKENDS), "builtins")]
    return pkgutil.resolve_name(target)(data, mode, decimals)
