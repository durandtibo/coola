# Rounding Values

:book: This page describes the `coola.rounding` package, which provides `round_values`, a function
that rounds (or floors, or ceils) numerical values of many types with one interface.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

`round_values` works on:

- native `int` and `float`
- NumPy arrays and scalars
- PyTorch tensors
- pandas Series and DataFrames
- polars Series and DataFrames
- pyarrow Arrays, ChunkedArrays, Tables and RecordBatches

The libraries are imported only when a value of that library is rounded.

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import torch
>>> from coola.rounding import round_values
>>> round_values(2.567, decimals=1)
2.6
>>> round_values(np.array([1.234, 5.678]), mode="floor", decimals=2)
array([1.23, 5.67])
>>> round_values(torch.tensor([1.234, 5.678]), mode="ceil", decimals=1)
tensor([1.3000, 5.7000])
>>> round_values(pd.DataFrame({"a": [1.234, 5.678], "name": ["x", "y"]}), decimals=1)
     a name
0  1.2    x
1  5.7    y

```

## Modes

| Mode      | Behavior                                                                    |
| --------- | --------------------------------------------------------------------------- |
| `"round"` | Nearest value. The halfway cases of the decimal go to the even digit.      |
| `"floor"` | Towards negative infinity.                                                  |
| `"ceil"`  | Towards positive infinity.                                                  |

`decimals` is the number of decimals to keep. It is `0` by default and must not be negative.
`mode` and `decimals` are keyword-only.

## Behavior

- **Integers are never changed**, and a float stays a float of the same type: `round_values(2.5)`
  is `2.0`, not `2`.
- **`NaN`, infinite and missing values are kept.**
- **The dtype is kept.** The computation is made in `float64`, and the result is cast back to the dtype of the data.
- **The input is not modified**, and the result never shares memory with it.
- **Tables are heterogeneous**: only the floating-point columns of a pandas, polars or pyarrow
  table are rounded. The other columns (integers, strings, booleans, dates, ...) are left as they
  are. A Series, an array or a tensor must be numerical, otherwise a `TypeError` is raised.
- **polars and pyarrow need NumPy.** Their floating-point columns are rounded by the NumPy kernel,
  so the results are the same as for NumPy. Without NumPy, a `MissingPackageError` is raised.
- **Not supported**: date and time types, complex numbers, `Decimal` and `Fraction`.

### One decimal rule

A float is read as the decimal that `repr` shows (the shortest string that round-trips in its
dtype). The three modes act on that decimal, and the result is the float of the dtype that is
nearest to the grid point. The native `float`, NumPy, PyTorch and pandas give the same result, bit
for bit.

```pycon
>>> from coola.rounding import round_values
>>> round_values(0.29, mode="floor", decimals=2)
0.29
>>> round_values(2.675, decimals=2)
2.68
>>> round_values(-0.4, mode="ceil")
-0.0

```

The result equals `Decimal(repr(x)).quantize(...)` while `abs(x) * 10**decimals < 1e13`. Above, it
is within one grid step of it. This is **not** the rule of the Python `round` function (which rounds
the exact binary value, so `round(2.675, 2)` is `2.67`) nor of `numpy.round` (scale and `rint`).
A value that is zero after the rounding keeps the sign of the input, like `numpy.ceil`. `decimals`
can be any integer, a NumPy integer included. A NumPy `matrix`, a subclass of `ndarray` and a masked
array keep their type (and the mask).

## Nested data

With `recursive=True`, `round_values` rounds every leaf of a nested structure, using
[`coola.recursive`](recursive.md). The leaves that are not supported (strings, `None`, dates, ...)
are kept as they are.

```pycon
>>> from coola.rounding import round_values
>>> round_values({"a": [1.234, "x"], "b": (5.678, 3)}, decimals=1, recursive=True)
{'a': [1.2, 'x'], 'b': (5.7, 3)}

```
