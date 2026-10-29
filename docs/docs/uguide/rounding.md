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

## Properties

`round_values` rounds, floors or ceils the numerical values of an object with one interface, and
the answer is the same whatever the library that holds the numbers. These are the rules of the
package.

| Property                 | Statement                                                                                                                                                                                                                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| One decimal rule         | A float is read as the decimal that `repr` shows (the shortest string that round-trips in its dtype). `"floor"`, `"ceil"` and `"round"` (halfway cases to the even digit) act on that decimal, and the result is the float of the dtype that is nearest to the grid point. So `0.29` floored to 2 decimals is `0.29`. |
| Exact                    | The result equals `Decimal(repr(x)).quantize(10**-decimals, mode)` while `abs(x) * 10**decimals < 1e13`. Above, the result is within one grid step of it. From `2**53` the value is already on the grid and is returned as it is.                                                                                     |
| Mode contract            | `floor(x) <= x <= ceil(x)`, `abs(round(x) - x)` is at most half a step, the result is a grid point, rounding is idempotent and monotone (`x <= y` implies `f(x) <= f(y)`).                                                                                                                                            |
| Same everywhere          | A value gives the same result, bit for bit, in the native scalar, a NumPy array, a torch tensor and a pandas, polars or pyarrow column of the same dtype. The backends differ in structure, never in arithmetic.                                                                                                      |
| Values preserved         | Integers and booleans are never changed. A float stays a float of the same type (`round_values(2.5)` is `2.0`). `NaN`, the infinite values and the missing values are kept. A result that is zero has the sign of the input (`ceil(-0.4)` is `-0.0`, like `numpy.ceil`).                                              |
| Type-preserving          | The result has the type, dtype, shape, device, index, columns, names and mask of the input; a subclass keeps its type (`MaskedArray` with its mask and fill value, `np.matrix`, a `Series` subclass). Two exceptions: a subclass of `float` becomes a `float`, and a `Parameter` becomes a `Tensor` without grad.     |
| Tables are heterogeneous | Only the floating-point columns of a pandas, polars or pyarrow table are rounded. The other columns (integers, strings, booleans, dates, decimals, nested lists and structs) are left as they are. A Series, an array or a tensor must be numerical.                                                                  |
| Strict contract          | An invalid `mode`, `decimals` or `on_missing_package` raises before any work. An unsupported type or dtype raises `UnsupportedTypeError` (a `TypeError`) that names the type and how to register a rounder. With `recursive=True` a leaf that raises it is kept; any other exception propagates unchanged.            |
| Pure                     | The input is never modified. The result of a mutable container (array, tensor, pandas object) owns its memory; the immutable Arrow buffers of polars and pyarrow can be shared. No state, no lock: a registry and a rounder can be shared by threads.                                                                 |
| Registry-extensible      | A new type is supported by registering one function `(data, mode, decimals) -> result`, without changing another one. The most specific registered type along the MRO wins. A plugin group registers rounders at import.                                                                                              |
| Lazy                     | `import coola.rounding` imports no optional library. The rounders of a library are loaded the first time a value of that library is rounded.                                                                                                                                                                          |
| Consistent with siblings | Options are keyword-only (`mode`, `decimals`, `recursive`, `on_missing_package`, `registry`). With `recursive=True` the structure is walked by `recursive_apply` and dataclasses are entered like in `equality`, `hashing` and `summary`.                                                                             |

## Modes

| Mode      | Behavior                                                              |
| --------- | --------------------------------------------------------------------- |
| `"round"` | Nearest value. The halfway cases of the decimal go to the even digit. |
| `"floor"` | Towards negative infinity.                                            |
| `"ceil"`  | Towards positive infinity.                                            |

`decimals` is the number of decimals to keep. It is `0` by default and must be an integer (a NumPy
integer is accepted, a `bool` is not) that is not negative. `mode`, `decimals`, `recursive`,
`on_missing_package` and `registry` are keyword-only.

## The rule

The three modes act on the decimal that `repr` shows, and the result is the float of the dtype that
is nearest to the grid point. The native `float`, NumPy, PyTorch, pandas, polars and pyarrow give
the same result, bit for bit.

```pycon
>>> from coola.rounding import round_values
>>> round_values(0.29, mode="floor", decimals=2)
0.29
>>> round_values(2.675, decimals=2)
2.68
>>> round_values(-0.4, mode="ceil")
-0.0

```

This is **not** the rule of the Python `round` function, which rounds the exact binary value (so
`round(2.675, 2)` is `2.67`), nor of `numpy.round`, which scales the value and rounds it. With the
rule of `round_values`, `0.29` floored to 2 decimals is `0.29` and not `0.28`, although `0.29 * 100`
is slightly below `29` in floating-point arithmetic.

### Exactness

The result equals `Decimal(repr(x)).quantize(...)` while `abs(x) * 10**decimals < 1e13`. Past this
bound the result is still a grid point within one step of the exact one, so the guarantee stops at
about 13 significant digits. A user who needs more digits registers a rounder for `Decimal` (see
below). In `float16` and `bfloat16` the result is the nearest value the dtype can hold
(`bfloat16(1.2345)` rounded to 2 decimals is `1.2266`), and rounding it again gives the same value.

## Behavior by type

| Type                                                                                         | Result                                                                                                            |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `int`, `bool`                                                                                | Returned as they are.                                                                                             |
| `float`                                                                                      | A new `float` (a subclass becomes a `float`).                                                                     |
| NumPy array, scalar                                                                          | Same type and dtype; `matrix`, `ndarray` subclasses and `MaskedArray` keep type, mask and fill value.             |
| PyTorch tensor                                                                               | Same dtype and device. A `Parameter` becomes a `Tensor` without grad. A sparse or nested tensor is not supported. |
| pandas Series, DataFrame                                                                     | Float columns are rounded; index, names and duplicate column names are kept.                                      |
| polars Series, DataFrame                                                                     | Float columns are rounded (`Float32` is kept); nulls are kept. Needs NumPy.                                       |
| pyarrow Array, ChunkedArray, Table, RecordBatch                                              | Float columns are rounded (the type and the chunks are kept); nulls are kept. Needs NumPy.                        |
| Integer, string, boolean, date columns                                                       | Left as they are.                                                                                                 |
| Nested columns (`list<float>`, struct)                                                       | Left as they are: the values inside the cells are not rounded.                                                    |
| `pandas.Index`, `polars.LazyFrame`, `Decimal`, complex numbers, dates, times, sparse tensors | `UnsupportedTypeError` (register a rounder to support them).                                                      |

The date and time types and the complex numbers are not numerical in the sense of the package.

## Nested data

With `recursive=True`, `round_values` rounds every leaf of a nested structure, using
[`coola.recursive`](recursive.md). The leaves that are not supported (strings, `None`, dates, ...)
are kept as they are. Only an `UnsupportedTypeError` is caught: any other exception of a rounder
propagates, so a bug is not hidden. Dataclasses are entered.

```pycon
>>> from coola.rounding import round_values
>>> round_values({"a": [1.234, "x"], "b": (5.678, 3)}, decimals=1, recursive=True)
{'a': [1.2, 'x'], 'b': (5.7, 3)}

```

## Errors

An object with no rounder raises `UnsupportedTypeError`, a subclass of `TypeError`:

```pycon
>>> from coola.rounding import UnsupportedTypeError, round_values
>>> try:
...     round_values("abc")
... except UnsupportedTypeError as error:
...     print(error)
Cannot round an object of type str; register a rounder with register_rounders

```

## Missing packages

Rounding a polars or a pyarrow object needs NumPy (the values go through the NumPy kernel, which is
what makes the results the same everywhere). `on_missing_package` says what happens when a rounder
needs a package that is not installed:

| Value               | Behavior                                                                               |
| ------------------- | -------------------------------------------------------------------------------------- |
| `"raise"` (default) | `UnsupportedTypeError` naming the missing package, `MissingPackageError` as the cause. |
| `"warn"`            | `UserWarning` naming the type and the package; the object is returned unchanged.       |
| `"ignore"`          | The object is returned unchanged, silently.                                            |

The option is about a missing package, not about every unsupported type: an object with no rounder
(a `str`, a `Decimal`) always raises `UnsupportedTypeError`, which `recursive=True` keeps. With
`recursive=True` the option applies to each leaf, so `"raise"` stops the walk at the first leaf
whose package is missing. `"warn"` and `"ignore"` can hide a column that was not rounded, so they
are opt-in.

## Extending

A rounder is a plain function `(data, mode, decimals) -> result`. It receives a validated `mode` and
`decimals`, returns a **new** object and raises `UnsupportedTypeError` for a type or a dtype that it
does not support. Register it with `register_rounders`; the rounder of the most specific type along
the MRO is used.

```pycon
>>> from coola.rounding import register_rounders, round_values
>>> class Meters:
...     def __init__(self, value):
...         self.value = value
>>> def round_meters(data, mode, decimals):
...     return Meters(round_values(data.value, mode=mode, decimals=decimals))
>>> register_rounders({Meters: round_meters})
>>> round_values(Meters(2.567), decimals=1).value
2.6

```

`registry=` of `round_values` uses another `TypeRegistry` instead of the default one
(`get_default_registry()`). A package can register its rounders at import through the plugin group
`coola.rounding.rounders`: the entry point is a callable that receives the registry. A rounder that
needs an optional package can raise `MissingPackageError` and `on_missing_package` is applied to it.
See [How to Add Support for a New Library](../dev/add_backend.md) for a library.

## What `round_values` does not do

- **`pandas.Index`, `polars.LazyFrame` and Arrow scalars** raise `UnsupportedTypeError`. Call
  `collect()` on a `LazyFrame` first.
- **`Decimal`, `Fraction`, complex numbers, dates, times and durations** are not numerical in the
  sense of the package. `Decimal` is the typical type to register a rounder for.
- **JAX, xarray and CuPy** have no default rounder. A rounder is a function of about 20 lines on the
  pattern of the one for PyTorch: register it, or ship it in a plugin.
- **Sparse, nested and quantized tensors** raise `UnsupportedTypeError`.
- **Rounding to significant digits**, and the `"trunc"` and `"half_up"` modes: the modes are
  `"floor"`, `"ceil"` and `"round"`.
- **In-place rounding** (`out=`): the input is never modified, the package is pure.
