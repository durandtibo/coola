# `coola.rounding`: target state and the changes to reach it

This document describes the design that `coola.rounding` should have, then the changes to make to
the current implementation (`main`, `c8a6f570`) to get there. The design follows the rules of the
user guide (`docs/docs/uguide/rounding.md`, which has no **Properties** table yet: section 1 is the
table to add) and the architecture guide (`docs/docs/dev/architecture.md`: registry dispatch, MRO
lookup, lazy imports, strict contracts, "coola owns the semantics, the libraries own the
structure"), and it stays aligned with the sibling packages `equality`, `hashing`, `recursive` and
`reducer`. Its aim is a small package that is easy to maintain: **one rounding rule, one kernel per
array library, one adapter per table library, one extension point**. Breaking changes are accepted,
and the future state is more important than the past one.

Evidence comes from probes and throw-away prototypes run on `main` (Python 3.14, NumPy 2.5.4, torch
2.14.1, pandas 3.0.6, polars 2.0.0, pyarrow 26.0.0; scripts kept outside the repository). Items
marked **[verified]** were reproduced; **[reading]** comes from the code. Nothing in `src`, `tests`
or `docs` was changed.

The headline finding: the package is young (one commit, #2364) and well documented, but **its
results are not correct and not the same across backends**. polars returns values that are one ulp
off for more than a third of a random column, `floor` and `ceil` return a value on the wrong side
of the input near a grid point, and `round` has three different definitions (section 2). The 239
unit tests pass with 100 % coverage because they use a handful of values and a tolerant comparison.
The design below fixes this **by construction** (one rule, checked against an independent oracle)
instead of by more per-backend patches.

## Constraints

1. **Simple and easy to maintain.** The package is 704 lines for one function. Five backends
   (`native`, `numpy`, `torch`, `polars`, `pyarrow`) each carry their own copy of the same formula
   with their own fix-ups, which is why they disagree. The target keeps **two** kernels (NumPy and
   torch, like the numeric kernel of `equality`), a scalar version of the same formula for Python
   floats, and makes the three table libraries thin adapters over the NumPy kernel.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no
   adapter, no alias, no deprecation period. Results change (ties, values next to a grid point,
   polars), the release notes list it.
3. **Correct first, then fast.** A rule that is exact and costs a few nanoseconds per element more
   is preferred to a fast rule that is wrong on 1 % of the inputs. The gate of section 8 bounds the
   cost (`floor`/`ceil` unchanged, `round` at most 5 times slower on a large array, a scalar and the
   recursive path faster).
4. **Consistent with the siblings.** One extension point (a registry, `register_*`, a plugin group),
   keyword-only options, errors raised before any work with a message that names the remedy,
   dataclasses entered with `recursive=True`, stateless and thread-safe.
5. **Optional libraries stay lazy.** `import coola.rounding` imports no optional library (verified:
   `numpy`, `torch`, `pandas`, `polars` and `pyarrow` are all absent from `sys.modules` after it) and
   `coola` still has zero hard dependencies. New: rounding a polars or a pyarrow object needs NumPy
   (decision D3); without it `on_missing_package` decides: raise (default), warn or ignore (D9).

## 1. Properties of the sub-package

`round_values` rounds, floors or ceils the numerical values of an object with one interface, and
the answer is the same whatever the library that holds the numbers. These are the rules of the
package. Every design choice below exists to make one of them true **by construction** (in the
kernel and in the interface) instead of by the discipline of each backend.

| Property                 | Statement                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Enforced by                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| One decimal rule         | A float is read as the decimal that `repr` shows (the shortest string that round-trips in its dtype). `"floor"`, `"ceil"` and `"round"` (halfway cases to the even digit) act on that decimal, and the result is the float of the dtype that is nearest to the grid point. So `0.29` floored to 2 decimals is `0.29`.                                                                                                                                               | the kernels                               |
| Exact                    | The result equals `Decimal(repr(x)).quantize(10**-decimals, mode)` while `abs(x) * 10**decimals < 1e13`. Above, the result is within one grid step of it. From `2**53` the value is already on the grid and is returned as it is.                                                                                                                                                                                                                                   | the kernels, the oracle test              |
| Mode contract            | `floor(x) <= x <= ceil(x)`, `abs(round(x) - x)` is at most half a step, the result is a grid point, rounding is idempotent (`round_values(round_values(x)) == round_values(x)`) and monotone (`x <= y` implies `f(x) <= f(y)`).                                                                                                                                                                                                                                     | the kernels, property tests               |
| Same everywhere          | A value gives the same result, bit for bit, in the native scalar, a NumPy array, a torch tensor and a pandas, polars or pyarrow column of the same dtype. The backends differ in structure, never in arithmetic.                                                                                                                                                                                                                                                    | one formula, the conformance test         |
| Values preserved         | Integers and booleans are never changed. A float stays a float of the same type (`round_values(2.5)` is `2.0`). `NaN`, the infinite values and the missing values are kept. A result that is zero has the sign of the input (`ceil(-0.4)` is `-0.0`, like `numpy.ceil`).                                                                                                                                                                                            | the kernels                               |
| Type-preserving          | The result has the type, dtype, shape, device, index, columns, names and mask of the input; a subclass keeps its type (`MaskedArray` with its mask and fill value, `np.matrix`, a `Series` subclass). Two exceptions, documented: a subclass of `float` becomes a `float`, and a `Parameter` becomes a `Tensor` without grad.                                                                                                                                       | `copy` then assignment; one test per type |
| Tables are heterogeneous | Only the floating-point columns of a pandas, polars or pyarrow table are rounded. The other columns (integers, strings, booleans, dates, decimals, nested lists and structs) are left as they are. A Series, an array or a tensor must be numerical.                                                                                                                                                                                                                | the table adapters                        |
| Strict contract          | An invalid `mode` or `decimals` raises before any work. An unsupported type or dtype raises `UnsupportedTypeError` (a `TypeError`) that names the type and how to register a rounder. With `recursive=True` a leaf that raises it is kept; any other exception propagates unchanged. A rounder whose package is missing (NumPy for polars and pyarrow) follows `on_missing_package`: raise (default), warn and return the object unchanged, or return it unchanged. | `validate_arguments`, one exception class |
| Pure                     | The input is never modified. The result of a mutable container (array, tensor, pandas object) owns its memory; the immutable Arrow buffers of polars and pyarrow can be shared. No state, no lock, no thread-local: a registry and a rounder can be shared by threads.                                                                                                                                                                                              | stateless rounders                        |
| Registry-extensible      | A new type is supported by registering one function `(data, mode, decimals) -> result`, without changing another one. The most specific registered type along the MRO wins. A plugin group registers rounders at import.                                                                                                                                                                                                                                            | `TypeRegistry`, `register_rounders`       |
| Lazy                     | `import coola.rounding` imports no optional library. The rounders of a library are loaded the first time a value of that library is rounded.                                                                                                                                                                                                                                                                                                                        | `register_lazy`                           |
| Consistent with siblings | Options are keyword-only (`mode`, `decimals`, `recursive`, `on_missing_package`, `registry`). With `recursive=True` the structure is walked by `recursive_apply` and dataclasses are entered like in `equality`, `hashing` and `summary`.                                                                                                                                                                                                                           | the interface                             |
| Fast                     | Arrays and columns are processed by vectorised operations, with no Python loop per element; a Python float costs well under a microsecond; the dispatch of the recursive path is a cached lookup.                                                                                                                                                                                                                                                                   | the kernels, the registry                 |

Not guaranteed, and documented as such:

- **Agreement with `round()` of Python and `numpy.round`.** They round the exact binary value
  (Python) or scale and `rint` (NumPy), so the halfway cases of a decimal literal differ:
  `round(2.675, 2)` is `2.67`, `round_values(2.675, decimals=2)` is `2.68`. `round_values(0.29,
mode="floor", decimals=2)` is `0.29`, not `0.28`.
- **Exactness at 14 or more significant digits.** The bound of the "Exact" row; the result is still
  on the grid and within one step.
- **Decimal results in `float16` and `bfloat16`.** The result is the nearest value the dtype can
  hold (`bfloat16(1.2345)` rounded to 2 decimals is `1.2266`); it is idempotent.
- **Anything but the listed types.** `pandas.Index`, `polars.LazyFrame`, `Decimal`, complex numbers,
  dates and times, sparse and nested tensors raise `UnsupportedTypeError` (section 10).
- **Polars and pyarrow without NumPy.** They cannot be rounded; `on_missing_package` says what
  happens (D9).

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Status         |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **polars returns wrong values.** On 20,000 random values with 3 decimals, `round_values(pl.Series(x), decimals=1)` differs from the NumPy result for 6,968 values (`-4.822` gives `-4.800000000000001`) and 2,610 values at 2 decimals; `floor` and `ceil` the same. Cause: polars divides a column by a scalar as a multiplication by the reciprocal, `pl.Series([-48.0, 12.0]) / 10.0` is `[-4.800000000000001, 1.2000000000000002]`, also column by column. The comment in `_round_series` knows the problem and works around it with an expression, which only works for one row (constant folding). pyarrow, torch and pandas agree with NumPy (0 differences). Resolved by rounding polars through NumPy (D3). | **[verified]** |
| F2  | **`floor` and `ceil` return a value on the wrong side of the input.** The fix-up looks at one neighbour of the grid point only. `ceil(5.5200000000000005, 2)` is `5.52` (smaller than the input) and `ceil(3.3000000000000003, 1)` is `3.3`; 2,857 of 40,000 values one ulp away from a grid point break `floor(x) <= x <= ceil(x)`. Against the decimal oracle the NumPy kernel is wrong on about 1.5 % of such values (66 to 426 of 24,000 for the numbers of decimals tried, 1 to 8).                                                                                                                                                                                                                             | **[verified]** |
| F3  | **`round` has three definitions.** The native scalar uses Python's `round` (the exact binary value), the arrays scale and `rint` (inexact), and `floor`/`ceil` use the decimal value. `round_values(6.55, decimals=1)` is `6.5` for a float and `6.6` for a NumPy array; 66 of 20,000 values differ at 1 decimal and 830 at 2 decimals (`0.225` gives `0.23` and `0.22`). Against the decimal oracle, the scalar is wrong for 66 and 962 values and NumPy, torch and pyarrow for 0 and 133 (`8.995` gives `8.99`, the decimal is a tie and goes to `9.0`). The guide presents it as a library quirk.                                                                                                                 | **[verified]** |
| F4  | **No extension point.** The dispatch is a hard-coded dict `_BACKENDS` keyed by the top-level module of the classes in the MRO, with a string that is imported (`import_object`) for every value. `Decimal`, `Fraction`, JAX, xarray or a user type cannot be added without changing `coola`, although every sibling is registry-based (`TypeRegistry`, `register_lazy`, a plugin group) and the guides promise "Registry-extensible".                                                                                                                                                                                                                                                                                | **[reading]**  |
| F5  | **`recursive=True` swallows every `TypeError`.** `_round_leaf` returns the leaf unchanged on any `TypeError`, including one raised inside a library by a bug, and a leaf that fails with another error (a sparse tensor raises `NotImplementedError: Could not run 'aten::lt.Scalar' ...`) escapes in a different way.                                                                                                                                                                                                                                                                                                                                                                                               | **[verified]** |
| F6  | **One formula, five copies.** `native`, `numpy`, `torch`, `polars` and `pyarrow` each repeat the scale, the `MAX_EXACT` guard, the grid check in the dtype and the special values. F1 to F3 are consequences: a fix has to be made five times and nothing compares the five. The only cross-backend test checks NumPy against torch on six values.                                                                                                                                                                                                                                                                                                                                                                   | **[reading]**  |
| F7  | **The type is not preserved**, although the docstring promises "the same type, dtype and shape". `np.matrix` and any `ndarray` subclass become `ndarray`, a `float` subclass becomes `float`, a `Parameter` becomes a `Tensor`. A masked array has its own branch (6 lines). `data.copy()` followed by an assignment keeps a `matrix`, a subclass and a `MaskedArray` with its mask and fill value.                                                                                                                                                                                                                                                                                                                  | **[verified]** |
| F8  | **`decimals` rejects NumPy integers.** `round_values(1.26, decimals=np.int64(1))` raises `TypeError: decimals must be an int, got int64`, although `decimals` often comes from a config or an array.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | **[verified]** |
| F9  | **A common Arrow type is missing.** `pa.RecordBatch` raises `Unsupported type` while `pa.Table` is supported, because the table branch uses `set_column`, which a record batch does not have. `from_arrays(columns, schema=...)` builds both **[verified]**.                                                                                                                                                                                                                                                                                                                                                                                                                                                         | **[verified]** |
| F10 | **The sign of zero is not consistent.** `round_values(-0.4, mode="ceil")` is `0.0` for a float (`math.ceil` returns an `int`) and `-0.0` for a NumPy array, a torch tensor and the `Decimal` quantisation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | **[verified]** |
| F11 | **The tests cannot see F1 to F3.** 239 tests pass with 100 % line and branch coverage. The polars and pandas tests compare with `assert_series_equal`, whose default is a relative tolerance of `1e-5`, on inputs of two to four values; there is no oracle, no property test, no large input, no thread test, and no benchmark. The only integration test is the dataclass one.                                                                                                                                                                                                                                                                                                                                     | **[verified]** |
| F12 | **Per-value cost of the recursive path.** Each leaf goes through `import_object` and a scan of the MRO by module name: 1.4 µs per float, 141 ms for a list of 100,000 floats with `recursive=True`, five times the cost of the builtin `round` (28 ms).                                                                                                                                                                                                                                                                                                                                                                                                                                                              | **[verified]** |
| F13 | **Documentation.** The guide has no **Properties** table (`recursive.md` and `hashing.md` have one), states that Python rounds the exact binary value as if it were a library detail, and does not say what happens to a subclass or to a nested column. `dev/add_backend.md` has no row for `coola.rounding`.                                                                                                                                                                                                                                                                                                                                                                                                       | **[reading]**  |

What is already right and must not regress: nothing optional is imported with `coola.rounding`;
options are keyword-only; `mode` and `decimals` are validated before any work; the input is never
modified; tables are heterogeneous; masks, `NaN`, infinite and missing values are kept; the grid is
compared in the dtype of the data (float32 results are float32 values); the `MAX_EXACT` guard and
`get_scale` for a huge `decimals`; `recursive=True` enters dataclasses (#2377); 100 % coverage.

## 3. Architecture

```
round_values(data, *, mode="round", decimals=0, recursive=False, on_missing_package="raise", registry=None)
        |   validate mode, decimals and on_missing_package once; one registry (default: the singleton)
        |   recursive ? recursive_apply(data, leaf) : leaf(data)
        v
leaf(x): rounder = registry.resolve(type(x))                  MRO, lazy libraries
        |   rounder(x, mode, decimals)  ->  result | raises UnsupportedTypeError
        v
rounders (coola/rounding/*.py)
        |-- native.py    float, int (and bool)          the scalar version of the formula
        |-- numpy.py     ndarray, generic               THE kernel (about 20 lines)
        |-- torch.py     Tensor                         the same formula, torch operations
        |-- pandas.py    Series, DataFrame  \
        |-- polars.py    Series, DataFrame   >  float columns -> round_numpy, nulls restored
        '-- pyarrow.py   Array, ChunkedArray, Table, RecordBatch  /
```

The interface owns the invariants (validation, dispatch, the recursive walk, the error contract). A
rounder **only describes one type**: it receives a validated `mode` and `decimals` and returns a new
object or raises `UnsupportedTypeError`. This is the split of `equality` ("coola owns the semantics,
the libraries own the structure"): every number goes through the kernel; a library says which
columns are floating point and gives the values back, never what `2.675` rounds to.

### 3.1 The rule

For a finite `x` and `step = 10**-decimals`, with `repr(x)` the shortest string that round-trips in
the dtype of `x`:

- `"floor"`: the largest grid point that is `<= Decimal(repr(x))`; `"ceil"`: the smallest that is
  `>=`; `"round"`: the nearest, a halfway case goes to the grid point whose last digit is even.
- The result is that grid point converted to the dtype (the nearest float).
- Not changed: integers, booleans, `NaN`, infinities, missing values, and `abs(x) * 10**decimals >=
2**53` (the value is already on the grid, and `10**decimals` that overflows a float has no effect).
- A zero result takes the sign of `x`.

The grid point is found with float arithmetic and two checks, which is what makes the rule exact
without `Decimal`. With `s = x * scale` and `k = floor(s)` the index is wrong by at most one in
either direction, because `x * scale` is rounded; comparing the grid points **in the dtype of `x`**
repairs it in both directions (the current code repairs one direction only, F2). For `"round"`
the midpoint `(2k + 1) / (2 * scale)` is computed as a float of the dtype: `x == mid` is a halfway
case if and only if `repr(x)` is the midpoint, as long as the grid is coarser than the spacing of the
floats (the bound of the "Exact" row).

```python
def round_numpy(data, mode, decimals):  # the float case of the kernel
    scale = get_scale(decimals)  # 10.0**decimals, None if it overflows
    x = data.astype(np.float64, copy=False)
    low = data.dtype != np.float64

    def grid(k):  # a grid point as the dtype holds it, compared in float64
        g = k / scale
        return g.astype(data.dtype).astype(np.float64) if low else g

    s = x * scale
    k = np.floor(s)
    k += grid(k + 1) <= x  # s was rounded down: the next grid point is not above x
    k -= grid(k) > x  # s was rounded up: this grid point is above x
    if mode == "ceil":
        k += grid(k) < x  # x is strictly between two grid points
    elif mode == "round":
        mid = (2 * k + 1) / (2 * scale)
        if low:
            mid = mid.astype(data.dtype).astype(np.float64)
        k = np.where(x > mid, k + 1, np.where(x == mid, np.rint(k + 0.5), k))  # rint: even
    result = np.where(np.abs(s) < MAX_EXACT, k / scale, x)  # NaN, inf, huge: unchanged
    return np.copysign(result, x).astype(data.dtype, copy=False)
```

The code above is the prototype of section 6 without its `errstate` and the integer/boolean
branch; the real function wraps it as today (copy for integers and booleans, `UnsupportedTypeError`
for the other dtypes, `np.errstate(over="ignore", invalid="ignore")`).

### 3.2 The scalar, the tensor, the tables

- **`round_float` (native).** The same formula on one `float`, with `math.floor`, integer `k` and
  `math.copysign`: about 15 lines, 0.30 µs per call, bit-identical to the NumPy kernel **[verified
  on 100,010 values, 3 modes, 5 values of `decimals`, signs of zero included]**. `int` (and `bool`) are returned
  as they are. The `Decimal` implementation is the specification and lives in the tests only (D2).
- **`round_torch`.** The same formula with `torch.floor`, `torch.where`, `torch.round` (which is
  half-to-even), the dtype of the tensor for the grid, `float32` instead of `float64` on `mps` (as
  today), `detach().clone()` for the integers. The result is bit-identical to the NumPy kernel for
  `float64`, `float32` and `float16` **[verified]**. A non-strided layout (sparse, nested) raises
  `UnsupportedTypeError` instead of the raw `NotImplementedError` of torch (F5).
- **Type preservation (F7).** NumPy: `out = data.copy()` then `np.copyto(out, rounded)`. It keeps
  a `matrix`, a subclass and a `MaskedArray` with its mask and fill value **[verified]**, so the
  masked branch goes away. `np.generic` goes through a 0-d array and `[()]`, as today.
- **Tables.** The three adapters share one shape: find the floating-point columns, round each with
  `round_numpy` on its NumPy values, restore the nulls, put the column back with its name and dtype.
  - pandas: as today (`isetitem` keeps duplicate column names; the `pandas` nullable and Arrow dtypes
    already go through `to_numpy(na_value=nan)`); a Series of integers is returned as a copy.
  - polars: `pl.Series(name, round_numpy(s.to_numpy(), ...), dtype=s.dtype)` then `zip_with(
s.is_not_null(), s)` when there are nulls. 20 lines instead of 40, no expression, no
    `Series / scalar` (F1). **[verified]** equal to the kernel on 1,000,000 values, in three modes,
    with nulls and `NaN`, `Float32` kept.
  - pyarrow: `pa.array(round_numpy(a.to_numpy(zero_copy_only=False), ...), type=a.type, mask=...)`
    per column and `type(data).from_arrays(columns, schema=data.schema)` for `Table` and
    `RecordBatch` (F9). **[verified]** equal to the kernel, `float32` kept, nulls kept.
  - A column that is not floating point is returned as it is (an integer column is not copied for
    polars and pyarrow: their buffers are immutable).
  - NumPy is imported when the adapter runs, not when it is registered. If it is missing the adapter
    raises `MissingPackageError` before any work, and `round_values` applies `on_missing_package`
    (section 3.4, D9).

### 3.3 The rounder contract and the registry

```python
Rounder = Callable[[Any, RoundMode, int], Any]


class UnsupportedTypeError(TypeError):
    """The object, or its dtype, cannot be rounded."""
```

A rounder is a plain function: no base class, no protocol, no state (D4). It receives a `mode` and a
`decimals` that `round_values` validated, returns a **new** object (never `data` itself unless
`data` is immutable and unchanged), and raises `UnsupportedTypeError` for a type or a dtype it does
not support. The default registry is a `TypeRegistry[Rounder]` built by
`make_default_registry_singleton`, like the registries of the siblings:

| Registered for                                                         | Rounder                               | Loaded                           |
| ---------------------------------------------------------------------- | ------------------------------------- | -------------------------------- |
| `float`, `int` (covers `bool` and `IntEnum`)                           | `round_float`, `round_int` (identity) | at import                        |
| `np.ndarray`, `np.generic` (before `float` in the MRO of `np.float64`) | `round_numpy`                         | `register_lazy(("numpy",), ...)` |
| `torch.Tensor`                                                         | `round_torch`                         | lazily                           |
| `pd.Series`, `pd.DataFrame`                                            | `round_pandas`                        | lazily                           |
| `pl.Series`, `pl.DataFrame`                                            | `round_polars`                        | lazily                           |
| `pa.Array`, `pa.ChunkedArray`, `pa.Table`, `pa.RecordBatch`            | `round_pyarrow`                       | lazily                           |

Each library module has a `get_default_rounders()` returning its `{type: rounder}` mapping (and `{}`
when the library is missing), and `interface.py` has a `_LAZY_ROUNDERS` table, as `hashing` does. The rounders of polars and pyarrow are registered when their library is installed
and import NumPy when they run (D9). The
plugin group is `coola.rounding.rounders` (a callable that receives the registry). A type with no
rounder makes `resolve` raise `TypeNotRegisteredError`; `round_values` turns it into
`UnsupportedTypeError("Cannot round an object of type X; register a rounder with register_rounders")`,
with the registry error as the cause.

The example of the guide, a complete extension in six lines:

```python
from decimal import Decimal
from coola.rounding import register_rounders

register_rounders(
    {Decimal: lambda x, mode, decimals: x.quantize(Decimal(1).scaleb(-decimals), ...)}
)
```

### 3.4 The interface

```python
def round_values(
    data, *, mode="round", decimals=0, recursive=False, on_missing_package="raise", registry=None
):
    validate_arguments(mode, decimals, on_missing_package)
    registry = get_default_registry() if registry is None else registry
    if recursive:
        return recursive_apply(
            data, lambda x: _round_leaf(x, registry, mode, decimals, on_missing_package)
        )
    return _round_object(data, registry, mode, decimals, on_missing_package)
```

- `_round_object` does the lookup and the call; it is the only place that converts the registry error
  and that applies `on_missing_package`. When a rounder raises `MissingPackageError`: `"raise"`
  raises `UnsupportedTypeError` naming the package, with the `MissingPackageError` as the cause;
  `"warn"` emits a `UserWarning` naming the type and the package and returns `data` unchanged;
  `"ignore"` returns `data` unchanged. The policy covers any rounder (a user rounder that needs a
  missing package gets it too). A type with **no** rounder is not concerned: it raises
  `UnsupportedTypeError` (kept by `recursive=True`), whatever the option.
- With `recursive=True` the policy applies to each leaf, so `"raise"` stops the walk at the first
  polars or pyarrow leaf and `"warn"` warns once per leaf (the default warning filter shows one per
  location).
- `_round_leaf` calls it and catches **only** `UnsupportedTypeError` (F5). A string, `None` or a date
  inside a nested structure is kept; a bug in a rounder is not hidden.
- `validate_arguments` also checks `on_missing_package` (`"raise"`, `"warn"` or `"ignore"`) before
  any work. It accepts any integer (`operator.index`), so `np.int64(2)` works (F8), and still
  rejects `bool` and `float`. The messages do not change.

## 4. Public API after the change

```python
from coola.rounding import (
    round_values,  # (data, *, mode="round", decimals=0, recursive=False, on_missing_package="raise", registry=None)
    RoundMode,  # Literal["round", "floor", "ceil"]
    Rounder,  # Callable[[Any, RoundMode, int], Any]
    UnsupportedTypeError,  # TypeError
    register_rounders,  # (mapping, *, exist_ok=False)
    get_default_registry,  # TypeRegistry[Rounder]
)
```

The modules `coola.rounding.{native,numpy,torch,pandas,polars,pyarrow}` stay public with the
functions they have (`round_numpy(data, mode, decimals)`, ...) so that a user can compose or wrap a
rounder; they are not re-exported from `coola.rounding` (same decision as D6 of `recursive`).
`get_scale`, `MAX_EXACT` and `validate_arguments` stay in `coola.rounding.common`.

## 5. Dependents

| Package                                  | Change                                                                                                                                                                                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.recursive`                        | None. `round_values(recursive=True)` keeps calling `recursive_apply(data, func)` with the default registry of `recursive`; dataclasses are already entered.                                                                                 |
| `coola.utils.singleton`                  | None. `make_default_registry_singleton` and the plugin loader are reused; the default registries test helper (`temporary_default_registries`) must know the rounding singleton (checked by `tests/integration/test_default_registries.py`). |
| `tests/integration/test_lazy_imports.py` | Add the case "importing `coola.rounding` imports no optional library".                                                                                                                                                                      |
| `docs/docs/dev/add_backend.md`           | Add a row `coola.rounding` to the table of section 3: "write a `get_default_rounders` in `rounding/<pkg>.py` and add it to `_LAZY_ROUNDERS`; a float column goes through `round_numpy`"; default: unsupported.                              |
| `docs/docs/dev/architecture.md`          | A paragraph "Rounding" next to "Recursive Transformation" and "Summary": one rule, two kernels, adapters, registry.                                                                                                                         |
| `import-linter`                          | `coola.rounding` is already forbidden to the core packages; it imports `coola.registry`, `coola.recursive` and `coola.utils` only. No change.                                                                                               |

## 6. Evidence: prototypes

Three prototypes were run on `main` and are not part of the repository: the NumPy kernel of section
3.1, its scalar and torch versions, and the polars and pyarrow adapters. The oracle is
`Decimal(repr(x)).quantize(10**-decimals, rounding=ROUND_HALF_EVEN | ROUND_FLOOR | ROUND_CEILING)`
(`repr(np.float32(x))` for `float32`). **[verified]**

| Check                                                                                                                                          | `main`                                        | Prototype                                                                                                                             |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| NumPy `round` against the oracle: 20,000 random values with 3 decimals, 2 decimals                                                             | 133 wrong (0.7 %); the scalar 962             | 0 wrong                                                                                                                               |
| Values one ulp above or below a grid point (24,000 per mode and number of decimals, 1, 2, 3, 5 and 8 decimals)                                 | `floor`/`ceil`: 66 to 426 wrong (up to 1.8 %) | 0 wrong                                                                                                                               |
| 30,000 random values rounded to 1 to 4 decimals, 5,000 uniform values in ±1000, the ties `k/8` and `k/1000 + 0.0005`, 0 to 8 decimals, 3 modes | see above                                     | 0 wrong while `abs(x) * 10**decimals < 1e13` (the sweep below finds none either for `floor`/`ceil` up to the largest magnitude tried) |
| `float32` against the `float32` oracle, 1 to 3 decimals, 3 modes                                                                               | not measured                                  | 0 wrong                                                                                                                               |
| Scalar against the NumPy kernel (value and sign of zero), 3 modes, 0 to 6 decimals                                                             | `round` differs (F3), sign differs (F10)      | identical                                                                                                                             |
| torch against the NumPy kernel, `float64`, `float32`, `float16`, 3 modes, 0 to 3 decimals                                                      | not measured                                  | identical                                                                                                                             |
| polars and pyarrow against the NumPy kernel on 1,000,000 values (the pyarrow input has two chunks)                                             | polars 2,600 to 7,100 wrong per 20,000        | identical, `Float32` kept, nulls kept                                                                                                 |
| Where the kernel stops being exact (magnitudes 1e-4 to 1e6, 2 to 14 decimals, 17-digit values and their grid neighbours)                       | n/a                                           | first mismatch for `round` at `abs(x) * 10**decimals` = 5.6e14, none for `floor`/`ceil`; the error is one step                        |

Time of the prototypes (best of five, Apple M1, 1,000,000 `float64` values, 2 decimals), against
`main` on the same machine:

| Operation                       | `main`                      | Prototype              | Ratio |
| ------------------------------- | --------------------------- | ---------------------- | ----- |
| NumPy `round`                   | 2.2 ms                      | 8.7 ms                 | 4.0x  |
| NumPy `floor`                   | 4.3 ms                      | 4.9 ms                 | 1.1x  |
| NumPy `ceil`                    | 4.2 ms                      | 6.1 ms                 | 1.5x  |
| torch `round`                   | 1.5 ms                      | 7.5 ms                 | 5.0x  |
| torch `floor`                   | 3.0 ms                      | 4.2 ms                 | 1.4x  |
| polars `round` (through NumPy)  | 4.4 ms                      | 8.9 ms                 | 2.0x  |
| polars `floor`                  | 6.2 ms                      | 5.0 ms                 | 0.8x  |
| pyarrow `round` (through NumPy) | 4.0 ms                      | 8.7 ms                 | 2.2x  |
| pyarrow `floor`                 | 4.6 ms                      | 5.1 ms                 | 1.1x  |
| Python `float`, `round`         | 1.27 µs (with the dispatch) | 0.30 µs (the function) | 4x    |

`round` is the slow one: the first prototype used `k % 2` for the even case and took 37 ms;
`np.rint(k + 0.5)` (which is half-to-even) is the same result in 8.7 ms. The gate of section 8 keeps
the ratio at most 5. For the recursive path, the registry lookup is a cached `dict` hit and the
scalar function is 4 times faster, so a list of floats is expected to gain; the real number is a
gate, not a claim.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Breaking                              |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| 1    | **Oracle and conformance tests first (F11).** `tests/unit/rounding/helpers.py` with the `Decimal` oracle, `tests/properties/test_rounding.py` (oracle, mode contract, idempotent, monotone, backends agree), exact comparisons in the table tests. The tests that fail on `main` are marked `xfail(strict=True)` with the finding number, and each following step removes its marks.                                                                       | no                                    |
| 2    | **The rule (F2, F3, F7, F8, F10).** The kernel of section 3.1 in `numpy.py`, `torch.py` and the scalar in `native.py`; `copy` + `copyto` (no masked branch); `copysign`; `decimals` accepts integers. Docstrings, the guide and the "Notes" of `round_values` rewritten around the one rule.                                                                                                                                                               | yes (results at ties and near a grid) |
| 3    | **Tables through NumPy (F1, F9).** `polars.py` and `pyarrow.py` call `round_numpy` per float column; `pa.RecordBatch`; pandas unchanged apart from the exact tests. Delete the polars expression and the pyarrow compute kernel. NumPy is imported when the adapter runs (a missing NumPy raises `MissingPackageError`; step 4 adds the policy).                                                                                                           | yes (needs NumPy; polars results)     |
| 4    | **Registry, error class, plugin group (F4, F5, F6, F12).** `UnsupportedTypeError`, `Rounder`, `register_rounders`, `get_default_registry`, `registry=`, `on_missing_package=` (D9), `_LAZY_ROUNDERS`, `get_default_rounders` in each library module, delete `_BACKENDS` and `import_object` dispatch. `_round_leaf` catches `UnsupportedTypeError` only; a non-strided tensor raises it. Integration tests for a user rounder (`Decimal`) and for threads. | yes (rounder signature for plugins)   |
| 5    | **Documentation.** Rewrite `uguide/rounding.md` (**Properties** table of section 1, the rule, the extension example, the exactness bound, a table of the behaviour by type), the package docstring, `dev/add_backend.md`, the "Rounding" paragraph of `dev/architecture.md`, the migration page (results change, `UnsupportedTypeError`, NumPy for polars and pyarrow, `on_missing_package`), and the status table of `review/suggestions.md`.             | no                                    |
| 6    | **Benchmarks and gate.** `tests/benchmarks/test_rounding_benchmark.py` (section 8).                                                                                                                                                                                                                                                                                                                                                                        | no                                    |

Status of the steps (stacked pull requests): step 1 #2596, step 2 #2597, step 3 #2599, step 4
#2600, step 5 this documentation pull request; step 6 (benchmarks) follows it.

Steps 2 and 3 are independent of 4; step 1 must come first so that every fix has a failing test
before it (`git stash push src`, as `CLAUDE.md` asks).

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/rounding/`); the conventions of `CLAUDE.md` apply
(one behaviour per test, literal expected values, a regression test for every bug, no local
imports, the whole composite output asserted against a literal at least once).

- **The oracle** (`helpers.py`): `Decimal(repr(x)).quantize(...)` with a context large enough for
  300 digits; used by the property tests and by one parametrised unit test per mode over a literal
  list of hard values (ties `2.675`, `0.125`, `8.995`, `-9.995`, `1.005`; one ulp around `0.29`,
  `0.7`, `5.52`, `3.3`; `5e-324`; `1e15 + 0.5`).
- **Kernel** (`test_numpy.py`, `test_torch.py`, `test_native.py`): typical, empty, single-element,
  duplicate values; the three modes with 0, 1, 2 and 8 decimals; ties to even; negative values;
  `-0.0` and the sign of a zero result (`math.copysign`, since `==` hides it); `NaN`, infinities,
  `1e308`, `5e-324`; `decimals=400` (unchanged); float16, float32, float64, bfloat16; non-contiguous
  and 0-d arrays; `np.generic`; the input is not modified and the result shares no memory;
  `matrix`, an `ndarray` subclass and `MaskedArray` keep type, mask and fill value; an integer and a
  boolean array are copies; `UnsupportedTypeError` for complex, object, datetime, string and (torch)
  sparse tensors, with the dtype in the message.
- **Tables** (`test_pandas.py`, `test_polars.py`, `test_pyarrow.py`): a float column is rounded and
  an integer, string, boolean, date and nested column is not; nulls and `NaN` stay where they are;
  `Float32` and nullable `Float64` dtypes are kept; duplicate column names (pandas), chunked arrays
  and `RecordBatch` (pyarrow); the index and the column names are kept; a Series of strings raises.
  **Every comparison is exact** (`check_exact=True`, or `to_list()` against a literal) and at least
  one test uses more than 1 row (the polars bug needs 2). Without NumPy (patched out of
  `sys.modules`) the adapter raises `MissingPackageError`.
- **Interface** (`test_interface.py`): validation order (mode, then decimals, before any rounder
  runs); `np.int64` accepted, `True` and `1.0` rejected; `registry=` is used; a registry miss gives
  `UnsupportedTypeError` with the registry error as the cause; `recursive=True` keeps an unsupported
  leaf and propagates a `ValueError` raised by a rounder (a regression test for F5); the whole
  nested output asserted against a literal; `on_missing_package` with a rounder that raises
  `MissingPackageError`: `"raise"` gives `UnsupportedTypeError` with the cause, `"warn"` warns and
  returns the same object, `"ignore"` returns it silently, an invalid value raises before any work, a
  type with no rounder raises in all three, and with `recursive=True` the policy applies per leaf.
  These are patched unit tests; the integration test below exercises the real case.
- **Properties** (`tests/properties/test_rounding.py`, hypothesis, floats with 1 to 17 significant
  digits and the neighbours of grid points): result equals the oracle while `abs(x) * 10**d <
1e13`; `floor <= x <= ceil`; `abs(round - x)` at most half a step; idempotent; monotone; **the
  native scalar, NumPy, torch, pandas, polars and pyarrow give the same bits** (new, ties the
  backends); the sign of zero.
- **Integration** (`tests/integration/rounding/`): the default registry end to end for every
  installed library with no patching; a user type registered with `register_rounders`; a plugin
  group entry (as the other packages test theirs); `round_values(recursive=True)` on a dataclass
  holding an array, a Series and a float; a subprocess without NumPy rounding a polars or a pyarrow
  object with the three values of `on_missing_package` (no patching); threads running `round_values` on the default registry
  with a lock around the shared list (`run_threads`, docstring "Test that ..."); lazy imports.
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Benchmark gate** (`tests/benchmarks/test_rounding_benchmark.py`, new): a Python float; 1,000,000
  `float64` in NumPy, torch, polars and pyarrow; a pandas Series; a list and a nested dict of
  100,000 floats with `recursive=True`. Gate, against `main`: `floor` and `ceil` at most 1.5 times
  slower, `round` at most 5 times, the scalar and the recursive path not slower. A failing gate
  rejects the step, except for a ratio that the prototype of section 6 already explains.
- Coverage: 100 % of `src/coola/rounding` (`--cov=coola`), no `# pragma: no cover`.

## 9. Decisions for the maintainers

**D1. Which rule for `"round"`?** _Recommended: the decimal value, halfway to even._ `floor` and
`ceil` already read `0.29` as `0.29`; `round` should read `2.675` as `2.675`, a tie, and give
`2.68`. This is the only rule under which the three modes mean the same thing, under which the
backends can agree, and that a user can predict by looking at the number. Alternatives: (a) the
exact binary value like Python (`round(2.675, 2)` is `2.67`, but then `floor(0.29, 2)` must be
`0.28`, which the package refuses on purpose, or the modes disagree as they do today); (b) keep
scale and `rint` (inexact, F3). Consequence: `round_values` no longer agrees with `round()` on
the halfway cases of a decimal literal, which the guide says in one sentence.

Decision: follow recommendation

**D2. The native scalar: the float formula or `Decimal`?** _Recommended: the float formula._
`Decimal(repr(x)).quantize(...)` is simple and exact, but it costs about 1.2 µs and needs a context
for huge exponents (it raises `InvalidOperation` for `decimals=400`). The float formula costs 0.30 µs, is bit-identical to the arrays by
construction, and the `Decimal` version stays in the tests as the independent oracle: a kernel
checked against a copy of itself proves nothing.

Decision: follow recommendation

**D3. Tables through NumPy, or one kernel per table library?** _Recommended: through NumPy._ polars
cannot do the exact division (F1), and a second and a third copy of the formula is how the package
got here. Cost: a conversion per column (`to_numpy`, an `array` back), the timings of section 6
(0.8x to 2.2x of today), and **NumPy becomes required to round a polars or pyarrow object** (pandas
already needs it). The error is the usual `MissingPackageError` for NumPy. Alternative: keep the
polars and pyarrow kernels and make the conformance test the only guard; rejected because polars
has no exact operation to build it from.

Decision: polars and pyarrow go through NumPy when it is available. When it is not, the user
chooses what happens with `on_missing_package` (D9): raise an error (default), warn, or ignore the
object.

**D4. A function or a class as the extension point?** _Recommended: a function (`Rounder`)._ A
rounder has no state, no children and one job; `Hasher`/`BaseHasher` exist because a hasher has a
protocol with several members. A plain `TypeRegistry[Rounder]` with `register_rounders` and
`get_default_registry` is about 40 lines; a `BaseRounder` hierarchy would triple it. Alternative:
the family convention (`Rounder` protocol, `BaseRounder`, `RounderRegistry` with `find_rounder`);
reconsider if a rounder ever needs more than one method.

Decision: follow recommendation

**D5. Keep `recursive=False` as the default?** _Recommended: yes._ Strict by default (a string or
a `Decimal` raises) is what a user rounding one object wants; `recursive=True` is the explicit
opt-in to the lenient walk where unsupported leaves are kept. Alternatives: always recursive (a
`list` of floats would work, an error for `"abc"` would be lost) or a separate function (two names
for one concept, and `nested` already has the walk).

Decision: follow recommendation

**D6. Negative `decimals` (round to tens, hundreds)?** _Recommended: not now._ Python, NumPy and
pandas accept it, but `x * 10**-2` is not exact, so the kernel needs a second path (divide by
`10**n`, multiply back) in three implementations, for a need that nobody has asked for. The error
message already names the rule. Revisit with the request; the rule of section 3.1 extends to it
(`step = 10**n`).

Decision: follow recommendation

**D7. Nested columns of a table (`list<float>`, struct).** _Recommended: left unchanged, documented._
Entering them needs a second walk (Arrow lists, polars `List`) in two libraries and is a different
feature (rounding the values inside the cells). The property "only floating-point columns" states it.

Decision: follow recommendation

**D8. Where the guarantee stops.** _Recommended: document `1e13`, do not widen it._ The first
mismatch measured is at `9.9e13`, and the error is one step, but a proof is out of reach with floats.
Past the bound the answer is still a grid point within one step; a user who needs exactness at 15
digits uses `Decimal` with a registered rounder.

Decision: follow recommendation

**D9. What happens when NumPy is missing for polars and pyarrow?** _Decision: an option,
`on_missing_package`, with three values; the default raises._

| Value               | Behaviour                                                                              |
| ------------------- | -------------------------------------------------------------------------------------- |
| `"raise"` (default) | `UnsupportedTypeError` naming the missing package, `MissingPackageError` as the cause. |
| `"warn"`            | `UserWarning` naming the type and the package; the object is returned unchanged.       |
| `"ignore"`          | The object is returned unchanged, silently.                                            |

Refinements, so that the option stays small and predictable:

- The option is about a **missing package**, not about every unsupported type. It applies to any
  rounder that raises `MissingPackageError` (NumPy for polars and pyarrow today, and a user rounder
  that needs another package). A type with no rounder (a `str`, a `Decimal`, a sparse tensor) is not
  affected: `UnsupportedTypeError`, kept by `recursive=True`. This avoids a default of `"raise"`
  that would change the lenient walk of `recursive=True`.
- The rounders are registered when the library is installed and check NumPy when they run, so the
  registry, the lazy loading and `import coola.rounding` do not change.
- `"warn"` and `"ignore"` can hide an unrounded column; they are opt-in. `"warn"` is the compromise
  for a pipeline that must not fail on an optional step.
- The name is `on_missing_package` (not `on_missing_numpy`) because the mechanism is general; the
  keyword is keyword-only and last before `registry`.

## 10. Not covered

- `pandas.Index`, `polars.LazyFrame`, `pyarrow` scalars: `UnsupportedTypeError`. A `LazyFrame` could
  be rounded with a `map_batches`; it is a separate item.
- `Decimal`, `Fraction`, complex numbers, dates, times and durations: not numerical in the sense of
  the package. `Decimal` is the example of the extension guide.
- JAX, xarray, CuPy: a rounder is about 20 lines on the pattern of `round_torch`; registered by a
  plugin or by the user. Not part of this design.
- Sparse, nested and quantized tensors: `UnsupportedTypeError`.
- Rounding to significant digits, `"trunc"` and `"half_up"` modes: new `RoundMode` values and the
  same kernel with a different last step; a separate item.
- In-place rounding (`out=`): the package is pure by design.

## Appendix: effect on the size of the package

| Module         | Lines today | Expected after | Why                                                                                                              |
| -------------- | ----------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `common.py`    | 73          | about 110      | `UnsupportedTypeError`, `Rounder`, `validate_arguments` accepts integers                                         |
| `interface.py` | 135         | about 250      | registry, `register_rounders`, `get_default_registry`, `registry=`, `on_missing_package`, lazy table, docstrings |
| `native.py`    | 63          | about 70       | the corrected formula, `round_int`, `get_default_rounders`                                                       |
| `numpy.py`     | 89          | about 85       | the corrected formula; the masked-array branch is gone; a longer docstring                                       |
| `torch.py`     | 74          | about 85       | the corrected formula, the layout check                                                                          |
| `pandas.py`    | 81          | about 75       | the `_is_supported` helper and the registration                                                                  |
| `polars.py`    | 89          | about 50       | no expression, no kernel                                                                                         |
| `pyarrow.py`   | 88          | about 65       | no compute kernel, `RecordBatch`                                                                                 |
| `__init__.py`  | 12          | about 25       | the new names                                                                                                    |
| **Total**      | **704**     | **about 820**  | the registry is the one real addition; two kernels and the masked branch pay for part of it                      |

The size grows by about 120 lines, all in the interface and its docstrings; the _logic_ shrinks
(five copies of the formula become two kernels and a scalar). The line counts are estimates from
the prototypes; the gate is the behaviour (sections 1 and 8), not the size.
