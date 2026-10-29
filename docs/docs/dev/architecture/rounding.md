# Architecture of `coola.rounding`

:book: This page describes the design of the `coola.rounding` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../../uguide/rounding.md#properties); the API is in the [reference](../../refs/rounding.md).

## Overview

`round_values` has **one rule, two kernels, one adapter per table library and one extension
point**. A float is read as the decimal that `repr` shows, and `"floor"`, `"ceil"` and `"round"`
(halfway cases to the even digit) act on that decimal. The NumPy kernel (`coola.rounding.numpy`)
finds the grid point with float arithmetic and two checks that compare the grid points in the dtype
of the data, so the result equals `Decimal(repr(x)).quantize(...)` without `Decimal`. The torch
kernel and the native scalar are the same formula with the operations of their library, and the
tests compare them bit for bit with the NumPy kernel and with a `Decimal` oracle that is independent
of the implementation. The adapters of pandas, polars and pyarrow find the floating-point columns,
round each with the NumPy kernel, restore the nulls and put the column back: "coola owns the
semantics, the libraries own the structure". A rounder is a plain function
`(data, mode, decimals) -> result` registered in a `TypeRegistry` (lazy per library, MRO lookup,
the plugin group `coola.rounding.rounders`). `round_values` owns the validation, the dispatch, the
recursive walk and the error contract (`UnsupportedTypeError`, `on_missing_package`). See the
[user guide](../../uguide/rounding.md#properties) for the properties of the package.

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
   is preferred to a fast rule that is wrong on 1 % of the inputs. The gate of bounds the
   cost (`floor`/`ceil` unchanged, `round` at most 5 times slower on a large array, a scalar and the
   recursive path faster).
4. **Consistent with the siblings.** One extension point (a registry, `register_*`, a plugin group),
   keyword-only options, errors raised before any work with a message that names the remedy,
   dataclasses entered with `recursive=True`, stateless and thread-safe.
5. **Optional libraries stay lazy.** `import coola.rounding` imports no optional library (verified:
   `numpy`, `torch`, `pandas`, `polars` and `pyarrow` are all absent from `sys.modules` after it) and
   `coola` still has zero hard dependencies. New: rounding a polars or a pyarrow object needs NumPy
   ; without it `on_missing_package` decides: raise (default), warn or ignore.

## Architecture in detail

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

### The rule

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
repairs it in both directions (the current code repairs one direction only, ). For `"round"`
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

The code above is the prototype without its `errstate` and the integer/boolean
branch; the real function wraps it as before (copy for integers and booleans, `UnsupportedTypeError`
for the other dtypes, `np.errstate(over="ignore", invalid="ignore")`).

### The scalar, the tensor, the tables

- **`round_float` (native).** The same formula on one `float`, with `math.floor`, integer `k` and
  `math.copysign`: about 15 lines, 0.30 µs per call, bit-identical to the NumPy kernel **[verified
  on 100,010 values, 3 modes, 5 values of `decimals`, signs of zero included]**. `int` (and `bool`) are returned
  as they are. The `Decimal` implementation is the specification and lives in the tests only.
- **`round_torch`.** The same formula with `torch.floor`, `torch.where`, `torch.round` (which is
  half-to-even), the dtype of the tensor for the grid, `float32` instead of `float64` on `mps` (as
  today), `detach().clone()` for the integers. The result is bit-identical to the NumPy kernel for
  `float64`, `float32` and `float16`. A non-strided layout (sparse, nested) raises
  `UnsupportedTypeError` instead of the raw `NotImplementedError` of torch.
- **Type preservation.** NumPy: `out = data.copy()` then `np.copyto(out, rounded)`. It keeps
  a `matrix`, a subclass and a `MaskedArray` with its mask and fill value, so the
  masked branch goes away. `np.generic` goes through a 0-d array and `[()]`, as before.
- **Tables.** The three adapters share one shape: find the floating-point columns, round each with
  `round_numpy` on its NumPy values, restore the nulls, put the column back with its name and dtype.
  - pandas: as before (`isetitem` keeps duplicate column names; the `pandas` nullable and Arrow dtypes
    already go through `to_numpy(na_value=nan)`); a Series of integers is returned as a copy.
  - polars: `pl.Series(name, round_numpy(s.to_numpy(), ...), dtype=s.dtype)` then `zip_with(
s.is_not_null(), s)` when there are nulls. 20 lines instead of 40, no expression, no
    `Series / scalar`. equal to the kernel on 1,000,000 values, in three modes,
    with nulls and `NaN`, `Float32` kept.
  - pyarrow: `pa.array(round_numpy(a.to_numpy(zero_copy_only=False), ...), type=a.type, mask=...)`
    per column and `type(data).from_arrays(columns, schema=data.schema)` for `Table` and
    `RecordBatch`. equal to the kernel, `float32` kept, nulls kept.
  - A column that is not floating point is returned as it is (an integer column is not copied for
    polars and pyarrow: their buffers are immutable).
  - NumPy is imported when the adapter runs, not when it is registered. If it is missing the adapter
    raises `MissingPackageError` before any work, and `round_values` applies `on_missing_package`
    .

### The rounder contract and the registry

```python
Rounder = Callable[[Any, RoundMode, int], Any]


class UnsupportedTypeError(TypeError):
    """The object, or its dtype, cannot be rounded."""
```

A rounder is a plain function: no base class, no protocol, no state. It receives a `mode` and a
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
and import NumPy when they run. The
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

### The interface

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
- `_round_leaf` calls it and catches **only** `UnsupportedTypeError`. A string, `None` or a date
  inside a nested structure is kept; a bug in a rounder is not hidden.
- `validate_arguments` also checks `on_missing_package` (`"raise"`, `"warn"` or `"ignore"`) before
  any work. It accepts any integer (`operator.index`), so `np.int64(2)` works, and still
  rejects `bool` and `float`. The messages do not change.

## Public API

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

## Dependents

| Package                                  | Change                                                                                                                                                                                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.recursive`                        | None. `round_values(recursive=True)` keeps calling `recursive_apply(data, func)` with the default registry of `recursive`; dataclasses are already entered.                                                                                 |
| `coola.utils.singleton`                  | None. `make_default_registry_singleton` and the plugin loader are reused; the default registries test helper (`temporary_default_registries`) must know the rounding singleton (checked by `tests/integration/test_default_registries.py`). |
| `tests/integration/test_lazy_imports.py` | Add the case "importing `coola.rounding` imports no optional library".                                                                                                                                                                      |
| `import-linter`                          | `coola.rounding` is already forbidden to the core packages; it imports `coola.registry`, `coola.recursive` and `coola.utils` only. No change.                                                                                               |

## Decisions for the maintainers

**D1. Which rule for `"round"`?** _Recommended: the decimal value, halfway to even._ `floor` and
`ceil` already read `0.29` as `0.29`; `round` should read `2.675` as `2.675`, a tie, and give
`2.68`. This is the only rule under which the three modes mean the same thing, under which the
backends can agree, and that a user can predict by looking at the number. Alternatives: (a) the
exact binary value like Python (`round(2.675, 2)` is `2.67`, but then `floor(0.29, 2)` must be
`0.28`, which the package refuses on purpose, or the modes disagree as they do today); (b) keep
scale and `rint` (inexact, ). Consequence: `round_values` no longer agrees with `round()` on
the halfway cases of a decimal literal, which the guide says in one sentence.

Decision: follow recommendation

**D2. The native scalar: the float formula or `Decimal`?** _Recommended: the float formula._
`Decimal(repr(x)).quantize(...)` is simple and exact, but it costs about 1.2 µs and needs a context
for huge exponents (it raises `InvalidOperation` for `decimals=400`). The float formula costs 0.30 µs, is bit-identical to the arrays by
construction, and the `Decimal` version stays in the tests as the independent oracle: a kernel
checked against a copy of itself proves nothing.

Decision: follow recommendation

**D3. Tables through NumPy, or one kernel per table library?** _Recommended: through NumPy._ polars
cannot do the exact division, and a second and a third copy of the formula is how the package
got here. Cost: a conversion per column (`to_numpy`, an `array` back), the timings of
(0.8x to 2.2x of the previous version), and **NumPy becomes required to round a polars or pyarrow object** (pandas
already needs it). The error is the usual `MissingPackageError` for NumPy. Alternative: keep the
polars and pyarrow kernels and make the conformance test the only guard; rejected because polars
has no exact operation to build it from.

Decision: polars and pyarrow go through NumPy when it is available. When it is not, the user
chooses what happens with `on_missing_package`: raise an error (default), warn, or ignore the
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
message already names the rule. Revisit with the request; the rule of extends to it
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
  rounder that raises `MissingPackageError` (NumPy for polars and pyarrow before, and a user rounder
  that needs another package). A type with no rounder (a `str`, a `Decimal`, a sparse tensor) is not
  affected: `UnsupportedTypeError`, kept by `recursive=True`. This avoids a default of `"raise"`
  that would change the lenient walk of `recursive=True`.
- The rounders are registered when the library is installed and check NumPy when they run, so the
  registry, the lazy loading and `import coola.rounding` do not change.
- `"warn"` and `"ignore"` can hide an unrounded column; they are opt-in. `"warn"` is the compromise
  for a pipeline that must not fail on an optional step.
- The name is `on_missing_package` (not `on_missing_numpy`) because the mechanism is general; the
  keyword is keyword-only and last before `registry`.

## Not covered

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
