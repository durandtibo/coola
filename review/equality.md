# `coola.equality`: target state and the changes to reach it

This document describes the design that `coola.equality` should have, then the changes to make to
the current implementation (`main`, `bb94c6fd`) to get there. The design follows the **Guarantees**
table of `docs/docs/uguide/equality.md`. Its aim is a small package that is easy to read: one engine,
one extension point, one numeric rule, one frame algorithm. Breaking changes are accepted.

## Constraints

1. **Performance is a requirement.** A comparison that is too slow is not useful. Both the passing and
   the failing path are measured, every step of section 10.2 must keep the benchmark gates of
   section 10.3, and a design that is simpler but slower is rejected unless the gate is met.
2. **Early exit.** Two objects are different as soon as one difference is found: the comparison
   stops there and nothing after it is visited, converted or formatted.
3. **First difference only, on demand.** Only the first difference is explained, and only when an
   explanation is asked for (`compare`, `assert_*`). `objects_are_equal` and `objects_are_allclose`
   build no message and no path.
4. **Equality drives hashing, not the reverse.** Equality is designed for its own users; `hash_object`
   is then aligned with it, in a later change outside this design (last section). The property
   "equal objects have the same hash" is a goal for the hashers to meet, never a reason to bend a
   comparison rule.
5. **Major release.** Breaking changes are fine and compatibility is not a design input: there is no
   adapter, no alias, no deprecation period. Deprecated and compatibility code is removed (for
   example the `RuntimeError` base of `HandlerCycleError` and the `RecursionError` base of
   `MaxDepthExceededError`, which has no reason to exist once the engine does not recurse). The
   release notes list the changes.

Evidence comes from probes and benchmarks run on `main` (Python 3.14, numpy 2.5, torch 2.14, pandas
3.0, polars 2.0, pyarrow 26, jax 0.12, xarray 2026.9, pydantic 2.14) and from a throw-away prototype
of the engine kept outside the repository (section 9). Items marked **[verified]** were reproduced on
`main`. Nothing in `src`, `tests` or `docs` was changed.

## 1. Properties the design must satisfy

These are the rules of the user guide. Every design choice below exists to make one of them true by
construction instead of by the discipline of each tester.

| Property                | Statement                                                                                                                                                                                                                                                                                                                          | Enforced by                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| Strict type             | Same exact type or different (`1`, `1.0`, `True`; `[1]`, `(1,)`; `dict`, `OrderedDict`; a subclass).                                                                                                                                                                                                                               | the engine, before any tester                 |
| Reflexive               | `x` equals `x` and a copy of `x`; the exception is a NaN-like value without `equal_nan`.                                                                                                                                                                                                                                           | the engine (identity); the missing-value rule |
| Recursive, order rules  | Containers item by item at any depth; sequences ordered; `dict` unordered; `OrderedDict` ordered.                                                                                                                                                                                                                                  | container testers                             |
| Value semantics         | `==` unless a tester exists; arrays and frames also compare dtype and shape.                                                                                                                                                                                                                                                       | the registry, the default tester              |
| Missing values          | A NaN-like value (float or complex `NaN`, `Decimal("NaN")`, NumPy `NaT`) equals another only with `equal_nan=True` or if it is the same object, in every container **except the nested and `object` cells of frames, which follow the library (Q6)**. A null (`None`, `pd.NA`, polars or Arrow null) equals a null, always.        | the numeric kernel and the table comparer     |
| Signed zero, time zones | `-0.0` equals `0.0`; aware datetimes compare by instant.                                                                                                                                                                                                                                                                           | kernel, `==`                                  |
| Tolerance               | `abs(a - b) <= atol + rtol * max(abs(a), abs(b))` per element, for every number in every container, exact for integers. `bool`, enum members, instants, strings and the structure are exact. Durations follow the tolerance (`atol` in seconds). The nested and `object` cells of frames are compared exactly by the library (Q6). | one kernel, one number tester                 |
| Symmetric               | `eq(a, b)` and `eq(b, a)` agree, with and without tolerance.                                                                                                                                                                                                                                                                       | kernel formula, engine type rule              |
| Not transitive, pure    | A tolerance is not transitive; a comparison never modifies its inputs.                                                                                                                                                                                                                                                             | (documented)                                  |
| Failure modes           | Cycle: equal. Nesting above `max_depth`: `MaxDepthExceededError`. `==` that raises or returns a non-bool: `TypeError` naming the type; other exceptions propagate. No `RecursionError`, whatever the caller's stack.                                                                                                               | the engine (explicit stack)                   |
| Consistent with hashing | Equal objects have the same `hash_object`. The equality rules above are the reference; where a hasher disagrees, the hasher is changed (last section).                                                                                                                                                                             | the hashers, later                            |
| Fast, early exit        | A comparison stops at the first difference and visits nothing after it; the failing path costs what the passing path costs (benchmarked); only the first difference is explained, and only on demand.                                                                                                                              | the engine, lazy reasons, benchmark gates     |
| Location                | `compare` and `assert_*` give the path of the first difference: keys, positions, attributes, the index in an array, the element of a set. A table (data frame, series, Arrow table) is one leaf: it is reported as different, and the user locates the difference with the library's own tools (R5).                               | the engine's stack and `ctx.fail`             |

## 2. Architecture

```
objects_are_equal / objects_are_allclose / compare / assert_objects_equal / assert_objects_allclose
        |   one EqualityConfig (frozen) and one Engine per call
        v
Engine.run(actual, expected)        explicit stack of child iterators; one loop
        |   identity -> type -> tester lookup (cached per call) -> verdict or children
        v
Tester.objects_are_equal(actual, expected, ctx) -> bool | Iterator[(actual, expected, path part)]
        |   leaves: numbers, arrays, scalars          containers: sequence, mapping, set, object, table
        v
_numeric.close(...)   one kernel for NumPy, torch, JAX
adapters  pandas, polars, pyarrow -> Table      (structure by the library, numbers by the kernel)
```

Four rules keep it simple:

1. **One extension point: the tester.** No handler chain, no steps. A tester is a class with one
   method.
2. **The engine owns the invariants**: identity, strict type, depth, cycles, path, first failure.
   A tester cannot forget them because it never recurses.
3. **A container does not recurse, it decomposes**: it returns the child pairs that must all be
   equal. A table compares its columns in a loop and stops at the first column that differs.
4. **coola owns the semantics, the libraries own the structure**: every number goes through the
   kernel; a library says whether two frames have the same columns, never whether `1` is close to `2`.

## 3. Public API

```python
objects_are_equal(actual, expected, *, equal_nan=False, max_depth=100, registry=None) -> bool
objects_are_allclose(actual, expected, *, rtol=1e-5, atol=1e-8, equal_nan=False, max_depth=100, registry=None) -> bool
compare(actual, expected, *, rtol=0.0, atol=0.0, equal_nan=False, max_depth=100, registry=None) -> ComparisonResult
assert_objects_equal(actual, expected, *, equal_nan=False, max_depth=100, registry=None, root="actual") -> None
assert_objects_allclose(actual, expected, *, rtol=1e-5, atol=1e-8, equal_nan=False, max_depth=100, registry=None, root="actual") -> None
```

The five functions have the same keywords with the same meaning. `compare` is exact unless `rtol` or
`atol` is given. There is no `show_difference`: `compare` and `assert_*` explain.

```python
@dataclass(frozen=True, slots=True)
class Tolerance:  # validated: finite, non-negative, not a bool
    rtol: float = 0.0
    atol: float = 0.0


@dataclass(frozen=True, slots=True)
class EqualityConfig:  # shareable, hashable, thread-safe
    registry: EqualityTesterRegistry
    tolerance: Tolerance = Tolerance()
    equal_nan: bool = False
    max_depth: int = 100  # validated: positive int, not a bool


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    equal: bool
    path: tuple[object, ...] = ()  # keys, positions, AttributeName, Element
    reason: str | None = None

    def __bool__(self) -> bool: ...
    def format_message(self, root: str = "actual") -> str: ...
```

Exported by `coola.equality`: the five functions, `ComparisonResult`, `EqualityConfig`, `Tolerance`,
`MaxDepthExceededError`, `AttributeName`, `Element`, `BaseEqualityTester`, `Engine` (the type of
`ctx`), `EqualityTesterRegistry`, `get_default_registry`, `register_equality_testers`, and the
the named testers (`ObjectAttributesEqualityTester`, `EqualEqualityTester`,
`OrderedMappingEqualityTester`, `SequenceEqualityTester`, `MappingEqualityTester`, ... and the
library testers `NumpyArrayEqualityTester`, `TorchTensorEqualityTester`, `TableEqualityTester`, ...,
decision Q12) so that a user can compose, subclass or register them. The handlers and the helpers
are private; the kernel `close` is public (R3).

## 4. The engine and the testers

### 4.1 The tester contract

```python
class BaseEqualityTester(ABC, Generic[T]):
    @abstractmethod
    def objects_are_equal(
        self, actual: T, expected: T, ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        """A verdict (a leaf), or the child pairs (actual, expected, path part) that must all be
        equal (a container), produced lazily and in the order the first difference is wanted."""

    # default __eq__ / __repr__: same type and same vars
```

`ctx` gives `ctx.atol`, `ctx.rtol`, `ctx.equal_nan` (read once from the config), `ctx.fail(reason,
*path)`, `ctx.explaining` and `ctx.warn_once(key, message)`. `ctx.fail` returns `False`, so a tester
writes `return ctx.fail(...)`. `reason` may be a callable; it is called only when `ctx.explaining`
(`compare`, `assert_*`), so `objects_are_equal` formats nothing. `path` adds parts after the ones the
engine knows (the index in an array).

### 4.2 The engine

One loop with a stack of child iterators (depth first, so the **first** difference in iteration
order is the one reported). The full loop is in the appendix. For each pair:

1. `actual is expected`: equal.
2. `type(actual) is not type(expected)`: different (the engine writes the reason).
3. Look the tester up in a `dict[type, tester]` filled from the registry on the first use of a type
   in the call (the registry lookup costs 300 ns, so it is paid once per type).
4. Call the tester. A `bool` is the verdict. An iterator is a container: if its pair is already on
   the stack the pair is equal (cycle); if the stack is `max_depth` deep, raise
   `MaxDepthExceededError` (a plain `Exception`, no longer a `RecursionError`); otherwise push it and continue with its first child.
5. On `False`, the path is the path parts of the stack, the part of the failing pair and the parts
   given to `ctx.fail`. It is read once, here.

The loop returns at the first `False`: nothing after the first difference is visited, and nothing
is formatted unless the caller asked for an explanation.

Consequences: stack use is constant (100 000 nested dicts compare under the default interpreter
limit of 1 000), so `RecursionError` cannot happen and `max_depth` is a policy, not a safety net;
depth, cycles and the path are the bookkeeping of one function; explaining costs nothing in the loop.

### 4.3 Testers

```python
class SequenceEqualityTester(BaseEqualityTester[Sequence]):
    def objects_are_equal(self, actual, expected, ctx):
        if len(actual) != len(expected):
            return ctx.fail(lambda: f"lengths differ: {len(actual)} vs {len(expected)}")
        return ((a, e, i) for i, (a, e) in enumerate(zip(actual, expected)))


class MappingEqualityTester(BaseEqualityTester[Mapping]):
    def objects_are_equal(self, actual, expected, ctx):
        keys_a, keys_e = (
            key_index(actual, ctx),
            key_index(expected, ctx),
        )  # comparison_key: types, NaN keys
        if keys_a.keys() != keys_e.keys():
            return ctx.fail(lambda: describe_key_difference(keys_a, keys_e))
        return ((actual[keys_a[k]], expected[keys_e[k]], keys_a[k]) for k in keys_a)


class NumpyArrayEqualityTester(BaseEqualityTester[np.ndarray]):
    def objects_are_equal(self, actual, expected, ctx):
        if actual.dtype != expected.dtype:
            return ctx.fail(lambda: f"dtypes differ: {actual.dtype} vs {expected.dtype}")
        if actual.shape != expected.shape:
            return ctx.fail(lambda: f"shapes differ: {actual.shape} vs {expected.shape}")
        if actual.dtype.kind == "O":  # the cells are objects: children of the engine
            return (
                (a, e, i)
                for i, (a, e) in enumerate(zip(actual.ravel().tolist(), expected.ravel().tolist()))
            )
        close = close_numpy(actual, expected, ctx)  # boolean array, the kernel of section 6
        if close.all():
            return True
        index = np.unravel_index(np.argmin(close), close.shape)  # looked up only when it fails
        return ctx.fail(lambda: describe_values(actual[index], expected[index]), *index)


class ScalarEqualityTester(BaseEqualityTester[float]):  # int, float, complex, NumPy scalars
    def objects_are_equal(self, actual, expected, ctx):
        if isinstance(actual, (bool, enum.Enum)):  # labels, not quantities
            return actual == expected or ctx.fail(lambda: describe_values(actual, expected))
        return number_close(actual, expected, ctx) or ctx.fail(
            lambda: describe_values(actual, expected)
        )
```

A user tester for their own class returns the pairs of its attributes:

```python
class PointEqualityTester(BaseEqualityTester[Point]):
    def objects_are_equal(self, actual, expected, ctx):
        return ((actual.coords, expected.coords, AttributeName("coords")),)
```

A tester has no state but its configuration; `__eq__` and `__repr__` come from the base class.

### 4.4 Type rules

| Type                                                                        | Tester                                                   | Rule                                                                                                                             |
| --------------------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `object` (unregistered)                                                     | `DefaultEqualityTester`                                  | `==`; a raise or a non-bool result is a `TypeError` naming the type                                                              |
| `int`, `float`, `complex`, NumPy scalars                                    | `ScalarEqualityTester`                                   | kernel formula; `bool` and enums exact                                                                                           |
| `Decimal`, `Fraction`, `timedelta`                                          | `DifferenceCloseEqualityTester`                          | difference-based tolerance; `Decimal` NaN by `equal_nan`                                                                         |
| `list`, `tuple`, `deque`, `Sequence`, `array.array`, dict values            | `SequenceEqualityTester`                                 | length, then items in order                                                                                                      |
| `dict`, `Mapping`, `MappingProxyType`                                       | `MappingEqualityTester`                                  | same keys (by type, NaN by `equal_nan`), then values                                                                             |
| `OrderedDict`                                                               | `OrderedMappingEqualityTester`                           | and the same key order                                                                                                           |
| `set`, `frozenset`                                                          | `SetEqualityTester`                                      | elements matched by hash and type; reports `Element`                                                                             |
| dataclasses, pydantic models                                                | `DataclassEqualityTester`, `PydanticModelEqualityTester` | fields as `AttributeName` children                                                                                               |
| `np.ndarray`, `np.ma.MaskedArray`, NumPy `datetime64`/`timedelta64` scalars | `Numpy*EqualityTester`                                   | dtype, shape, kernel; for a masked array the data (masked values included), the mask and `fill_value` (Q5); `NaT` by `equal_nan` |
| `torch.Tensor`, `PackedSequence`, `jax.Array`, xarray                       | `Torch*`, `Jax*`, `Xarray*EqualityTester`                | device, dtype, shape, layout, kernel; xarray parts as children                                                                   |
| pandas, polars, pyarrow frames, series, arrays                              | `TableEqualityTester` (section 7)                        | structure, then columns by kind                                                                                                  |

## 5. Reporting

- One record: the first failure, `(path, reason)`. No list of reports, no reversal, no message per
  parent level (the path says where).
- A reason is a `str` or a callable and is built only when `explaining`. Values inside a reason go
  through one bounded formatter (`reprlib.Repr` with limits; the array libraries summarise
  themselves), so a failure on 1M elements formats a few values, not the objects.
- `ComparisonResult` carries `equal`, `path` and `reason`. The reason holds the two differing values;
  the roots are not repeated.
- Path parts: a key or a position is itself (`['a']`, `[3]`), `AttributeName("x")` is `.x`,
  `Element(2)` is `{2}`. An array reports its index `(1, 2)`. A table has no location below itself (R5).

```
objects are not equal at actual['prices']           # a dict holding a DataFrame
tables have different values (use the library to find where)

objects are not equal at actual.weights[1][2]       # dataclass field, array element
values differ: 0.5 vs 0.25

objects are not equal at actual{2}                  # set
element has no match in expected: 2

objects are not equal at actual['a']
types differ: int vs float
```

## 6. The numeric kernel

`coola/equality/_numeric.py` holds the formula for NumPy, torch and JAX (`xp` is the module) and the
scalar helpers:

```
close = (a == b) | (finite & (|a - b| <= atol + rtol * max(|a|, |b|))) | (isnan(a) & isnan(b) if equal_nan)
```

| Dtype                                | Preparation                                                      | Rule                                     |
| ------------------------------------ | ---------------------------------------------------------------- | ---------------------------------------- |
| `bool`                               | none                                                             | exact (`==`)                             |
| signed and unsigned integers         | exact difference in integers (wrap-around, `+ 2**64` if wrapped) | kernel, tolerance side in `float64`      |
| `float16`, `bfloat16`, `float32`     | cast to `float64`                                                | kernel                                   |
| `float64`, `complex128`              | none (`complex64` cast up)                                       | kernel (`abs` of the complex difference) |
| `timedelta64`                        | `int64` nanoseconds, `atol * 1e9`; `NaT` mask apart              | kernel; `NaT` by `equal_nan`             |
| `datetime64`, `str`, `bytes`, `void` | none                                                             | exact; `NaT` by `equal_nan`              |
| `object`                             | none                                                             | cells as children of the engine          |

The integer difference is exact (checked on `2**62` against `2**62 + 1`, `-2**63` against
`2**63 - 1` and the `uint64` extremes, in NumPy and torch); the final comparison to a float tolerance
is made in `float64`, so a difference of exactly `atol` above 2^53 can round the other way than the
`Fraction` of `int_isclose` (a measure-zero case, documented in the docstring). Python scalars use the
same formula; `Decimal`, `Fraction` and `timedelta` keep their difference-based rule. The torch
variant needs `torch.maximum`; the NumPy caller silences `invalid` and `over`.

## 7. Frames, series and Arrow tables: one comparer, three adapters

pandas, polars and pyarrow run the same algorithm: check the structure, then compare the columns,
each by its kind, and stop at the first column that differs. It is written once.

```python
@dataclass(frozen=True, slots=True)
class Column:
    name: object
    kind: Literal["numeric", "duration", "exact"]
    values: object  # NumPy array (numeric, duration) or the library's own column (exact)
    nulls: np.ndarray | None  # boolean mask


class Table(Protocol):  # what an adapter returns
    kind_name: str  # "table", "series" or "array": the word used in the reasons (S3)

    def structure(self) -> Hashable: ...  # equal for the same shape, names, dtypes, index
    def columns(self) -> Iterator[Column]: ...


class TableEqualityTester(BaseEqualityTester[object]):
    def objects_are_equal(self, actual, expected, ctx):
        a, e = to_table(actual), to_table(expected)  # the adapter of the type
        if a.structure() != e.structure():
            return ctx.fail(lambda: f"{a.kind_name}s have a different structure")
        for ca, ce in zip(
            a.columns(), e.columns()
        ):  # lazily: a column is converted only when reached
            if not column_close(ca, ce, ctx):  # the kernel, or the library's equals
                return ctx.fail(  # first differing column: stop
                    lambda: f"{a.kind_name}s have different values: column {ca.name!r} differs"
                )
        return True
```

A table is a leaf for the engine and is reported as a whole (R5): "tables have a different structure",
or "tables have different values: column 'price' differs" (S1: the name of the first differing
column is free, since the loop knows it; a row is never looked up). A `Series` or an `Array` says
"series" or "arrays" (S3, the `kind_name` of its adapter). There is no row in the path: the user finds
the difference with the library (`DataFrame.compare`, `polars.testing`, `Table.equals`). The
comparison stops at the first column that differs, so the failing path is as cheap as the passing one.

| Kind       | Dtypes                                                                                                                                          | Comparison                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `numeric`  | integers, floats, nullable `Int64`/`Float64`, Arrow and polars numerics                                                                         | null masks equal, then the kernel on the values (`0` under a null)                          |
| `duration` | `timedelta64`, polars `Duration`, Arrow `duration`                                                                                              | kernel on integer nanoseconds                                                               |
| `exact`    | booleans, strings, categories, instants, periods, intervals, decimals, **and the nested (`List`, `Struct`, Arrow nested) and `object` columns** | the library's `equals` (a null equals a null); never sees the tolerance or `equal_nan` (Q6) |

The nested and `object` columns are the one place where the library, not coola, decides (Q6): the
tolerance is not applied to the floats inside them, and `NaN` equals `NaN` there (the `equals` of
pandas, polars and Arrow). The three libraries then agree with each other, which they do not today.

| Library | `structure()` is                                                               | Never used any more                                      |
| ------- | ------------------------------------------------------------------------------ | -------------------------------------------------------- |
| pandas  | type, shape, `index.equals` (values, dtype, names), `columns.equals`, `dtypes` | `pandas.testing`, copies of frames, `dt.total_seconds()` |
| polars  | `schema` (names, order, dtypes), height; a `LazyFrame` is collected            | `polars.testing`                                         |
| pyarrow | `type`, length; for a table `schema.equals(check_metadata=False)`, row count   | `equals` followed by a NumPy fallback                    |

A `Series` or an `Array` is a one-column table. pandas `attrs` and `flags` and Arrow schema metadata
are ignored by design. The dtype includes the time resolution (`timedelta64[s]` and `timedelta64[ns]`
differ, as in NumPy). Every comparison is a boolean array and `all()` or the library's `equals`, so
a failing comparison costs what a passing one does. Adding a library means writing one adapter of
about 40 lines.

## 8. Package layout

```
coola/equality/
  __init__.py       the public names of section 3
  interface.py      the five functions (build config and engine, run, wrap the result)
  config.py         Tolerance, EqualityConfig
  engine.py         Engine, MaxDepthExceededError, path parts (AttributeName, Element)
  result.py         ComparisonResult, bounded formatter
  _numeric.py       kernel, number_close, int_isclose, difference_close
  tester/
    base.py         BaseEqualityTester
    registry.py     EqualityTesterRegistry (unchanged); interface.py: default registry, lazy loading
    native.py       default, scalar, difference, sequence, mapping, ordered mapping, set, dataclass, attributes, equal
    numpy.py  torch.py  jax.py  xarray.py  pydantic.py
    table.py        Column, Table, TableEqualityTester
    pandas.py  polars.py  pyarrow.py     adapters
```

The registry, its lazy per-library registration and the plugin group `coola.equality.testers` stay
as they are. Library modules import their library lazily.

## 9. Evidence: a prototype of the engine

A throw-away script (not in the repository; about 70 lines: the engine, scalar, sequence, mapping
and default testers, no library, no `Decimal`; its loop is in the appendix). Equal but distinct
objects, best of 7 runs:

| Case                                | `main`  | Prototype         |
| ----------------------------------- | ------- | ----------------- |
| 20 000 floats, exact                | 15.5 ms | 5.1 ms            |
| dict of 5 000 `{float, [int, int]}` | 55.9 ms | 12.0 ms           |
| same dict, `explaining=True`        | n/a     | 12.0 ms           |
| deepest nesting without error       | 141     | 100 000+          |
| first difference reported           | n/a     | `('k42', 'y', 1)` |

`main` uses 6.4 Python frames per dict level and makes 830 177 function calls for the 5 000-entry
dict (30 003 `contextmanager` objects among them). The registry lookup costs 300 ns per element if
it is not cached per call: without the cache the same design takes 9.4 ms and 15.6 ms with a
recursive loop. A recursive variant of the design (a tester calls `ctx.compare`) is about 1.5x
faster than the iterative one (3.3 ms, 7.5 ms) but reintroduces the interpreter stack limit, a depth
counter and a `try/finally` in every container; it is the fallback if a benchmark gate cannot be
met. The prototype bounds the gain of the engine, not of the whole rewrite.

## 10. Changes to reach the target state

### 10.1 What differs today, and the change

| #   | Current behaviour on `main`                                                                                                                                                                                                                                                                                                                                   | Change                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| C1  | **[verified]** `ac(np.array([2**62]), np.array([2**62 + 1]), atol=0.5, rtol=0)` is `True` (torch, pandas, polars, pyarrow, `uint64` too); a list and a Python `int` give `False`                                                                                                                                                                              | integer difference in integer arithmetic (section 6)                                                                     |
| C2  | **[verified]** `ac(IntEnum.A, IntEnum.B, atol=5)` and `IntFlag` members are `True`; a plain `Enum` is `False`                                                                                                                                                                                                                                                 | `ScalarEqualityTester` is exact for `(bool, Enum)`                                                                       |
| C3  | **[verified]** pandas `Series`, `DataFrame`, `Index`, `"boolean"` arrays: `True` is close to `False` with `atol=1`                                                                                                                                                                                                                                            | the `exact` column kind                                                                                                  |
| C4  | **[verified]** nested cells: polars applies the tolerance inside a list, pyarrow does not, pandas uses its own rule and ignores `equal_nan` (`eq(pd.Series([[nan]]), same, equal_nan=True)` is `False`)                                                                                                                                                       | the `exact` kind: nested cells are compared by the library's `equals`, with no tolerance (Q6); the three libraries agree |
| C5  | **[verified]** a failing exact comparison of two pandas series of 2M floats takes 5 898 ms (equal copy: 2.9 ms; `pd.testing.assert_series_equal` alone: 5 804 ms); a 200k x 10 frame: 538 ms                                                                                                                                                                  | no `pandas.testing`; boolean arrays                                                                                      |
| C6  | **[verified]** pandas `object` column: two distinct `NaN` are equal without `equal_nan`; `None` equals `NaN` but `hash_object` differs                                                                                                                                                                                                                        | none in equality (Q6: the column stays with the library); the hasher is aligned later (last section)                     |
| C7  | **[verified]** `MappingProxyType` and `array.array` fall to `==`: no tolerance, a `mappingproxy` holding arrays raises `TypeError`                                                                                                                                                                                                                            | register both (two lines)                                                                                                |
| C8  | **[verified]** `float16`/`float32` arrays at extreme values: `RuntimeWarning` and a wrong `True` (`60000` against `-60000`, `rtol=1.5`); a Python float gives `False`                                                                                                                                                                                         | cast to `float64` in the kernel                                                                                          |
| C9  | **[verified]** masked arrays: the data under the mask is compared, so arrays that print identically differ; `hash_object` ignores it                                                                                                                                                                                                                          | **none (Q5)**: kept and documented (the hash ignores it, which keeps "equal implies same hash")                          |
| C10 | **[verified]** `compare` gives an empty path for an array, a frame and an Arrow table; a set element is shown as an index (`actual[2]`)                                                                                                                                                                                                                       | the index in the path of an array (S2, kept); `Element` for sets; a table stays one leaf (R5)                            |
| C11 | **[verified]** `show_difference=True` is silent by default, exists on two functions, and there are 50 occurrences and 39 `report_difference` calls behind `if config.show_difference`                                                                                                                                                                         | one first-failure record; remove the option and the logger                                                               |
| C12 | **[verified]** `compare` on 1M floats differing at index 0 takes 231 ms (`objects_are_equal`: 10 ms): it builds the `repr` of the roots                                                                                                                                                                                                                       | bounded reasons; drop the root `repr`s                                                                                   |
| C13 | **[verified]** 6.4 frames per dict level; a 99-level dict fails with `RecursionError` from a 300-frame caller; 830k calls for 5 000 entries                                                                                                                                                                                                                   | the engine (section 4.2)                                                                                                 |
| C14 | **[reading]** 50 handler classes, a mutable linked list with 500 lines of machinery (`handler/base.py` 375, `mixin.py` 123), `SameObject`/`SameType` copied into 29 testers                                                                                                                                                                                   | tester method; the engine owns both checks                                                                               |
| C15 | **[reading]** `EqualityConfig` holds settings and the state of one run, the registry is passed twice, `compare` takes a `Tolerance` and the others `rtol`/`atol`                                                                                                                                                                                              | frozen config; one `Engine` per call; `compare(rtol, atol)`                                                              |
| C16 | **[reading]** pandas 503, polars 368, pyarrow 172 lines: three implementations of one algorithm                                                                                                                                                                                                                                                               | `TableEqualityTester` and three adapters                                                                                 |
| C17 | **[verified]** docs: the pandas paragraph says a frame with a missing value is never equal to its copy (false); Design Principles lists `show_difference`; `extending.md:139` raises `ImportError` in `python -m doctest` (`get_default_registry` is in `coola.equality.tester`); the `objects_are_equal` docstring Note is the one of `objects_are_allclose` | fix the four statements                                                                                                  |
| C18 | **[reading]** `coola.equality.handler` exports 50 names, `tester` 36; `coola.io`, `coola.hashing.registry`, `coola.registry.base` import the private path `coola.equality.interface`                                                                                                                                                                          | about 25 public names; import from `coola.equality`                                                                      |
| C19 | **[verified]** no property compares the same random numbers in a list, an array, a tensor, a series and an Arrow array; library types appear in hashing and summary properties only                                                                                                                                                                           | cross-container property tests (step 1)                                                                                  |

Already true on `main` and kept: strict type, order rules, signed zero, time zones, the unregistered
fallback, co-inductive cycles, thread safety of `compare` (8 threads x 200 calls), reflexivity of
pandas frames with nulls, `compare` costing the same as `objects_are_equal` on equal data, the
elementwise symmetric kernel `handler/elementwise.py` (the base of `_numeric.py`), the registry and its
lazy loading.

### 10.2 Steps

One PR per row, as `CLAUDE.md` asks. "B" marks a breaking change. Steps 1 to 8 do not touch the handler
API and fix wrong answers; steps 9 to 14 move the architecture one family at a time, with `main` green
after each.

| Step | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Closes             | B   | Depends on |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | --- | ---------- |
| 1    | **Cross-container property tests**: the same random `list[float]` and `list[int]` in a list, NumPy, torch (float64, int64), pandas, polars, pyarrow, xarray give the same answer for random `rtol`, `atol`, `equal_nan`; a copy is equal; the swap is symmetric; `xfail` for C1, C3, C4, C6 until fixed                                                                                                                                                                     | C19                |     |            |
| 2    | Enum members exact in the number tester; pandas booleans exact (`Series`, `DataFrame`, `Index`, extension arrays)                                                                                                                                                                                                                                                                                                                                                           | C2, C3             | B   | 1          |
| 3    | Kernel: exact integer difference, cast of small floats to `float64`, `over` silenced; applies to NumPy, torch, JAX and the numeric columns of frames                                                                                                                                                                                                                                                                                                                        | C1, C8             |     | 1          |
| 4    | Register `types.MappingProxyType` and `array.array`; regression test with arrays inside a `mappingproxy`                                                                                                                                                                                                                                                                                                                                                                    | C7                 |     |            |
| 5    | Fix the four documentation statements; find why the Markdown doctest job did not stop on `extending.md`                                                                                                                                                                                                                                                                                                                                                                     | C17                |     |            |
| 6    | ~~Masked arrays~~ dropped by Q5 (documentation only: say that the masked values are compared)                                                                                                                                                                                                                                                                                                                                                                               | C9 (doc)           |     |            |
| 7    | pandas on column kinds inside the current handler (no `pandas.testing`; the `object` and nested columns through `Series.equals`); failing-path benchmark within 2x of the passing path                                                                                                                                                                                                                                                                                      | C4, C5, C6         | B   | 1, 3       |
| 8    | polars and pyarrow on column kinds: nested columns through `equals` (polars loses the tolerance inside nested cells)                                                                                                                                                                                                                                                                                                                                                        | C4                 | B   | 1, 3       |
| 9    | **Engine**: add `Engine`, the lookup cache, the explicit stack, identity/type/depth/cycle ownership. The existing chain testers are called as leaf testers while the families are ported, to keep each PR small and the benchmarks comparable; this staging is removed by step 14 and is not a compatibility layer                                                                                                                                                          | C13, part C14, C15 |     | 1          |
| 10   | One reporting mechanism: `report_difference` forwards to `ctx.fail`; first-failure record; lazy bounded reasons; remove `show_difference`, the logger sink, `compare(tolerance=, max_repr_chars=)` and the root fields of the result                                                                                                                                                                                                                                        | C11, C12, C15      | B   | 9          |
| 11   | Port the Python testers (object, scalar, sequence, mapping, set, dataclass, attributes, pydantic) from chains to `objects_are_equal` returning a verdict or children; a port reports its location; delete the handlers nobody uses                                                                                                                                                                                                                                          | C14                | B   | 10         |
| 12   | Port the array testers (NumPy, masked, torch, JAX, xarray): index in the path, bounded value reasons; the library testers stay public (Q12)                                                                                                                                                                                                                                                                                                                                 | C14, C10           | B   | 11         |
| 13   | `TableEqualityTester` and the three adapters replace the frame handlers; a table is one leaf (R5); `Element` for sets                                                                                                                                                                                                                                                                                                                                                       | C16, C10           | B   | 11, 7, 8   |
| 14   | Delete the chain machinery (`BaseEqualityHandler`, mixin, `create_chain`, `BaseChainEqualityTester`, `HandlerEqualityTester`, `check_recursion_depth`, `track_comparison`); freeze `EqualityConfig`; library code to private modules; trim `__all__`; `coola.io` and the others import from `coola.equality`; a migration guide (`docs/docs/migration/`, Q11) and a migration guide in `docs/docs/migration/` (Q11); rewrite `equality_handler.md` and `equality_tester.md` | C14, C15, C18      | B   | 12, 13     |

Steps 11 to 14 break every custom handler and tester (`handle(actual, expected, config)` and
`objects_are_equal(actual, expected, config)` become `objects_are_equal(actual, expected, ctx)` that
returns a verdict or the child pairs, with `config.registry.objects_are_equal(a, b, config)` replaced
by returning `(a, b, part)`). They share the major release; the migration guide shows that change with a before/after custom tester.

### 10.3 Tests and gates

- **Unit tests** move with the code: the handler tests become tester tests, one behaviour per test,
  one `assert`. The engine has its own tests (identity, type, cycle, depth, order of the first
  difference, path, lazy reasons, a tester that returns a bool and one that returns an iterator).
- **Properties** (step 1) are the safety net of the migration. Add: reflexive for a copy of each
  library type with and without nulls, symmetric for each library type, `equal` implies equal hash.
- **Benchmarks** in `tests/benchmarks/test_equality_benchmark.py`, added **before step 9**, fail on a
  regression above 10%: 20 000 floats in a list 15.5 ms; dict of 5 000 `{float, [int, int]}` 55.9 ms
  (targets after step 11: 6 ms and 15 ms); `compare` of 1M floats differing at index 0 231 ms
  (target below 30 ms after step 10); two pandas series of 2M floats differing at index 0 5 898 ms
  (target below 10 ms after step 7); a 200k x 10 frame differing in one cell 538 ms (target below
  10 ms after step 7). Add a depth test: 100 nested dicts, lists, dataclasses and pydantic models from
  a caller 500 frames deep, and 100 000 nested dicts with `max_depth` raised.
- **Docs**: the guide's Guarantees table takes the wording of section 1; "Finding a Difference" uses
  `compare`; the NaN, tolerance and Design Principles sections follow sections 5 to 7.

## 11. Decisions for the maintainers

Each choice has the options, what each one costs, and a recommendation. "Decision" is left empty for
the maintainers; the steps of section 10.2 that depend on a choice are named.

### Q1. Engine: iterative or recursive (steps 9, 11)

| Option       | For                                                                                          | Against                                                                                                                             |
| ------------ | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| A. Iterative | No `RecursionError`, explaining is free, container testers are pure functions of two objects | About 1.5x slower than B on the happy path (still 3 to 5x faster than `main`); a tester returns an iterator, which is less familiar |
| B. Recursive | Fastest; testers call `ctx.compare`, the familiar shape                                      | Interpreter stack limit and `max_depth` stay; depth, cycle and path need a `try/finally`                                            |

Recommendation: A. Decision: A

### Q2. Tester return protocol (step 9)

| Option                                                                          | Note                                                                    |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| A. One method returns `bool` or an iterator of children                         | One name, no flag; the engine tests the type of the result              |
| B. Two methods, `equal` (leaf) and `expand` (container) plus a `container` flag | Explicit and type-checkable; two entry points to document and to forget |
| C. Always an iterator (a leaf yields nothing and fails through `ctx.fail`)      | One shape; leaves pay for a generator                                   |

Recommendation: A. Decision: A

### Q3. `show_difference` and the logger sink (step 10)

**Facts.** `show_difference=True` exists on `objects_are_equal` and `objects_are_allclose` only. It
logs at `INFO` on the `coola.equality` logger, which is silent with the default logging
configuration (the guide tells the user to call `logging.basicConfig(level=logging.INFO)`). It is
mentioned in 21 lines of the documentation, once in the README, in 4 lines of `examples/`, and used
in about 547 lines of tests. In `src` it costs 50 occurrences, 39 `report_difference` calls, each
behind an `if config.show_difference:` that every tester must remember, a `reporter` callback that
`compare` installs, and a logger path that formats the message **before** the logger can drop it.
Under the constraints (first difference only, on demand, no formatting when nobody asks), the
explanation has one natural home: `compare` and `assert_*`.

| Option                                                                | Pros                                                                                                                                                                                                                                                                                                                                                                                                                                                | Cons                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Remove it** (`compare` and `assert_*` explain)                   | One way to get an explanation, with a structured result (`path`, `reason`) instead of log text. No `if show_difference` guard in testers, no `reporter`, no logger, no global state, so the engine has one failure record and the entry points have the same signature. `objects_are_equal` is guaranteed to format nothing. Thread-safe by construction. Output appears by default (`assert_*` raises, `compare` returns) instead of being silent. | Breaking: every call site `objects_are_equal(a, b, show_difference=True)` must become `compare(a, b)` or `assert_objects_equal(a, b)` (a one-line change, but about 25 places in docs and examples and the README to rewrite). A user who wants "the answer and a log line in one call" now writes two lines (`log.info(compare(a, b).format_message())`). Tests that check log output (a large part of the 547 lines) are rewritten as tests of `ComparisonResult`. |
| **B. Keep it on the five functions, implemented on top of `compare`** | Existing call sites and docs keep working; `show_difference` becomes a thin wrapper: run the explaining engine, log `format_message()`, return the bool. Still one engine path and one failure record; the `reporter`, the guards and the handler-side logging disappear. Could also be added to `assert_*` for symmetry.                                                                                                                           | Two ways to ask for an explanation, forever (the second one silent by default). `objects_are_equal` is no longer guaranteed to be free of formatting: with `show_difference=True` it explains, so the fast path and the explaining path share a signature and the "build no message" rule needs an exception. A logging side effect in a comparison function (threads, handlers, test log capture).                                                                  |
| **C. Keep it as it is**                                               | No work, no break.                                                                                                                                                                                                                                                                                                                                                                                                                                  | Keeps the 50 occurrences, the guards that testers can forget (a forgotten guard formats on every call, a missing report loses the explanation), the two sinks and the reversal of reports in `compare`; contradicts "first difference only, on demand" and costs the failure path (`compare` of 1M floats: 231 ms against 10 ms).                                                                                                                                    |
| **D. Replace it by `explain=` taking a callable** (`explain=print`)   | Lets a caller route the explanation (print, logger, collector) in one call without a global logger; no silent default.                                                                                                                                                                                                                                                                                                                              | A new parameter to design, test and document for a use that `compare` already covers; a callback invoked inside a comparison (re-entrancy, exceptions); still needs the explaining engine.                                                                                                                                                                                                                                                                           |

**Recommendation: A.** It is the only option that removes the guards and makes "nothing is formatted
unless asked" true by construction; the break is mechanical and the release is a major one. B is the
fallback if the maintainers want the old call sites to survive one release, but it contradicts the
"no compatibility layer" constraint. Decision: A

### Q4. Shape of `ComparisonResult` and of the entry points (step 10)

Three independent sub-decisions.

**Q4a. Do `ComparisonResult.actual` and `.expected` (the `repr` of the roots) stay?**

**Facts.** Today they are `truncate_str(safe_repr(root), max_len=max_repr_chars)` of the **whole**
objects, built on every failure (the 231 ms of `compare` on 1M floats, and 700 ms for a dataclass
holding a dataclass holding the list). The caller already holds the roots, and the root `repr` does not
say where the difference is; the message already contains the two **differing values**. They are used
in about 21 lines of `src`, tests and docs together.

| Option                                                                          | Pros                                                                                                                                                                                                                                     | Cons                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Drop them and `max_repr_chars`** (reason has the two values)               | The failure path no longer depends on the size of the roots (honours "performance" and "first difference only"); a smaller result (`equal`, `path`, `reason`); one bounded formatter for the values in the reason; no parameter to tune. | Breaking for code that reads `result.actual`. A user who liked seeing the whole object in the assertion message loses it (the path and the differing values are shown instead, as pytest does for a long list). |
| **B. Keep them, truncated, computed lazily** (properties built on first access) | Compatible; the cost is paid only by readers; `assert_*` messages do not include them, so the common path stays cheap.                                                                                                                   | A result must keep references to the roots (memory, and mutated objects give a different `repr` later); lazy properties on a frozen dataclass are awkward; still two ways to see a value.                       |
| **C. Keep them, computed eagerly (today)**                                      | No change.                                                                                                                                                                                                                               | Keeps the 20x cost of the failure path and the dependency on the size of the roots; contradicts the performance constraint.                                                                                     |

Recommendation: A. Decision: A

**Q4b. How does `compare` take a tolerance?**

**Facts.** `objects_are_allclose` and `assert_objects_allclose` take `rtol` and `atol` (the NumPy and
torch names, with the defaults `1e-5` and `1e-8`); `compare` takes `tolerance: Tolerance | None`
(exact by default). The config stores two floats. So a user moves between three shapes for one
concept.

| Option                                                  | Pros                                                                                                                                                                                                     | Cons                                                                                                                                                                                                                           |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **A. `compare(rtol=0.0, atol=0.0)`**                    | The five functions have the same keywords; a call can be moved from `objects_are_allclose` to `compare` by renaming; no import of `Tolerance` for the common case; validation in one place (the config). | Breaking for `compare(a, b, tolerance=Tolerance(...))`. Exact is `0.0`, allclose is `1e-5`/`1e-8`: `compare(a, b, rtol=..., atol=...)` does not share the defaults of `objects_are_allclose`, which can surprise (documented). |
| **B. Keep `tolerance: Tolerance \| None` on `compare`** | No break; a `Tolerance` is a reusable, validated, hashable value (`Tolerance.numpy_default()`), good for a test suite that shares one tolerance.                                                         | Two shapes in the public API; the other four functions cannot take it, or must also accept it (then every function has three ways to give a tolerance).                                                                        |
| **C. All five take `tolerance: Tolerance`**             | One shape everywhere; a `Tolerance` is validated once.                                                                                                                                                   | Unlike every array library (`rtol=`, `atol=`); `objects_are_allclose(a, b, rtol=1e-3)` is the call people know and write; breaks the most used function.                                                                       |

Recommendation: A. Decision: A

**Q4c. Is `Tolerance` public?**

| Option                                                   | Pros                                                                                                         | Cons                                                                                                                        |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **A. Public**, the value held by `EqualityConfig`        | Custom testers and a user-built `EqualityConfig` read `ctx.config.tolerance`; documented and tested already. | One more exported name that a caller of the five functions never needs.                                                     |
| **B. Private** (`config` keeps `rtol` and `atol` fields) | Smaller public surface; no second tolerance concept.                                                         | A config has two loose floats again, validated twice; custom code that wants to pass a tolerance around has no type for it. |

Recommendation: A. Decision: A

### Q5. Masked arrays (step 6)

A. Compare the mask and the unmasked values, ignore the data under the mask (the hasher already does;
two arrays that print the same are equal). B. Keep comparing the data under the mask. Also decide
whether `fill_value` stays compared (recommended: yes, it is visible through `filled()`).
Recommendation: A. Decision: B

### Q6. Nested and `object` cells of frames (steps 7, 8, 13)

A. Compare each cell through the engine: one rule per value (`None` is not `NaN`, a nested float gets
the tolerance), slower; keep the library's `equals` as the fast path when there is no NaN and no
tolerance. B. Leave nested cells to the library (fast, three rules, wrong `equal_nan`). C. Forbid
nested cells (raise). Recommendation: A. Decision: B

### Q7. Time resolution is part of the dtype (step 13)

`timedelta64[s]` and `timedelta64[ns]` differ, as in NumPy. pandas 3 infers different resolutions
for values built in different ways, so two frames of the same instants can differ. A. keep strict
(document the `astype` fix); B. cast both to the finer resolution before comparing. B is friendlier
and breaks the "same dtype" rule of the guide. Recommendation: A. Decision: A

### Q8. Precision policy of the kernel (step 3)

A. Cast `float16`, `bfloat16`, `float32` to `float64` (correct at the edge of the range, memory x2 for
the temporaries); B. keep the dtype (fast, wrong at extremes, warns). Integers: C. exact difference,
tolerance in `float64` (recommended), or D. exact `Fraction` arithmetic for every integer array (exact
everywhere, much slower). Recommendation: A and C. Decision: C

### Q9. What stays ignored (no step)

`requires_grad`, pandas `attrs` and `flags`, Arrow schema metadata, `pydantic` private attributes,
LazyFrame query plan (collected). Options: keep ignored and document each (recommended), or add
opt-in flags. Decision: keep ignored and document each

### Q10. Strictness of enums and labels (step 2)

A. `bool` and every `enum.Enum` member are compared exactly, tolerance never applies (recommended).
B. Also `str`, `bytes` and every non-numeric type through an explicit "label" list. Decision: A

### Q11. Removal of compatibility code (steps 9 to 14)

Decided by the constraints: major release, no adapter, no alias, no deprecation period, deprecated and
compatibility code removed. Remaining question: should the release notes carry a before/after
example of a custom tester? Recommendation: yes, five lines. Decision: yes, create a migration guide

### Q12. Public surface (step 14)

About 25 names in `coola.equality`; the library testers are private. Options: A. private library
modules reached through the registry (recommended); B. keep the named library testers public
(`NumpyArrayEqualityTester`, ...) so that users can compose or subclass them. Also decide the
public name of the engine type (`Engine`, `ComparisonContext`) and whether `Element` and
`AttributeName` are exported. Decision: B

### Q13. Registry lookup cache (step 9)

A. Per-call `dict` (recommended: no lock in the hot path; a registration during a call is seen by
the next call). B. A lock-free immutable snapshot inside the registry (benefits `hashing` and
`summary`, which share the registry class, but changes a shared component). Decision: A

### Q14. Size of the first engine PR (step 9)

A. Engine plus the current chain testers called as leaves during the port (recommended: small PRs,
benchmarks can compare; removed in step 14). B. Engine and the port of the Python testers in one PR (fewer
intermediate states, a large diff). Decision: A

### Decisions taken and their effect on the design

| Q   | Decision                                                                                              | Effect, already applied in the sections above                                                                                     |
| --- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | Iterative engine                                                                                      | sections 2, 4                                                                                                                     |
| Q2  | One method returns a verdict or an iterator                                                           | section 4.1                                                                                                                       |
| Q3  | Remove `show_difference` and the logger                                                               | section 3, steps 10                                                                                                               |
| Q4  | Drop root `repr`s; `compare(rtol, atol)`; `Tolerance` public                                          | section 3                                                                                                                         |
| Q5  | Masked arrays: keep comparing the data under the mask                                                 | masked row of 4.4; step 6 dropped; C9 closed                                                                                      |
| Q6  | Nested and `object` cells of frames left to the library                                               | section 7: three kinds only, no cell children, the tolerance and `equal_nan` do not reach nested cells; C4 reworded; C6 left open |
| Q7  | Time resolution stays in the dtype                                                                    | section 7                                                                                                                         |
| Q8  | Integers: exact difference, tolerance in `float64`                                                    | section 6; small floats cast to `float64` (R2)                                                                                    |
| Q9  | Ignored state stays ignored and is documented                                                         | guide sections                                                                                                                    |
| Q10 | `bool` and every `Enum` exact                                                                         | section 4.3                                                                                                                       |
| Q11 | A migration guide in `docs/docs/migration/`                                                           | step 14                                                                                                                           |
| Q12 | Library testers stay public                                                                           | section 3 exports; open details in R3                                                                                             |
| Q13 | Per-call lookup cache                                                                                 | section 4.2                                                                                                                       |
| Q14 | Engine PR with chain testers called as leaves                                                         | step 9                                                                                                                            |
| R1  | Document the `NaN`-in-nested-cell exception                                                           | property rows of section 1                                                                                                        |
| R2  | Cast small floats to `float64`                                                                        | section 6                                                                                                                         |
| R3  | `TableEqualityTester(adapter)` only, `Engine`, export both path parts, helpers private except `close` | section 3                                                                                                                         |
| R4  | Keep `max_depth=100` and `MaxDepthExceededError`                                                      | section 4.2                                                                                                                       |
| R5  | A table is one leaf, reported as different, no row (the column name comes from S1)                    | sections 5, 7                                                                                                                     |
| S1  | The reason of a table names the first differing column (never a row)                                  | section 7                                                                                                                         |
| S2  | Arrays keep their index in the path                                                                   | sections 4.3, 5                                                                                                                   |
| S3  | A `Series` says "series", an `Array` "arrays" (`kind_name`)                                           | section 7                                                                                                                         |
| H1  | The hasher of masked arrays is left as it is (may change later)                                       | last section                                                                                                                      |

### Follow-up questions

**R1. (answered by the principle that equality drives hashing.)** With nested and `object` columns left
to the library, `None` against `NaN` in a pandas `object` column are equal but have different
`hash_object`s, and `NaN` in a nested cell equals `NaN` without `equal_nan`. The equality rule is not
changed for the hash: the hasher is aligned later (last section). The remaining question is only
whether the `NaN`-in-nested-cell exception is documented in the Guarantees table, as part of the Q6
wording.

Recommendation: yes, document it.
Decision: yes, document it

**R2. Do small floats (`float16`, `bfloat16`, `float32`) keep their dtype in the kernel?** Q8 chose the
integer rule only. Keeping the dtype means a `RuntimeWarning` and a wrong `True` at the edge of the
range (`60000` against `-60000`, `rtol=1.5`); casting to `float64` costs temporaries twice as large.
Options: A. cast to `float64` (correct, simple); B. keep the dtype and handle overflow
explicitly (`errstate`, an `isfinite` check on the tolerance), which is faster for large `float32`
arrays and adds branches to the kernel.

Recommendation: A.
Decision: A

**R3. Details of the public surface (Q12 B).** Library testers stay public, so decide: (a) the names
for the pandas, polars and pyarrow testers now that one `TableEqualityTester` replaces the handlers:
keep `PandasDataFrameEqualityTester` and friends as thin subclasses that fix the adapter (more names,
familiar), or expose `TableEqualityTester(adapter)` only (fewer names; "no alias" constraint favours
this); (b) the public name of the type of `ctx` (`Engine` or `ComparisonContext`); (c) whether
`Element` and `AttributeName` are exported (they appear in `ComparisonResult.path`); (d) whether the helper functions that custom testers need (`comparison_key`, `describe_values`,
the kernel `close`) are public.
Recommendation: (a) `TableEqualityTester(adapter)` only, (b) `Engine`, (c) export both path parts,
(d) keep the helpers private except the kernel `close`.
Decision: (a) `TableEqualityTester(adapter)` only, (b) `Engine`, (c) export both path parts,
(d) keep the helpers private except the kernel `close`.

**R4. Is `max_depth` still needed?** With the explicit stack there is no `RecursionError`, and cycles
are handled, so `max_depth` no longer protects the interpreter. It still bounds the work and memory
on a pathological, non-cyclic structure (an object whose attribute creates a new child on each
access). Options: A. keep `max_depth=100` and `MaxDepthExceededError` (documented today, cheap, the
error no longer derives from `RecursionError`); B. remove the parameter and the error (one option
less on five functions, one validator less; an infinite generated structure runs until memory is
exhausted); C. keep it with a larger default (for example 1 000) because 100 is a legitimate nesting
for JSON-like data.
Recommendation: A, with C if a real structure goes beyond 100.
Decision: A

**R5. How are tables reported?** For an `exact` column the
library's `equals` returns a bool. The path is then `(column,)` and the reason names the column, or the
row is found by a second, explaining-only pass (`a != b` on the column; `NaN`-aware for floats). The
second is more helpful and costs nothing on the passing path and on the failing path of
`objects_are_equal`.
Recommendation: the explaining-only pass for pandas, polars and pyarrow, and the column alone when
it is not available (nested columns).
Decision: I think it does not need to give the exact row or column. I would suggest to keep things simple and efficient. It is okay to say the DataFrame are different. Then, the user can use a library specific function to find the difference between the DataFrames.

**S1. Does a table report anything more than "different"?** R5 keeps tables as one leaf. The loop
over the columns knows the name of the first column that differs at no cost, so the reason could say
`column 'price' differs`. Options: A. only "tables have different values"; B. add the column name
(free, no extra pass, helps a frame of 200 columns), but only the name, never a row.

Recommendation: B.
Decision: B. Anything that is free is fine.

**S2. Do arrays keep their index in the path?** For NumPy, torch and JAX the kernel returns a boolean
array and the first `False` gives the index (`np.argmin`), looked up only when explaining, at no cost
on the passing path or on `objects_are_equal`. Options: A. keep the index (`actual['w'][1][2]`);
B. like tables, report the array as one leaf ("arrays have different values"), simplest.

Recommendation: A.
Decision: A

**S3. Is a `Series` a table?** The adapters treat a `Series` or an `Array` as a one-column table, so a
failing `Series` says "tables have different values". Options: A. keep it (one tester, no special
case); B. give a `Series` its own wording ("series have different values") through the adapter's
name for itself (a `kind_name` string), still one tester.

Recommendation: B.
Decision: B

**T1. Which release carries the breaking steps?** Steps 2, 7, 8 and 10 to 14 are breaking; the other
steps are not. Options: A. every step lands in the next major release branch, in the order of
section 10.2, and nothing is released until step 14 (one big break, a long-lived branch);
B. the non-breaking steps (1, 3, 4, 5) are released in the current minor releases, the breaking ones
wait for the major release, in order; C. all steps are released as they are merged, each breaking
one with a version bump.

Recommendation: B.
Decision: A

**T2. Are the benchmark gates a failing CI check?** `ci-benchmark.yaml` exists. The gates of
section 10.3 (regression above 10%, and the targets after steps 7, 10, 11) can fail the build, or be
reported only. Failing builds on timing are flaky on shared runners.
Options: A. fail the build on a regression above 10% against the stored baseline of `main`, with a
generous margin for the absolute targets; B. report only, and check by hand at each step.

Recommendation: A for the relative gate, B for the absolute targets.
Decision: B

**T3. Who and when for the hashing follow-up?** The last section lists the hasher changes. Options:
A. a separate design document and PRs after step 14; B. in parallel, after step 1 (the properties
exist), so that the cross-container properties also test the hash from the start.

Recommendation: B.
Decision: A

## 12. Not covered

- Per-library corner cases beyond the probes: sparse and quantized torch tensors, nested tensors,
  dask-backed xarray, pandas `Interval`, `Period` and `Sparse` columns on the `exact` path.
- The GPU path, Windows, fork behaviour.
- A registration made while a comparison runs is seen by the next call only (the lookup cache is per
  call).
- The prototype has no `Decimal` and no library tester. The package-size estimate (about 4k lines
  instead of 8.6k, unit tests from 20.5k to about 12k lines) is an extrapolation, not a measurement.
- Mutation testing of the package.

## Appendix: the engine loop

```python
def run(self, a, e):  # Engine.run
    stack, keys, parts, visiting = [], [], [], set()  # child iterators, cycle keys, path parts
    pair = (a, e, None)
    while True:
        a, e, part = pair
        ok = True
        if a is not e:
            t = type(a)
            if t is not type(e):
                ok = self.fail("types differ")
            else:
                result = self.find(t).objects_are_equal(a, e, self)  # a verdict, or children
                if result is True or result is False:
                    ok = result
                elif (key := (id(a), id(e))) not in visiting:  # a pair on the stack is equal
                    if len(stack) >= self.max_depth:
                        raise MaxDepthExceededError
                    visiting.add(key)
                    stack.append(result)
                    keys.append(key)
                    parts.append(part)
        if ok is False:
            return self.record_failure(
                parts, part
            )  # path = parts[1:] + (part,) + the parts of fail()
        while True:  # next pair: the top iterator, popping the exhausted ones
            if not stack:
                return True
            if (nxt := next(stack[-1], None)) is not None:
                pair = nxt
                break
            stack.pop()
            visiting.discard(keys.pop())
            parts.pop()
```

The measured prototype uses a `container` flag on the tester instead of testing the type of the
result; its output on the 5 000-entry dict, on cycles and on `max_depth` is the one of section 9.

## Changes in other sub-packages (later, outside this design)

The design above is for `coola.equality` only. Equality is the reference: the changes below follow
from it and are made afterwards, one PR each, without driving any choice made above. They are listed
so that nothing is forgotten.

### Hashing (`coola.hashing`)

The property "equal objects have the same `hash_object`" has to hold with the new equality. What the
design implies for the hashers:

| Case                                                  | Equality (target)                       | Hasher today / change                                                                              |
| ----------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------- |
| pandas `object` column, `None` against `NaN`          | equal (the library's `equals`)          | hashes differ **[verified]**: hash a `None` and a `NaN` of an `object` column the same             |
| `NaN` inside nested cells (lists, structs) of a frame | equal without `equal_nan` (the library) | check the hash of nested `NaN`: it must not depend on the NaN bit pattern                          |
| Nested floats of a frame, tolerance                   | exact (no tolerance)                    | nothing to do: equal values have equal hashes                                                      |
| Masked array, data under the mask                     | compared (Q5)                           | the hasher ignores it; allowed (different values may share a hash); decide if it should include it |
| Integers above 2^53 in arrays, `bool` and enums       | exact                                   | check that the hash is exact (no `float64` conversion), and the same for `bool` and `Enum` members |
| Enum members, `IntEnum` against `int`                 | different types                         | the hash of an `IntEnum` member may equal the hash of its `int`: allowed                           |
| `MappingProxyType`, `array.array`                     | compared as a mapping and a sequence    | check they have a hasher consistent with a `dict` and a `list` of the same content                 |
| Time resolution (`timedelta64[s]` against `[ns]`)     | different dtypes                        | allowed to share a hash                                                                            |

The hashing property tests (`tests/properties/test_hashing*.py`) get the cross-container equality
properties of step 1 as a second source: any equal pair that the equality tests generate must hash
equally. Where the hasher needs a rule that is not obvious, the rule is taken from the equality
guide, not the other way round.

**H1. Should the hash of a masked array include the data under the mask?** Equality compares it (Q5);
the hasher ignores it today, which keeps "equal implies same hash" true. Including it makes the hash
as discriminating as equality, at the cost of changing every existing hash of a masked array.

Recommendation: no, leave the hasher as it is (the property holds).
Decision: no, leave the hasher as it is. It may change later.

### Packages that depend on the equality API

| Package                                                                     | Change                                                                                                                                                                                                                                                |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.io`, `coola.nested`, `coola.hashing.registry`, `coola.registry.base` | import `objects_are_equal` from `coola.equality`, not from the private `coola.equality.interface` (C18)                                                                                                                                               |
| `coola.registry` (equality of two registries)                               | testers are compared with their `__eq__` (the `equal(other)` method of `BaseEqualityTester` is removed): the default registry registers the testers and handlers types of the removed chain with the default `==` tester, or no longer registers them |
| `coola.summary`, `coola.iterator`, `coola.recursive`, `coola.reducer`       | no change expected; check that none imports a removed handler or `HandlerCycleError`                                                                                                                                                                  |
| `coola.testing`                                                             | fixtures and helpers that use `show_difference` or the handler chain move to `compare` and `assert_*`                                                                                                                                                 |
| Documentation                                                               | `docs/docs/uguide/hashing.md` states the property against the equality guide; `docs/docs/dev/add_backend.md` loses the handler step (one tester, one adapter); README and `index.md` drop `show_difference`                                           |
