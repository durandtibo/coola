# `coola.equality`: target state and the changes to reach it

This document describes the design that `coola.equality` should have, then the changes to make to
the current implementation (`main`, `bb94c6fd`) to get there. The design follows the **Guarantees**
table of `docs/docs/uguide/equality.md`. Its aim is a small package that is easy to read: one engine,
one extension point, one numeric rule, one frame algorithm. Breaking changes are accepted.

Evidence comes from probes and benchmarks run on `main` (Python 3.14, numpy 2.5, torch 2.14, pandas
3.0, polars 2.0, pyarrow 26, jax 0.12, xarray 2026.9, pydantic 2.14) and from a throw-away prototype
of the engine kept outside the repository (section 9). Items marked **[verified]** were reproduced on
`main`. Nothing in `src`, `tests` or `docs` was changed.

## 1. Properties the design must satisfy

These are the rules of the user guide. Every design choice below exists to make one of them true by
construction instead of by the discipline of each tester.

| Property                | Statement                                                                                                                                                                                                                                                                                | Enforced by                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| Strict type             | Same exact type or different (`1`, `1.0`, `True`; `[1]`, `(1,)`; `dict`, `OrderedDict`; a subclass).                                                                                                                                                                                     | the engine, before any tester                 |
| Reflexive               | `x` equals `x` and a copy of `x`; the exception is a NaN-like value without `equal_nan`.                                                                                                                                                                                                 | the engine (identity); the missing-value rule |
| Recursive, order rules  | Containers item by item at any depth; sequences ordered; `dict` unordered; `OrderedDict` ordered.                                                                                                                                                                                        | container testers                             |
| Value semantics         | `==` unless a tester exists; arrays and frames also compare dtype and shape.                                                                                                                                                                                                             | the registry, the default tester              |
| Missing values          | A NaN-like value (float or complex `NaN`, `Decimal("NaN")`, NumPy `NaT`) equals another only with `equal_nan=True` or if it is the same object, **in every container**. A null (`None`, `pd.NA`, polars or Arrow null) equals a null, always.                                            | the numeric kernel and the table comparer     |
| Signed zero, time zones | `-0.0` equals `0.0`; aware datetimes compare by instant.                                                                                                                                                                                                                                 | kernel, `==`                                  |
| Tolerance               | `abs(a - b) <= atol + rtol * max(abs(a), abs(b))` per element, for every number in every container, exact for integers. `bool`, enum members, instants, strings and the structure are exact. Durations follow the tolerance (`atol` in seconds). Nested cells follow the rule of a list. | one kernel, one number tester                 |
| Symmetric               | `eq(a, b)` and `eq(b, a)` agree, with and without tolerance.                                                                                                                                                                                                                             | kernel formula, engine type rule              |
| Not transitive, pure    | A tolerance is not transitive; a comparison never modifies its inputs.                                                                                                                                                                                                                   | (documented)                                  |
| Failure modes           | Cycle: equal. Nesting above `max_depth`: `MaxDepthExceededError`. `==` that raises or returns a non-bool: `TypeError` naming the type; other exceptions propagate. No `RecursionError`, whatever the caller's stack.                                                                     | the engine (explicit stack)                   |
| Consistent with hashing | Equal objects have the same `hash_object`.                                                                                                                                                                                                                                               | the missing-value rule, the table comparer    |
| Location                | `compare` and `assert_*` give the path of the first difference: keys, positions, attributes, the column and row of a frame, the index in an array, the element of a set.                                                                                                                 | the engine's stack and `ctx.fail`             |

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
   equal. A frame is a container of columns, a nested column a container of cells.
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
testers a user composes for their own classes (`ObjectAttributesEqualityTester`,
`EqualEqualityTester`, `OrderedMappingEqualityTester`, `SequenceEqualityTester`,
`MappingEqualityTester`). About 25 names. Everything per library lives in private modules and is
reached through the registry.

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
engine knows (the index in an array, the row in a column).

### 4.2 The engine

One loop with a stack of child iterators (depth first, so the **first** difference in iteration
order is the one reported). The full loop is in the appendix. For each pair:

1. `actual is expected`: equal.
2. `type(actual) is not type(expected)`: different (the engine writes the reason).
3. Look the tester up in a `dict[type, tester]` filled from the registry on the first use of a type
   in the call (the registry lookup costs 300 ns, so it is paid once per type).
4. Call the tester. A `bool` is the verdict. An iterator is a container: if its pair is already on
   the stack the pair is equal (cycle); if the stack is `max_depth` deep, raise
   `MaxDepthExceededError`; otherwise push it and continue with its first child.
5. On `False`, the path is the path parts of the stack, the part of the failing pair and the parts
   given to `ctx.fail`. It is read once, here.

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

| Type                                                                        | Tester                                                   | Rule                                                                 |
| --------------------------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------- |
| `object` (unregistered)                                                     | `DefaultEqualityTester`                                  | `==`; a raise or a non-bool result is a `TypeError` naming the type  |
| `int`, `float`, `complex`, NumPy scalars                                    | `ScalarEqualityTester`                                   | kernel formula; `bool` and enums exact                               |
| `Decimal`, `Fraction`, `timedelta`                                          | `DifferenceCloseEqualityTester`                          | difference-based tolerance; `Decimal` NaN by `equal_nan`             |
| `list`, `tuple`, `deque`, `Sequence`, `array.array`, dict values            | `SequenceEqualityTester`                                 | length, then items in order                                          |
| `dict`, `Mapping`, `MappingProxyType`                                       | `MappingEqualityTester`                                  | same keys (by type, NaN by `equal_nan`), then values                 |
| `OrderedDict`                                                               | `OrderedMappingEqualityTester`                           | and the same key order                                               |
| `set`, `frozenset`                                                          | `SetEqualityTester`                                      | elements matched by hash and type; reports `Element`                 |
| dataclasses, pydantic models                                                | `DataclassEqualityTester`, `PydanticModelEqualityTester` | fields as `AttributeName` children                                   |
| `np.ndarray`, `np.ma.MaskedArray`, NumPy `datetime64`/`timedelta64` scalars | `Numpy*EqualityTester`                                   | dtype, shape, kernel; mask and unmasked values; `NaT` by `equal_nan` |
| `torch.Tensor`, `PackedSequence`, `jax.Array`, xarray                       | `Torch*`, `Jax*`, `Xarray*EqualityTester`                | device, dtype, shape, layout, kernel; xarray parts as children       |
| pandas, polars, pyarrow frames, series, arrays                              | `TableEqualityTester` (section 7)                        | structure, then columns by kind                                      |

## 5. Reporting

- One record: the first failure, `(path, reason)`. No list of reports, no reversal, no message per
  parent level (the path says where).
- A reason is a `str` or a callable and is built only when `explaining`. Values inside a reason go
  through one bounded formatter (`reprlib.Repr` with limits; the array libraries summarise
  themselves), so a failure on 1M elements formats a few values, not the objects.
- `ComparisonResult` carries `equal`, `path` and `reason`. The reason holds the two differing values;
  the roots are not repeated.
- Path parts: a key or a position is itself (`['a']`, `[3]`), `AttributeName("x")` is `.x`,
  `Element(2)` is `{2}`. A frame reports `("price", 3)` (column, row position), an array `(1, 2)`.

```
objects are not equal at actual['prices'][3]        # DataFrame, column "prices", row 3
values differ: 4.0 vs 5.0 (|diff| 1.0 > atol 1e-08 + rtol 1e-05 * 5.0)

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
each by its kind. It is written once.

```python
@dataclass(frozen=True, slots=True)
class Column:
    name: object
    kind: Literal["numeric", "duration", "exact", "nested"]
    values: object  # NumPy array (numeric, duration) or the library's own column (exact, nested)
    nulls: np.ndarray | None  # boolean mask


class Table(Protocol):  # what an adapter returns
    def structure(self) -> Hashable: ...  # equal for the same shape, names, dtypes, index
    def columns(self) -> Iterator[Column]: ...


class TableEqualityTester(BaseEqualityTester[object]):
    def objects_are_equal(self, actual, expected, ctx):
        a, e = to_table(actual), to_table(expected)  # the adapter of the type
        if a.structure() != e.structure():
            return ctx.fail(lambda: describe_structure_difference(a, e))
        return (
            (ca, ce, ca.name) for ca, ce in zip(a.columns(), e.columns())
        )  # Column has its own tester
```

A `Column` is a value with its own tester, compared by its kind:

| Kind       | Dtypes                                                                  | Comparison                                                                          |
| ---------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `numeric`  | integers, floats, nullable `Int64`/`Float64`, Arrow and polars numerics | null masks equal, then the kernel on the values (`0` under a null)                  |
| `duration` | `timedelta64`, polars `Duration`, Arrow `duration`                      | kernel on integer nanoseconds                                                       |
| `exact`    | booleans, strings, categories, instants, periods, intervals, decimals   | library `equals` (a null equals a null); never sees the tolerance                   |
| `nested`   | polars `List`/`Array`/`Struct`, Arrow nested, pandas `object`           | cells are children of the engine; the library's `equals` is the fast path if no NaN |

| Library | `structure()` is                                                               | Never used any more                                      |
| ------- | ------------------------------------------------------------------------------ | -------------------------------------------------------- |
| pandas  | type, shape, `index.equals` (values, dtype, names), `columns.equals`, `dtypes` | `pandas.testing`, copies of frames, `dt.total_seconds()` |
| polars  | `schema` (names, order, dtypes), height; a `LazyFrame` is collected            | `polars.testing`                                         |
| pyarrow | `type`, length; for a table `schema.equals(check_metadata=False)`, row count   | `equals` followed by a NumPy fallback                    |

A `Series` or an `Array` is a one-column table. pandas `attrs` and `flags` and Arrow schema metadata
are ignored by design. The dtype includes the time resolution (`timedelta64[s]` and `timedelta64[ns]`
differ, as in NumPy). Every comparison is a boolean array and `all()`, so a failing comparison costs
what a passing one does, and the first `False` gives the column and the row. Adding a library means
writing one adapter of about 40 lines.

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

| #   | Current behaviour on `main`                                                                                                                                                                                                                                                                                                                                   | Change                                                      |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| C1  | **[verified]** `ac(np.array([2**62]), np.array([2**62 + 1]), atol=0.5, rtol=0)` is `True` (torch, pandas, polars, pyarrow, `uint64` too); a list and a Python `int` give `False`                                                                                                                                                                              | integer difference in integer arithmetic (section 6)        |
| C2  | **[verified]** `ac(IntEnum.A, IntEnum.B, atol=5)` and `IntFlag` members are `True`; a plain `Enum` is `False`                                                                                                                                                                                                                                                 | `ScalarEqualityTester` is exact for `(bool, Enum)`          |
| C3  | **[verified]** pandas `Series`, `DataFrame`, `Index`, `"boolean"` arrays: `True` is close to `False` with `atol=1`                                                                                                                                                                                                                                            | the `exact` column kind                                     |
| C4  | **[verified]** nested cells: polars applies the tolerance inside a list, pyarrow does not, pandas uses its own rule and ignores `equal_nan` (`eq(pd.Series([[nan]]), same, equal_nan=True)` is `False`)                                                                                                                                                       | the `nested` kind: cells go through the engine              |
| C5  | **[verified]** a failing exact comparison of two pandas series of 2M floats takes 5 898 ms (equal copy: 2.9 ms; `pd.testing.assert_series_equal` alone: 5 804 ms); a 200k x 10 frame: 538 ms                                                                                                                                                                  | no `pandas.testing`; boolean arrays                         |
| C6  | **[verified]** pandas `object` column: two distinct `NaN` are equal without `equal_nan`; `None` equals `NaN` but `hash_object` differs                                                                                                                                                                                                                        | `nested` kind: strict type per cell                         |
| C7  | **[verified]** `MappingProxyType` and `array.array` fall to `==`: no tolerance, a `mappingproxy` holding arrays raises `TypeError`                                                                                                                                                                                                                            | register both (two lines)                                   |
| C8  | **[verified]** `float16`/`float32` arrays at extreme values: `RuntimeWarning` and a wrong `True` (`60000` against `-60000`, `rtol=1.5`); a Python float gives `False`                                                                                                                                                                                         | cast to `float64` in the kernel                             |
| C9  | **[verified]** masked arrays: the data under the mask is compared, so arrays that print identically differ; `hash_object` ignores it                                                                                                                                                                                                                          | compare mask and unmasked values; keep `fill_value`         |
| C10 | **[verified]** `compare` gives an empty path for an array, a frame and an Arrow table; a set element is shown as an index (`actual[2]`)                                                                                                                                                                                                                       | engine path, `ctx.fail(*path)`, `Element`                   |
| C11 | **[verified]** `show_difference=True` is silent by default, exists on two functions, and there are 50 occurrences and 39 `report_difference` calls behind `if config.show_difference`                                                                                                                                                                         | one first-failure record; remove the option and the logger  |
| C12 | **[verified]** `compare` on 1M floats differing at index 0 takes 231 ms (`objects_are_equal`: 10 ms): it builds the `repr` of the roots                                                                                                                                                                                                                       | bounded reasons; drop the root `repr`s                      |
| C13 | **[verified]** 6.4 frames per dict level; a 99-level dict fails with `RecursionError` from a 300-frame caller; 830k calls for 5 000 entries                                                                                                                                                                                                                   | the engine (section 4.2)                                    |
| C14 | **[reading]** 50 handler classes, a mutable linked list with 500 lines of machinery (`handler/base.py` 375, `mixin.py` 123), `SameObject`/`SameType` copied into 29 testers                                                                                                                                                                                   | tester method; the engine owns both checks                  |
| C15 | **[reading]** `EqualityConfig` holds settings and the state of one run, the registry is passed twice, `compare` takes a `Tolerance` and the others `rtol`/`atol`                                                                                                                                                                                              | frozen config; one `Engine` per call; `compare(rtol, atol)` |
| C16 | **[reading]** pandas 503, polars 368, pyarrow 172 lines: three implementations of one algorithm                                                                                                                                                                                                                                                               | `TableEqualityTester` and three adapters                    |
| C17 | **[verified]** docs: the pandas paragraph says a frame with a missing value is never equal to its copy (false); Design Principles lists `show_difference`; `extending.md:139` raises `ImportError` in `python -m doctest` (`get_default_registry` is in `coola.equality.tester`); the `objects_are_equal` docstring Note is the one of `objects_are_allclose` | fix the four statements                                     |
| C18 | **[reading]** `coola.equality.handler` exports 50 names, `tester` 36; `coola.io`, `coola.hashing.registry`, `coola.registry.base` import the private path `coola.equality.interface`                                                                                                                                                                          | about 25 public names; import from `coola.equality`         |
| C19 | **[verified]** no property compares the same random numbers in a list, an array, a tensor, a series and an Arrow array; library types appear in hashing and summary properties only                                                                                                                                                                           | cross-container property tests (step 1)                     |

Already true on `main` and kept: strict type, order rules, signed zero, time zones, the unregistered
fallback, co-inductive cycles, thread safety of `compare` (8 threads x 200 calls), reflexivity of
pandas frames with nulls, `compare` costing the same as `objects_are_equal` on equal data, the
elementwise symmetric kernel `handler/elementwise.py` (the base of `_numeric.py`), the registry and its
lazy loading.

### 10.2 Steps

One PR per row, as `CLAUDE.md` asks. "B" marks a breaking change. Steps 1 to 8 do not touch the handler
API and fix wrong answers; steps 9 to 14 move the architecture one family at a time, with `main` green
after each.

| Step | Content                                                                                                                                                                                                                                                                                                                                                                                                       | Closes             | B   | Depends on |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | --- | ---------- |
| 1    | **Cross-container property tests**: the same random `list[float]` and `list[int]` in a list, NumPy, torch (float64, int64), pandas, polars, pyarrow, xarray give the same answer for random `rtol`, `atol`, `equal_nan`; a copy is equal; the swap is symmetric; `xfail` for C1, C3, C4, C6 until fixed                                                                                                       | C19                |     |            |
| 2    | Enum members exact in the number tester; pandas booleans exact (`Series`, `DataFrame`, `Index`, extension arrays)                                                                                                                                                                                                                                                                                             | C2, C3             | B   | 1          |
| 3    | Kernel: exact integer difference, cast of small floats to `float64`, `over` silenced; applies to NumPy, torch, JAX and the numeric columns of frames                                                                                                                                                                                                                                                          | C1, C8             |     | 1          |
| 4    | Register `types.MappingProxyType` and `array.array`; regression test with arrays inside a `mappingproxy`                                                                                                                                                                                                                                                                                                      | C7                 |     |            |
| 5    | Fix the four documentation statements; find why the Markdown doctest job did not stop on `extending.md`                                                                                                                                                                                                                                                                                                       | C17                |     |            |
| 6    | Masked arrays: compare the mask and the unmasked values, keep `fill_value`                                                                                                                                                                                                                                                                                                                                    | C9                 | B   |            |
| 7    | pandas on column kinds inside the current handler (no `pandas.testing`; `object` and nested cells through the registry); failing-path benchmark within 2x of the passing path                                                                                                                                                                                                                                 | C4, C5, C6         | B   | 1, 3       |
| 8    | polars and pyarrow on column kinds: nested cells through the registry                                                                                                                                                                                                                                                                                                                                         | C4                 | B   | 1, 3       |
| 9    | **Engine**: add `Engine`, the lookup cache, the explicit stack, identity/type/depth/cycle ownership. Every existing chain tester becomes a leaf tester that the engine calls with the engine as `config` (the old attributes stay as properties), so nothing breaks yet                                                                                                                                       | C13, part C14, C15 |     | 1          |
| 10   | One reporting mechanism: `report_difference` forwards to `ctx.fail`; first-failure record; lazy bounded reasons; remove `show_difference`, the logger sink, `compare(tolerance=, max_repr_chars=)` and the root fields of the result                                                                                                                                                                          | C11, C12, C15      | B   | 9          |
| 11   | Port the Python testers (object, scalar, sequence, mapping, set, dataclass, attributes, pydantic) from chains to `objects_are_equal` returning a verdict or children; a port reports its location; delete the handlers nobody uses                                                                                                                                                                            | C14                | B   | 10         |
| 12   | Port the array testers (NumPy, masked, torch, JAX, xarray): index in the path, bounded value reasons                                                                                                                                                                                                                                                                                                          | C14, C10           | B   | 11         |
| 13   | `TableEqualityTester` and the three adapters replace the frame handlers; column and row in the path; `Element` for sets                                                                                                                                                                                                                                                                                       | C16, C10           | B   | 11, 7, 8   |
| 14   | Delete the chain machinery (`BaseEqualityHandler`, mixin, `create_chain`, `BaseChainEqualityTester`, `HandlerEqualityTester`, `check_recursion_depth`, `track_comparison`); freeze `EqualityConfig`; library code to private modules; trim `__all__`; `coola.io` and the others import from `coola.equality`; migration note in `docs/docs/migration`; rewrite `equality_handler.md` and `equality_tester.md` | C14, C15, C18      | B   | 12, 13     |

Steps 11 to 14 break every custom handler and tester (`handle(actual, expected, config)` and
`objects_are_equal(actual, expected, config)` become `objects_are_equal(actual, expected, ctx)` that
returns a verdict or the child pairs, with `config.registry.objects_are_equal(a, b, config)` replaced
by returning `(a, b, part)`). They share one release and one migration note showing that change.

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

The design choices that need a decision are listed, with options and a recommendation, in the last section
of this document ("Design choices to decide").

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

## Design choices to decide

Each choice has the options, what each one costs, and a recommendation. "Decision" is left empty for
the maintainers; the steps of section 10.2 that depend on a choice are named.

### Q1. Engine: iterative or recursive (steps 9, 11)

| Option       | For                                                                                          | Against                                                                                                                             |
| ------------ | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| A. Iterative | No `RecursionError`, explaining is free, container testers are pure functions of two objects | About 1.5x slower than B on the happy path (still 3 to 5x faster than `main`); a tester returns an iterator, which is less familiar |
| B. Recursive | Fastest; testers call `ctx.compare`, the familiar shape                                      | Interpreter stack limit and `max_depth` stay; depth, cycle and path need a `try/finally`                                            |

Recommendation: A. Decision: ____

### Q2. Tester return protocol (step 9)

| Option                                                                          | Note                                                                    |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| A. One method returns `bool` or an iterator of children                         | One name, no flag; the engine tests the type of the result              |
| B. Two methods, `equal` (leaf) and `expand` (container) plus a `container` flag | Explicit and type-checkable; two entry points to document and to forget |
| C. Always an iterator (a leaf yields nothing and fails through `ctx.fail`)      | One shape; leaves pay for a generator                                   |

Recommendation: A. Decision: ____

### Q3. Remove `show_difference` and the logger sink (step 10)

Options: A. remove (`compare` and `assert_*` explain); B. keep it on all five functions and route it
through the engine (an `INFO` log of `compare(...).format_message()`); C. keep as is.
A removes the option, 50 occurrences and the silent-by-default behaviour, and breaks users and the
guides' "Finding a Difference". B keeps the call sites working. Recommendation: A. Decision: ____

### Q4. Shape of `ComparisonResult` and of the entry points (step 10)

- Drop the root `repr`s (`actual`, `expected`) and `max_repr_chars`: the reason holds the two
  differing values (recommended), or keep the roots truncated.
- `compare(rtol, atol)` like the other four functions (recommended), or keep `Tolerance` as the
  parameter of `compare` only.
- Keep `Tolerance` public as the config value (recommended) or make it private.

Decision: ____

### Q5. Masked arrays (step 6)

A. Compare the mask and the unmasked values, ignore the data under the mask (the hasher already does;
two arrays that print the same are equal). B. Keep comparing the data under the mask. Also decide
whether `fill_value` stays compared (recommended: yes, it is visible through `filled()`).
Recommendation: A. Decision: ____

### Q6. Nested and `object` cells of frames (steps 7, 8, 13)

A. Compare each cell through the engine: one rule per value (`None` is not `NaN`, a nested float gets
the tolerance), slower; keep the library's `equals` as the fast path when there is no NaN and no
tolerance. B. Leave nested cells to the library (fast, three rules, wrong `equal_nan`). C. Forbid
nested cells (raise). Recommendation: A. Decision: ____

### Q7. Time resolution is part of the dtype (step 13)

`timedelta64[s]` and `timedelta64[ns]` differ, as in NumPy. pandas 3 infers different resolutions
for values built in different ways, so two frames of the same instants can differ. A. keep strict
(document the `astype` fix); B. cast both to the finer resolution before comparing. B is friendlier
and breaks the "same dtype" rule of the guide. Recommendation: A. Decision: ____

### Q8. Precision policy of the kernel (step 3)

A. Cast `float16`, `bfloat16`, `float32` to `float64` (correct at the edge of the range, memory x2 for
the temporaries); B. keep the dtype (fast, wrong at extremes, warns). Integers: C. exact difference,
tolerance in `float64` (recommended), or D. exact `Fraction` arithmetic for every integer array (exact
everywhere, much slower). Recommendation: A and C. Decision: ____

### Q9. What stays ignored (no step)

`requires_grad`, pandas `attrs` and `flags`, Arrow schema metadata, `pydantic` private attributes,
LazyFrame query plan (collected). Options: keep ignored and document each (recommended), or add
opt-in flags. Decision: ____

### Q10. Strictness of enums and labels (step 2)

A. `bool` and every `enum.Enum` member are compared exactly, tolerance never applies (recommended).
B. Also `str`, `bytes` and every non-numeric type through an explicit "label" list. Decision: ____

### Q11. Compatibility and release strategy (steps 9 to 14)

A. One breaking release for steps 11 to 14, with a migration note and the engine already present
(recommended). B. A compatibility adapter that keeps `BaseEqualityHandler`/`create_chain` working for
one more release, deprecated, at the cost of the 500 lines of chain machinery staying. C. Do not
break: stop after step 8 and keep the chain. Decision: ____

### Q12. Public surface (step 14)

About 25 names in `coola.equality`; the library testers are private. Options: A. private library
modules reached through the registry (recommended); B. keep the named library testers public
(`NumpyArrayEqualityTester`, ...) so that users can compose or subclass them. Also decide the
public name of the engine type (`Engine`, `ComparisonContext`) and whether `Element` and
`AttributeName` are exported. Decision: ____

### Q13. Registry lookup cache (step 9)

A. Per-call `dict` (recommended: no lock in the hot path; a registration during a call is seen by
the next call). B. A lock-free immutable snapshot inside the registry (benefits `hashing` and
`summary`, which share the registry class, but changes a shared component). Decision: ____

### Q14. Size of the first engine PR (step 9)

A. Engine plus an adapter that runs the current chain testers as leaves (recommended: nothing
breaks, benchmarks can compare). B. Engine and the port of the Python testers in one PR (fewer
intermediate states, a large diff). Decision: ____
