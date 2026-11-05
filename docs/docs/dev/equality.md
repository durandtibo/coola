# Architecture of `coola.equality`

:book: This page describes the design of the `coola.equality` package for contributors: the
invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules
that users rely on are in the [user guide](../uguide/equality.md#guarantees); the API is in the
[reference](../refs/equality.md).

## Overview

The package is small on purpose: one engine, one extension point, one numeric rule and one table
algorithm.

### Constraints

1. **Performance is a requirement.** Both the passing and the failing path are measured by the
   benchmarks; a simpler design that is slower is rejected.
2. **Early exit.** Two objects are different as soon as one difference is found: nothing after it is
   visited, converted or formatted.
3. **First difference only, on demand.** `objects_are_equal` and `objects_are_allclose` build no
   message and no path; only `compare` and `assert_*` explain, and only the first difference.
4. **Equality drives hashing, not the reverse.** `hash_object` is aligned with equality, never the
   other way round.

### Architecture

```
objects_are_equal / objects_are_allclose / compare / assert_objects_equal / assert_objects_allclose
        |   one EqualityConfig (frozen) and one Engine per call
        v
Engine.run(actual, expected)    explicit stack of child iterators; one loop
        |   identity -> type -> tester lookup (cached per call) -> verdict or children
        v
Tester.objects_are_equal(actual, expected, ctx) -> bool | Iterator[(actual, expected, path part)]
        |   leaves: numbers, arrays, scalars       containers: sequence, mapping, set, object, table
        v
numeric kernel (NumPy, torch, JAX)      adapters: pandas, polars, pyarrow -> table
```

Four rules follow from it:

1. **One extension point: the tester.** A tester is a class with one method.
2. **The engine owns the invariants**: identity, strict type, depth, cycles, path and first failure.
   A tester cannot forget them because it never recurses.
3. **A container decomposes, it does not recurse**: it returns the child pairs that must all be equal,
   lazily and in the order the first difference is wanted.
4. **coola owns the semantics, the libraries own the structure**: every number goes through the
   kernel; a library says whether two frames have the same columns, never whether `1` is close to `2`.

### The engine

The engine is one loop over a stack of child iterators (depth first, so the first difference in
iteration order is the one reported). For each pair:

1. `actual is expected`: equal.
2. `type(actual) is not type(expected)`: different.
3. The tester is looked up in a per-call `dict[type, tester]` filled from the registry the first time
   a type is met, so the registry lookup is paid once per type. A registration made during a
   comparison is seen by the next call.
4. The tester is called. A `bool` is the verdict. An iterator is a container: if its pair is already
   on the stack the pair is equal (cycle); if `max_depth` is set and the stack is that deep, `MaxDepthExceededError`
   is raised; otherwise it is pushed and the loop continues with its first child.
5. On `False`, the path is built once from the path parts of the stack, the part of the failing pair
   and the parts given to `ctx.fail`.

Consequences: the stack use is constant, `RecursionError` cannot happen, and explaining costs nothing
in the loop. `ctx.fail(reason, *path)` returns `False`; `reason` can be a callable that is called only
when the difference is explained. The full loop is in the [appendix](#appendix-the-engine-loop).

### Reporting

There is one record, the first failure `(path, reason)`: no list of reports and no message per parent
level. Values in a reason go through one bounded formatter, so a failure in a million-element array
formats a few values, not the array. A path part is a key or a position (`['a']`, `[3]`),
`AttributeName("x")` (`.x`) or `Element(2)` (`{2}`). An array reports its index. A table is a leaf
(the path stops at the table); its reason names the part of the structure that differs or the first
differing column, row and values.

### The numeric kernel

NumPy, torch and JAX share one formula:

```
close = (a == b) | (finite & (|a - b| <= atol + rtol * max(|a|, |b|))) | (isnan(a) & isnan(b) if equal_nan)
```

| Dtype                                | Preparation                           | Rule                             |
| ------------------------------------ | ------------------------------------- | -------------------------------- |
| `bool`                               | none                                  | exact                            |
| signed and unsigned integers         | exact difference in integers          | kernel, tolerance in `float64`   |
| `float16`, `bfloat16`, `float32`     | cast to `float64`                     | kernel                           |
| `float64`, `complex128`              | none (`complex64` is cast up)         | kernel                           |
| `timedelta64`                        | integer nanoseconds; `NaT` mask apart | kernel; `NaT` by `equal_nan`     |
| `datetime64`, `str`, `bytes`, `void` | none                                  | exact; `NaT` by `equal_nan`      |
| `object`                             | none                                  | cells are children of the engine |

Python scalars use the same formula; `Decimal`, `Fraction` and `timedelta` keep a difference-based
rule.

### Tables

pandas, polars and PyArrow share one algorithm in `TableEqualityTester`, which takes an adapter per
library: check the structure (shape, names, dtypes, index), then compare the columns lazily, one by
one, each by its kind, and stop at the first column that differs. A `Series` or an `Array` is a
one-column table (its failure says "series" or "arrays" through the `kind_name` of the adapter).

| Kind       | Dtypes                                                       | Comparison                                                      |
| ---------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| `numeric`  | integers, floats, nullable and Arrow numerics                | null masks equal, then the kernel                               |
| `duration` | `timedelta64`, polars `Duration`, Arrow `duration`           | kernel on integer nanoseconds                                   |
| `exact`    | booleans, strings, categories, instants, nested and `object` | the library's `equals`; never sees the tolerance or `equal_nan` |

Nested and `object` columns are the one place where the library decides: `None` against `NaN` in an
`object` column is equal, and a `NaN` inside a nested cell equals a `NaN` without `equal_nan`. `attrs`,
`flags` and Arrow schema metadata are ignored. Adding a library means writing one adapter of about 40
lines.

### Module layout

```
coola/equality/
  interface.py   objects_are_equal, objects_are_allclose
  result.py      ComparisonResult, compare, assert_objects_equal, assert_objects_allclose
  config.py      EqualityConfig
  tolerance.py   Tolerance
  engine.py      Engine, Element
  tester/        base, registry, one module per family (default, scalar, collection, difference,
                 dataclass, object, equal, tolerant), per library (numpy, torch, jax, xarray,
                 pydantic, scipy), and table.py with the adapters pandas.py, polars.py, pyarrow.py
  _helpers/      private helpers (formatting, numeric kernel, comparison keys)
```

## Design choices and trade-offs

Each row is a choice the current design makes, with the alternative it was preferred to. The sections
after the table give the longer arguments.

| Choice                                                                 | Reason                                                                      | Alternative and its cost                                                     |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Iterative engine                                                       | constant stack, one place for depth, cycles and path                        | a recursive engine is faster, but needs a `try/finally` per level            |
| One tester method returns a verdict or an iterator                     | one extension point                                                         | two methods (`equal`, `expand`) are explicit, but one can be forgotten       |
| No `show_difference` and no logger; `compare` and `assert_*` explain   | no cost on the fast path, thread safe                                       | a flag on the fast function makes it format messages                         |
| `bool` and every `Enum` are exact                                      | they are labels, not quantities                                             | a "label" list for `str` and `bytes` is more to maintain                     |
| Masked arrays compare the data under the mask, the mask and fill value | no hidden difference                                                        | ignoring the masked data matches how the array prints                        |
| Small floats are cast to `float64`                                     | no overflow of the tolerance at the edge of the range                       | keeping the dtype is faster for large `float32` arrays but wrong at edges    |
| Integer difference is exact                                            | no rounding of large integers                                               | exact `Fraction` arithmetic is exact everywhere but much slower              |
| Tables are one leaf, reported by column and row                        | simple and fast to compare; the row is found only when explaining           | a second pass per library, and no row for a nested column                    |
| Time resolution is part of the dtype                                   | as in NumPy                                                                 | casting to the finer resolution is friendlier but hides a dtype difference   |
| Library testers are public                                             | users can compose, subclass or register them                                | private testers shrink the public surface                                    |
| `max_depth` (default `None`) and `MaxDepthExceededError`               | opt-in guard on pathological structures; a deep valid structure is compared | by default, no limit runs until memory is exhausted on an infinite structure |
| `compare(rtol, atol)` instead of a `Tolerance` argument                | the five functions share the same keywords                                  | a `Tolerance` argument is reusable, but unlike the array libraries           |
| No root `repr` in `ComparisonResult`                                   | the failure path does not depend on the size of roots                       | `repr` of the roots costs 20x on a failing million-element comparison        |

### Iterative or recursive engine

The iterative engine is about 1.5x slower than a recursive one on the happy path, and a tester
returns an iterator, which is less familiar than calling `ctx.compare`. In exchange there is no
`RecursionError`, explaining is free, and container testers are pure functions of two objects. A
recursive engine keeps the interpreter limit and `max_depth`, and needs a `try/finally` to keep the
depth, the cycle set and the path right.

### Tester return protocol

One method that returns a `bool` or an iterator of children has one name and no flag; the engine
tests the type of the result. Two methods (`equal` for a leaf and `expand` for a container, plus a
`container` flag) are explicit and type-checkable, but are two entry points to document and to forget.
An always-iterator protocol has one shape, but every leaf pays for a generator.

### Explanation lives in `compare` and `assert_*`

The fast functions have no `show_difference` flag and no logger. A flag would force every tester to
guard its formatting behind `if config.show_difference:` (a forgotten guard formats on every call),
require a `reporter` callback, and format the message before a logger can drop it. With the
explanation in `compare` and `assert_*`:

- `objects_are_equal` is guaranteed to format nothing, and there is no global state, so it is thread
  safe;
- the engine has one failure record, and the five functions have the same signature;
- the explanation is a structured `ComparisonResult` (`path`, `reason`), not log text.

The cost is that a caller who wants "the answer and a log line in one call" writes two lines
(`log.info(compare(a, b).format_message())`). Passing a callable (`explain=print`) would route the
explanation, but it adds a parameter and a callback invoked inside a comparison (re-entrancy,
exceptions) for a use that `compare` already covers.

### Result and tolerance shape

- **`ComparisonResult` has no root `repr`.** The `repr` of the whole roots does not say where the
  difference is (the path and the two differing values do), and building it on every failure made the
  failing path depend on the size of the roots (231 ms for `compare` of 1M floats against 10 ms
  without). The caller already holds the roots. A lazy property would keep references to the roots and
  give a different `repr` if they are mutated.
- **`compare` takes `rtol` and `atol`**, like `objects_are_allclose` and the array libraries, so a
  call moves from one function to the other by renaming. The defaults differ on purpose: `compare` is
  exact unless a tolerance is given, `objects_are_allclose` uses `1e-5` and `1e-8`.
- **`Tolerance` is public.** `EqualityConfig` holds one validated, hashable value instead of two loose
  floats, and a custom tester or a user-built config can read `ctx.config.tolerance`. A caller of the
  five functions never needs it.

### Data under a masked array

A masked array compares the data under the mask, the mask and the `fill_value`, so no difference is
hidden. Ignoring the data under the mask matches how two arrays print, and is what the hasher does;
the price of the current rule is that two arrays that print the same can be different. `fill_value` is
compared because it is visible through `filled()`. The hasher ignores the masked data, which keeps
"equal implies same hash" true; including it would make the hash as discriminating as equality at the
cost of changing every hash of a masked array.

### Nested and `object` cells of frames

The library's `equals` decides, because it is fast and a cell-by-cell comparison through the engine
would be slow. The price is three rules (the library's, the engine's and the hasher's): the tolerance
and `equal_nan` do not reach nested cells, `None` equals `NaN` in an `object` column, and the hasher
has to treat them the same way. This exception is part of the Guarantees table of the user guide.
Comparing each cell through the engine gives one rule per value (a nested float gets the tolerance,
`None` is not `NaN`) at the cost of speed.

### Time resolution

`timedelta64[s]` and `timedelta64[ns]` are different dtypes, as in NumPy. pandas 3 infers different
resolutions for values built in different ways, so two frames of the same instants can differ; the fix
is an `astype`, which the guide documents. Casting both sides to the finer resolution would be
friendlier, but it breaks the "same dtype" rule.

### Precision

Small floats (`float16`, `bfloat16`, `float32`) are cast to `float64` before the kernel. Keeping the
dtype is faster for large `float32` arrays, but overflows the tolerance at the edge of the range (a
`RuntimeWarning` and a wrong `True` for `60000` against `-60000` with `rtol=1.5`); the cast doubles
the temporaries. Integers use an exact integer difference and the tolerance in `float64`; exact
`Fraction` arithmetic for every integer array would be exact everywhere and much slower.

### What is ignored

`requires_grad`, pandas `attrs` and `flags`, Arrow schema metadata and pydantic private attributes do
not take part in a comparison. They are documented in the user guide instead of being opt-in flags.
`bool` and every `enum.Enum` member are labels: the tolerance never applies to them.

### Public surface

The library testers (`NumpyArrayEqualityTester`, `TorchTensorEqualityTester`, ...) are public so that
a user can compose, subclass or register them. Pandas, polars and PyArrow share the public
`TableEqualityTester(adapter)` rather than one subclass per library, which keeps the number of names
small. `Engine` is the public type of `ctx`, and `Element` and `AttributeName` are exported because
they appear in `ComparisonResult.path`. The helpers are private, except the kernel `close`.

### Tables are reported by column and row

A table is a leaf, so the path stops at the table. The loop over the columns knows the name of the
first differing column at no cost, so the reason always names it. The row and the two values are found
by a second pass, only when the comparison explains a difference (`ctx.explaining`): the numeric and
duration columns use the NumPy kernel, the other columns use a function of the adapter
(`Column.difference`). A nested column has none, so its reason has no row. An adapter can say what
differs in the structure with `structure_difference`, which is called only to explain.

### Lookup cache

The tester cache is a per-call `dict`: there is no lock on the hot path, and a registration during a
call is seen by the next call. A lock-free immutable snapshot inside the registry would also benefit
`hashing` and `summary`, but changes a component that every package shares.

## Public API

```python
objects_are_equal(actual, expected, *, equal_nan=False, max_depth=None, registry=None) -> bool
objects_are_allclose(actual, expected, *, rtol=1e-5, atol=1e-8, equal_nan=False, max_depth=None, registry=None) -> bool
compare(actual, expected, *, rtol=0.0, atol=0.0, equal_nan=False, max_depth=None, registry=None) -> ComparisonResult
assert_objects_equal(actual, expected, *, equal_nan=False, max_depth=None, registry=None, root="actual") -> None
assert_objects_allclose(actual, expected, *, rtol=1e-5, atol=1e-8, equal_nan=False, max_depth=None, registry=None, root="actual") -> None
```

The five functions have the same keywords with the same meaning. `compare` is exact unless `rtol` or
`atol` is given.

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
    max_depth: int | None = None  # validated: None (no limit) or a positive int, not a bool


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
`ctx`), `EqualityTesterRegistry`, `get_default_registry`, `register_equality_testers`, and the named
testers (`ObjectAttributesEqualityTester`, `EqualEqualityTester`, `OrderedMappingEqualityTester`,
`SequenceEqualityTester`, `MappingEqualityTester`, ... and the library testers
`NumpyArrayEqualityTester`, `TorchTensorEqualityTester`, `TableEqualityTester`, ...).

## The engine and the testers

### The tester contract

```python
class BaseEqualityTester(ABC, Generic[T]):
    @abstractmethod
    def objects_are_equal(
        self, actual: T, expected: T, ctx: Engine
    ) -> bool | Iterator[tuple[object, object, object]]:
        """A verdict (a leaf), or the child pairs (actual, expected, path part) that must all be
        equal (a container), produced lazily and in the order the first difference is wanted."""

    # default __eq__: same type and same vars
```

`ctx` gives `ctx.atol`, `ctx.rtol`, `ctx.equal_nan` (read once from the config), `ctx.fail(reason,
*path)`, `ctx.explaining` and `ctx.warn_once(key, message)`. `ctx.fail` returns `False`, so a tester
writes `return ctx.fail(...)`. `reason` may be a callable; it is called only when `ctx.explaining`
(`compare`, `assert_*`), so `objects_are_equal` formats nothing. `path` adds parts after the ones the
engine knows (the index in an array).

A tester has no state but its configuration; `__eq__` comes from the base class. The engine runs the
loop described in [The engine](#the-engine): the registry lookup costs about 300 ns, so it is paid
once per type and call. Stack use is constant (100 000 nested dicts compare under the default
interpreter limit of 1 000), so `max_depth` is a policy, not a safety net. `MaxDepthExceededError` is a
plain `Exception`, not a `RecursionError`.

### Testers

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
        close = close_numpy(actual, expected, ctx)  # boolean array, the kernel of
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

### Type rules

| Type                                                                        | Tester                                                   | Rule                                                                                                                        |
| --------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `object` (unregistered)                                                     | `DefaultEqualityTester`                                  | `==`; a raise or a non-bool result is a `TypeError` naming the type                                                         |
| `int`, `float`, `complex`, NumPy scalars                                    | `ScalarEqualityTester`                                   | kernel formula; `bool` and enums exact                                                                                      |
| `Decimal`, `Fraction`, `timedelta`                                          | `DifferenceCloseEqualityTester`                          | difference-based tolerance; `Decimal` NaN by `equal_nan`                                                                    |
| `list`, `tuple`, `deque`, `Sequence`, `array.array`, dict values            | `SequenceEqualityTester`                                 | length, then items in order                                                                                                 |
| `dict`, `Mapping`, `MappingProxyType`                                       | `MappingEqualityTester`                                  | same keys (by type, NaN by `equal_nan`), then values                                                                        |
| `OrderedDict`                                                               | `OrderedMappingEqualityTester`                           | and the same key order                                                                                                      |
| `set`, `frozenset`                                                          | `SetEqualityTester`                                      | elements matched by hash and type; reports `Element`                                                                        |
| dataclasses, pydantic models                                                | `DataclassEqualityTester`, `PydanticModelEqualityTester` | fields as `AttributeName` children                                                                                          |
| `np.ndarray`, `np.ma.MaskedArray`, NumPy `datetime64`/`timedelta64` scalars | `Numpy*EqualityTester`                                   | dtype, shape, kernel; for a masked array the data (masked values included), the mask and `fill_value`; `NaT` by `equal_nan` |
| `torch.Tensor`, `PackedSequence`, `jax.Array`, xarray                       | `Torch*`, `Jax*`, `Xarray*EqualityTester`                | device, dtype, shape, layout, kernel; xarray parts as children                                                              |
| `scipy.sparse` matrices and arrays                                          | `ScipySparseEqualityTester`                              | class, shape, dtype, non-zero positions, then the stored values with the kernel; duplicates summed, stored zeros ignored    |
| pandas, polars, pyarrow frames, series, arrays                              | `TableEqualityTester`                                    | structure, then columns by kind                                                                                             |

### The registry

`EqualityTesterRegistry` is a `TypeRegistry` (`coola.registry`):

- `find_equality_tester(data_type)` returns the tester for the most specific registered type, walking
  the Method Resolution Order (MRO); `object` is registered with `DefaultEqualityTester`, so a tester
  is always found;
- more specific testers override general ones, and a subclass inherits the tester of its parent (a
  tester for `Sequence` applies to `list` and `tuple`);
- testers for optional libraries are registered as lazy groups, and loaded when a type of the library
  is looked up.

`get_default_registry()` returns the global registry and `register_equality_testers(mapping, *,
exist_ok=False)` adds testers to it. A test that registers a tester should not leak it: use
`coola.registry.default.temporary_default_registries()` (see the testing guide), or work on
`get_default_registry().copy()`.

### Error handling

When a specific tester is not available, the lookup falls back to a more general tester (MRO), then to
`DefaultEqualityTester` (registered for `object`), which uses `==`. When the comparison explains the
difference (`compare`, `assert_*`), the reason says what differs, where in the structure, and the
values that differ.

## Writing a tester

To add support for a custom type:

1. **Implement a tester:**

   ```python
   from coola.equality import BaseEqualityTester


   class MyTypeEqualityTester(BaseEqualityTester[MyType]):
       def objects_are_equal(self, actual, expected, ctx):
           # the engine checked the identity and the type: both objects are a MyType
           return actual.compare_to(expected) or ctx.fail("my objects differ")
   ```

2. **Register it:**

   ```python
   from coola.equality import register_equality_testers

   register_equality_testers({MyType: MyTypeEqualityTester()})
   ```

   To avoid modifying the global registry, create an `EqualityTesterRegistry`, register the tester on
   it and pass it with `registry=`.

3. **Use it:** `objects_are_equal(obj1, obj2)`.

See the [extending guide](../uguide/extending.md) for more details.

Checklist:

- Return a verdict (`ctx.fail(reason)` for a difference) or the child pairs `(actual, expected,
path part)`; never recurse: the engine checks the identity, the type, the depth and the cycles.
- Use `AttributeName` for an attribute and `ctx.fail(reason, *path)` for the location inside a leaf
  (the index in an array).
- Build the reason lazily (a callable), so `objects_are_equal` formats nothing.
- Honour `ctx.equal_nan` and the tolerance (`ctx.rtol`, `ctx.atol`: `|a - e| <= atol + rtol *
max(|a|, |e|)`), or document that your tester ignores them.
- Check the cheap properties first (metadata such as shape, dtype, device), then the values.

## Strict types

`coola` enforces strict type checking:

- `1` (int) ≠ `1.0` (float) ≠ `True` (bool)
- `list` ≠ `tuple`
- `dict` ≠ `OrderedDict`
- two `OrderedDict`s with the same items in a different order are different (the order of the keys of
  a `dict` is ignored)

This prevents subtle bugs from type coercion.

## Not covered

- Per-library corner cases beyond the tested ones: sparse and quantized torch tensors, nested tensors,
  dask-backed xarray, pandas `Interval`, `Period` and `Sparse` columns on the `exact` path.
- The GPU path, Windows, fork behaviour.
- A registration made while a comparison runs is seen by the next call only (the lookup cache is per
  call).
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
                    if self.max_depth is not None and len(stack) >= self.max_depth:
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

## Relation to hashing

The property "equal objects have the same `hash_object`" holds with this equality. The hashers are
allowed to be coarser than equality (different values may share a hash), never finer. The cases that
need care:

| Case                                                  | Equality                                | Hasher                                                               |
| ----------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------- |
| pandas `object` column, `None` against `NaN`          | equal (the library's `equals`)          | must hash a `None` and a `NaN` of an `object` column the same        |
| `NaN` inside nested cells (lists, structs) of a frame | equal without `equal_nan` (the library) | the hash of nested `NaN` must not depend on the NaN bit pattern      |
| Nested floats of a frame, tolerance                   | exact (no tolerance)                    | nothing to do: equal values have equal hashes                        |
| Masked array, data under the mask                     | compared                                | ignored by the hasher; allowed, because different values may collide |
| Integers above 2^53 in arrays, `bool` and enums       | exact                                   | the hash is exact (no `float64` conversion)                          |
| `IntEnum` member against `int`                        | different types                         | the hashes may be equal: allowed                                     |
| `MappingProxyType`, `array.array`                     | compared as a mapping and a sequence    | hashed consistently with a `dict` and a `list` of the same content   |
| Time resolution (`timedelta64[s]` against `[ns]`)     | different dtypes                        | allowed to share a hash                                              |

The hashing property tests (`tests/properties/test_hashing*.py`) take the equal pairs that the
equality tests generate and check that they hash equally. Where the hasher needs a rule that is not
obvious, the rule is taken from the equality guide, not the other way round. See
[the hashing page](hashing.md).

## Packages that depend on the equality API

| Package                                       | Use                                                                            |
| --------------------------------------------- | ------------------------------------------------------------------------------ |
| `coola.io`, `coola.nested`                    | import `objects_are_equal` from `coola.equality`, not from its private modules |
| `coola.registry` (equality of two registries) | testers are compared with their `__eq__`                                       |
| `coola.testing`                               | fixtures and helpers compare with `compare` and `assert_*`                     |
