# Architecture of `coola.equality`

:book: This page describes the design of the `coola.equality` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/equality.md#guarantees); the API is in the [reference](../refs/equality.md).

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
   on the stack the pair is equal (cycle); if the stack is `max_depth` deep, `MaxDepthExceededError`
   is raised; otherwise it is pushed and the loop continues with its first child.
5. On `False`, the path is built once from the path parts of the stack, the part of the failing pair
   and the parts given to `ctx.fail`.

Consequences: the stack use is constant, `RecursionError` cannot happen, and explaining costs nothing
in the loop. `ctx.fail(reason, *path)` returns `False`; `reason` can be a callable that is called only
when the difference is explained.

### Reporting

There is one record, the first failure `(path, reason)`: no list of reports and no message per parent
level. Values in a reason go through one bounded formatter, so a failure in a million-element array
formats a few values, not the array. A path part is a key or a position (`['a']`, `[3]`),
`AttributeName("x")` (`.x`) or `Element(2)` (`{2}`). An array reports its index. A table is a leaf
and names only the first differing column; the user finds the row with the library (for example
`DataFrame.compare`).

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
one-column table.

| Kind       | Dtypes                                                       | Comparison                                                      |
| ---------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| `numeric`  | integers, floats, nullable and Arrow numerics                | null masks equal, then the kernel                               |
| `duration` | `timedelta64`, polars `Duration`, Arrow `duration`           | kernel on integer nanoseconds                                   |
| `exact`    | booleans, strings, categories, instants, nested and `object` | the library's `equals`; never sees the tolerance or `equal_nan` |

Nested and `object` columns are the one place where the library decides. `attrs`, `flags` and Arrow
schema metadata are ignored. Adding a library means writing one adapter of about 40 lines.

### Module layout

```
coola/equality/
  interface.py   the five functions
  config.py      Tolerance, EqualityConfig
  engine.py      Engine, Element
  result.py      ComparisonResult
  tester/        base, registry, one module per family (default, scalar, collection, difference,
                 dataclass, object, equal, tolerant), per library (numpy, torch, jax, xarray,
                 pydantic), and table.py with the adapters pandas.py, polars.py, pyarrow.py
  _helpers/      private helpers (formatting, numeric kernel, comparison keys)
```

### Decisions

| Decision                                                               | Reason                                                |
| ---------------------------------------------------------------------- | ----------------------------------------------------- |
| Iterative engine, not recursion                                        | constant stack, one place for depth, cycles and path  |
| One tester method returns a verdict or an iterator                     | one extension point                                   |
| No `show_difference` and no logger; `compare` and `assert_*` explain   | no cost on the fast path, thread safe                 |
| `bool` and every `Enum` are exact                                      | they are labels, not quantities                       |
| Masked arrays compare the data under the mask, the mask and fill value | no hidden difference                                  |
| Small floats are cast to `float64`                                     | no overflow of the tolerance at the edge of the range |
| Integer difference is exact                                            | no rounding of large integers                         |
| Tables are one leaf, reported by column                                | simple and fast; the library finds the row            |
| Time resolution is part of the dtype                                   | as in NumPy                                           |
| Library testers are public                                             | users can compose, subclass or register them          |
| `max_depth=100` and `MaxDepthExceededError` are kept                   | bound the work on pathological structures             |

## Public API

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

    # default __eq__ / __repr__: same type and same vars
```

`ctx` gives `ctx.atol`, `ctx.rtol`, `ctx.equal_nan` (read once from the config), `ctx.fail(reason,
*path)`, `ctx.explaining` and `ctx.warn_once(key, message)`. `ctx.fail` returns `False`, so a tester
writes `return ctx.fail(...)`. `reason` may be a callable; it is called only when `ctx.explaining`
(`compare`, `assert_*`), so `objects_are_equal` formats nothing. `path` adds parts after the ones the
engine knows (the index in an array).

### The engine

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

A tester has no state but its configuration; `__eq__` and `__repr__` come from the base class.

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
| pandas, polars, pyarrow frames, series, arrays                              | `TableEqualityTester`                                    | structure, then columns by kind                                                                                             |

## Decision record

Each choice has the options, what each one costs, and a recommendation. "Decision" is left empty for
the maintainers; the steps of that depend on a choice are named.

### Q1. Engine: iterative or recursive

| Option       | For                                                                                          | Against                                                                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| A. Iterative | No `RecursionError`, explaining is free, container testers are pure functions of two objects | About 1.5x slower than B on the happy path (still 3 to 5x faster than the previous version); a tester returns an iterator, which is less familiar |
| B. Recursive | Fastest; testers call `ctx.compare`, the familiar shape                                      | Interpreter stack limit and `max_depth` stay; depth, cycle and path need a `try/finally`                                                          |

Recommendation: A. Decision: A

### Q2. Tester return protocol

| Option                                                                          | Note                                                                    |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| A. One method returns `bool` or an iterator of children                         | One name, no flag; the engine tests the type of the result              |
| B. Two methods, `equal` (leaf) and `expand` (container) plus a `container` flag | Explicit and type-checkable; two entry points to document and to forget |
| C. Always an iterator (a leaf yields nothing and fails through `ctx.fail`)      | One shape; leaves pay for a generator                                   |

Recommendation: A. Decision: A

### Q3. `show_difference` and the logger sink

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

### Q4. Shape of `ComparisonResult` and of the entry points

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

### Q5. Masked arrays

A. Compare the mask and the unmasked values, ignore the data under the mask (the hasher already does;
two arrays that print the same are equal). B. Keep comparing the data under the mask. Also decide
whether `fill_value` stays compared (recommended: yes, it is visible through `filled()`).
Recommendation: A. Decision: B

### Q6. Nested and `object` cells of frames

A. Compare each cell through the engine: one rule per value (`None` is not `NaN`, a nested float gets
the tolerance), slower; keep the library's `equals` as the fast path when there is no NaN and no
tolerance. B. Leave nested cells to the library (fast, three rules, wrong `equal_nan`). C. Forbid
nested cells (raise). Recommendation: A. Decision: B

### Q7. Time resolution is part of the dtype

`timedelta64[s]` and `timedelta64[ns]` differ, as in NumPy. pandas 3 infers different resolutions
for values built in different ways, so two frames of the same instants can differ. A. keep strict
(document the `astype` fix); B. cast both to the finer resolution before comparing. B is friendlier
and breaks the "same dtype" rule of the guide. Recommendation: A. Decision: A

### Q8. Precision policy of the kernel

A. Cast `float16`, `bfloat16`, `float32` to `float64` (correct at the edge of the range, memory x2 for
the temporaries); B. keep the dtype (fast, wrong at extremes, warns). Integers: C. exact difference,
tolerance in `float64` (recommended), or D. exact `Fraction` arithmetic for every integer array (exact
everywhere, much slower). Recommendation: A and C. Decision: C

### Q9. What stays ignored (no step)

`requires_grad`, pandas `attrs` and `flags`, Arrow schema metadata, `pydantic` private attributes,
LazyFrame query plan (collected). Options: keep ignored and document each (recommended), or add
opt-in flags. Decision: keep ignored and document each

### Q10. Strictness of enums and labels

A. `bool` and every `enum.Enum` member are compared exactly, tolerance never applies (recommended).
B. Also `str`, `bytes` and every non-numeric type through an explicit "label" list. Decision: A

### Q11. Removal of compatibility code

Decided by the constraints: major release, no adapter, no alias, no deprecation period, deprecated and
compatibility code removed. Remaining question: should the release notes carry a before/after
example of a custom tester? Recommendation: yes, five lines. Decision: yes, create a migration guide

### Q12. Public surface

About 25 names in `coola.equality`; the library testers are private. Options: A. private library
modules reached through the registry (recommended); B. keep the named library testers public
(`NumpyArrayEqualityTester`, ...) so that users can compose or subclass them. Also decide the
public name of the engine type (`Engine`, `ComparisonContext`) and whether `Element` and
`AttributeName` are exported. Decision: B

### Q13. Registry lookup cache

A. Per-call `dict` (recommended: no lock in the hot path; a registration during a call is seen by
the next call). B. A lock-free immutable snapshot inside the registry (benefits `hashing` and
`summary`, which share the registry class, but changes a shared component). Decision: A

### Q14. Size of the first engine PR

A. Engine plus the current chain testers called as leaves during the port (recommended: small PRs,
benchmarks can compare; removed in). B. Engine and the port of the Python testers in one PR (fewer
intermediate states, a large diff). Decision: A

### Decisions taken and their effect on the design

| Q   | Decision                                                                                              | Effect, already applied in the sections above                                                                            |
| --- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Q1  | Iterative engine                                                                                      | , 4                                                                                                                      |
| Q2  | One method returns a verdict or an iterator                                                           |                                                                                                                          |
| Q3  | Remove `show_difference` and the logger                                                               |                                                                                                                          |
| Q4  | Drop root `repr`s; `compare(rtol, atol)`; `Tolerance` public                                          |                                                                                                                          |
| Q5  | Masked arrays: keep comparing the data under the mask                                                 | masked row of 4.4; dropped; C9 closed                                                                                    |
| Q6  | Nested and `object` cells of frames left to the library                                               | : three kinds only, no cell children, the tolerance and `equal_nan` do not reach nested cells; C4 reworded; C6 left open |
| Q7  | Time resolution stays in the dtype                                                                    |                                                                                                                          |
| Q8  | Integers: exact difference, tolerance in `float64`                                                    | ; small floats cast to `float64` (R2)                                                                                    |
| Q9  | Ignored state stays ignored and is documented                                                         | guide sections                                                                                                           |
| Q10 | `bool` and every `Enum` exact                                                                         |                                                                                                                          |
| Q11 | A migration guide in `docs/docs/migration/`                                                           |                                                                                                                          |
| Q12 | Library testers stay public                                                                           | exports; open details in R3                                                                                              |
| Q13 | Per-call lookup cache                                                                                 |                                                                                                                          |
| Q14 | Engine PR with chain testers called as leaves                                                         |                                                                                                                          |
| R1  | Document the `NaN`-in-nested-cell exception                                                           | property rows of                                                                                                         |
| R2  | Cast small floats to `float64`                                                                        |                                                                                                                          |
| R3  | `TableEqualityTester(adapter)` only, `Engine`, export both path parts, helpers private except `close` |                                                                                                                          |
| R4  | Keep `max_depth=100` and `MaxDepthExceededError`                                                      |                                                                                                                          |
| R5  | A table is one leaf, reported as different, no row (the column name comes from S1)                    | , 7                                                                                                                      |
| S1  | The reason of a table names the first differing column (never a row)                                  |                                                                                                                          |
| S2  | Arrays keep their index in the path                                                                   | , 5                                                                                                                      |
| S3  | A `Series` says "series", an `Array` "arrays" (`kind_name`)                                           |                                                                                                                          |
| H1  | The hasher of masked arrays is left as it is (may change later)                                       | last section                                                                                                             |
| T1  | All steps land on a major-release branch; nothing is released before                                  |                                                                                                                          |
| T2  | Benchmark gates are reported and checked by hand, not a failing CI check                              |                                                                                                                          |
| T3  | The hashing follow-up is done after                                                                   | last section                                                                                                             |

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
access). Options: A. keep `max_depth=100` and `MaxDepthExceededError` (documented before, cheap, the
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
, and nothing is released until (one big break, a long-lived branch);
B. the non-breaking steps (1, 3, 4, 5) are released in the current minor releases, the breaking ones
wait for the major release, in order; C. all steps are released as they are merged, each breaking
one with a version bump.

Recommendation: B.
Decision: A

**T2. Are the benchmark gates a failing CI check?** `ci-benchmark.yaml` exists. The gates of
(regression above 10%, and the targets after, 10, 11) can fail the build, or be
reported only. Failing builds on timing are flaky on shared runners.
Options: A. fail the build on a regression above 10% against the stored baseline of the previous version, with a
generous margin for the absolute targets; B. report only, and check by hand at each step.

Recommendation: A for the relative gate, B for the absolute targets.
Decision: B

**T3. Who and when for the hashing follow-up?** The last section lists the hasher changes. Options:
A. a separate design document and PRs after; B. in parallel, after (the properties
exist), so that the cross-container properties also test the hash from the start.

Recommendation: B.
Decision: A

## Not covered

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
result; its output on the 5 000-entry dict, on cycles and on `max_depth` is the one of.

## Changes in other sub-packages (later, outside this design)

The design above is for `coola.equality` only. Equality is the reference: the changes below follow
from it and are made after (T3), one PR each, without driving any choice made above. They are listed
so that nothing is forgotten.

### Hashing (`coola.hashing`)

The property "equal objects have the same `hash_object`" has to hold with the new equality. What the
design implies for the hashers:

| Case                                                  | Equality (target)                       | Hasher before / change                                                                             |
| ----------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------- |
| pandas `object` column, `None` against `NaN`          | equal (the library's `equals`)          | hashes differ: hash a `None` and a `NaN` of an `object` column the same                            |
| `NaN` inside nested cells (lists, structs) of a frame | equal without `equal_nan` (the library) | check the hash of nested `NaN`: it must not depend on the NaN bit pattern                          |
| Nested floats of a frame, tolerance                   | exact (no tolerance)                    | nothing to do: equal values have equal hashes                                                      |
| Masked array, data under the mask                     | compared                                | the hasher ignores it; allowed (different values may share a hash); decide if it should include it |
| Integers above 2^53 in arrays, `bool` and enums       | exact                                   | check that the hash is exact (no `float64` conversion), and the same for `bool` and `Enum` members |
| Enum members, `IntEnum` against `int`                 | different types                         | the hash of an `IntEnum` member may equal the hash of its `int`: allowed                           |
| `MappingProxyType`, `array.array`                     | compared as a mapping and a sequence    | check they have a hasher consistent with a `dict` and a `list` of the same content                 |
| Time resolution (`timedelta64[s]` against `[ns]`)     | different dtypes                        | allowed to share a hash                                                                            |

After, the hashing property tests (`tests/properties/test_hashing*.py`) get the cross-container
equality properties of as a second source: any equal pair that the equality tests generate must hash
equally. Where the hasher needs a rule that is not obvious, the rule is taken from the equality
guide, not the other way round.

**H1. Should the hash of a masked array include the data under the mask?** Equality compares it;
the hasher ignores it before, which keeps "equal implies same hash" true. Including it makes the hash
as discriminating as equality, at the cost of changing every existing hash of a masked array.

Recommendation: no, leave the hasher as it is (the property holds).
Decision: no, leave the hasher as it is. It may change later.

### Packages that depend on the equality API

| Package                                                                     | Change                                                                                                                                                                                                                                                |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.io`, `coola.nested`, `coola.hashing.registry`, `coola.registry.base` | import `objects_are_equal` from `coola.equality`, not from the private `coola.equality.interface`                                                                                                                                                     |
| `coola.registry` (equality of two registries)                               | testers are compared with their `__eq__` (the `equal(other)` method of `BaseEqualityTester` is removed): the default registry registers the testers and handlers types of the removed chain with the default `==` tester, or no longer registers them |
| `coola.summary`, `coola.iterator`, `coola.recursive`, `coola.reducer`       | no change expected; check that none imports a removed handler or `HandlerCycleError`                                                                                                                                                                  |
| `coola.testing`                                                             | fixtures and helpers that use `show_difference` or the handler chain move to `compare` and `assert_*`                                                                                                                                                 |

## Walkthrough for contributors

The sections below describe how a comparison flows through the package and how to add a tester.

### Core Components

#### 1. Comparison Functions

The main entry points for users:

- **`objects_are_equal`**: Checks exact equality
- **`objects_are_allclose`**: Checks equality within tolerance

These functions provide a simple interface while delegating to the internal comparison system.

#### 2. Testers

A tester implements the comparison logic for one type (or family of types). Testers live in
`coola.equality.tester`.

##### `BaseEqualityTester`

Abstract base class defining the tester interface:

```python
class BaseEqualityTester(ABC, Generic[T]):
    @abstractmethod
    def objects_are_equal(self, actual: T, expected: T, ctx: Engine) -> bool | Iterator[tuple]:
        """A verdict, or the child pairs (actual, expected, path part) that must all be equal."""
```

##### `BaseEqualityTester` and the engine

The Python testers use the engine (`coola.equality.engine.Engine`). The engine owns the invariants of
a comparison: identity, strict type, depth, cycles and the location of the first difference. A tester
that derives from `BaseEqualityTester` never recurses: its `objects_are_equal(actual, expected,
ctx)` receives two objects of the same type and the engine, and returns a verdict (`True`, or
`ctx.fail(reason, *path)`, which returns `False`) or the child pairs `(actual, expected, path part)`
that must all be equal, produced lazily. The engine keeps the child iterators on an explicit stack, so
there is no `RecursionError` whatever the nesting, and a comparison stops at the first difference.

```python
class PointTester(BaseEqualityTester[Point]):
    def objects_are_equal(self, actual, expected, ctx):
        return ((actual.coords, expected.coords, AttributeName("coords")),)
```

The testers of NumPy, PyTorch, JAX and xarray use the engine too: an array tester checks the data
type and the shape (and the device for a tensor), then compares the elements with the shared kernel;
the location of a difference is the index of the first element that differs, looked up only when the
difference is explained. #### Built-in Testers

- **`DefaultEqualityTester`**: Fallback for any type (`==` on two objects of the same type)
- **`MappingEqualityTester`**, **`OrderedMappingEqualityTester`** (also checks the key order, used
  for `OrderedDict`), **`SequenceEqualityTester`** and **`SetEqualityTester`**: Return the children of
  mappings, sequences and sets
- **`DataclassEqualityTester`**, **`ObjectAttributesEqualityTester`** and
  **`PydanticModelEqualityTester`**: Return the fields or the attributes as children
- **`ScalarEqualityTester`**, **`DifferenceCloseEqualityTester`**, **`EqualEqualityTester`**,
  **`EqualNanEqualityTester`**, **`TolerantEqualEqualityTester`**: Numbers and objects with an
  `equal` method
- Array testers for NumPy, PyTorch, xarray and JAX (for example `NumpyArrayEqualityTester`,
  `TorchTensorEqualityTester`)
- **`TableEqualityTester`**: pandas, polars and PyArrow tables, with one adapter per library

#### 3. Registry

##### `EqualityTesterRegistry`

The registry maps types to testers. It is a `TypeRegistry` (`coola.registry`):

- `find_equality_tester(data_type)` returns the tester for the most specific registered type,
  walking the Method Resolution Order (MRO); `object` is registered with `DefaultEqualityTester`,
  so a tester is always found
- `objects_are_equal(actual, expected, config)` finds the tester for `type(actual)` and delegates (a tester that uses the engine runs a nested `Engine`)
  to it
- Lookup results are cached per type, and the cache is cleared whenever the registry changes

`get_default_registry()` returns the global registry. Testers for optional libraries are only
registered when the library is installed. `register_equality_testers(mapping, *, exist_ok=False)`
adds testers to it. A test that registers a tester should not leak it: use
`coola.registry.default.temporary_default_registries()` (see the testing guide), or work on
`get_default_registry().copy()`.

#### 4. Configuration

##### `EqualityConfig`

A frozen dataclass that carries the comparison settings. It can be shared between comparisons and
threads; the state of one comparison (the cache of testers, the first failure) lives in the `Engine`:

```python
@dataclass(frozen=True, slots=True)
class EqualityConfig:
    registry: EqualityTesterRegistry  # defaults to the default registry
    tolerance: Tolerance = Tolerance()  # rtol, atol
    equal_nan: bool = False
    max_depth: int = 100
```

The tester reads the settings from the engine (`ctx.atol`, `ctx.rtol`, `ctx.equal_nan`).

#### 5. Comparison Results

`compare` returns a `ComparisonResult`, and `assert_objects_equal` / `assert_objects_allclose`
raise an `AssertionError` with a description of the difference.

### Data Flow

Here's how a comparison flows through the system:

```
User calls objects_are_equal(obj1, obj2)
    ↓
Creates EqualityConfig with settings (and the registry)
    ↓
Runs Engine(config).run(obj1, obj2)
    ↓
For each pair: identity, strict type, tester lookup (cached per call, MRO lookup)
    ↓
Calls tester.objects_are_equal(obj1, obj2, engine): a verdict, or the child pairs
(pushed on the explicit stack, depth first)
    ↓
Returns boolean result
```

### Design Patterns

#### 1. Strategy Pattern

Testers implement different comparison strategies for different types, allowing the algorithm to
vary independently from the clients that use it.

#### 2. Template Method

Many testers follow a template:

1. Check types match
2. Check metadata (shape, dtype, etc.)
3. Check values
4. Optionally show differences

#### 3. Registry Pattern

The tester registry allows runtime type-to-tester mapping, enabling extensibility.

#### 4. Visitor Pattern

The recursive nature of comparison through nested structures follows a visitor-like pattern.

### Extension Points

#### Adding Support for New Types

To add support for a custom type:

1. **Implement a Tester:**

   ```python
   from coola.equality.config import EqualityConfig
   from coola.equality.tester import BaseEqualityTester


   class MyTypeEqualityTester(BaseEqualityTester[MyType]):
       def equal(self, other: object) -> bool:
           return type(other) is type(self)

       def objects_are_equal(self, actual: MyType, expected: object, config: EqualityConfig) -> bool:
           # Type check
           if type(actual) is not type(expected):
               return False

           # Custom comparison logic
           return actual.compare_to(expected)
   ```

2. **Register the Tester:**

   ```python
   from coola.equality.tester import register_equality_testers

   register_equality_testers({MyType: MyTypeEqualityTester()})
   ```

   To avoid modifying the global registry, create a `EqualityTesterRegistry`, register the tester
   on it and pass it with `registry=`.

3. **Use it:**

   ```python
   from coola.equality import objects_are_equal

   objects_are_equal(obj1, obj2)
   ```

See the [extending guide](../uguide/extending.md) for more details.

#### Checklist: writing a tester

- Return a verdict (`ctx.fail(reason)` for a difference) or the child pairs `(actual, expected,
path part)`; never recurse: the engine checks the identity, the type, the depth and the cycles
- Use `AttributeName` for an attribute and `ctx.fail(reason, *path)` for the location inside a leaf
  (the index in an array)
- Build the reason lazily (a callable), so `objects_are_equal` formats nothing
- Honour `ctx.equal_nan` and the tolerance (`ctx.rtol`, `ctx.atol`: `|a - e| <= atol + rtol *
max(|a|, |e|)`), or document that your tester ignores them

### Type System

#### Strict Type Checking

`coola` enforces strict type checking:

- `1` (int) ≠ `1.0` (float) ≠ `True` (bool)
- `list` ≠ `tuple`
- `dict` ≠ `OrderedDict`
- two `OrderedDict`s with the same items in a different order are different (the order of the keys of
  a `dict` is ignored)

This prevents subtle bugs from type coercion.

#### Type Hierarchy Support

Through MRO-based lookup, `coola` supports inheritance:

- A tester for `Sequence` applies to `list`, `tuple`, etc.
- More specific testers override general ones
- Custom subclasses inherit parent testers

### Performance Considerations

#### Early Exit

Testers check fast properties first:

1. Type check (very fast)
2. Metadata checks (fast: shape, dtype, device)
3. Value comparison (potentially slow)

#### Lazy Evaluation

Comparisons short-circuit on first difference when possible.

#### Caching

The registry caches tester lookups by type for performance.

#### Depth

The engine does not recurse (see [Design of `coola.equality`](#overview)), so a deep
structure cannot overflow the interpreter stack. `EqualityConfig.max_depth` (default 100) bounds the
work on a pathological structure and raises `MaxDepthExceededError`, a plain `Exception`.

### Error Handling

#### Graceful Degradation

When a specific tester is not available, `coola` falls back to:

1. More general tester (via MRO)
2. `DefaultEqualityTester` (registered for `object`), which uses `==`

#### Informative Messages

When the comparison explains the difference (`compare`, `assert_*`), testers report:

- What objects differ
- Where in the structure the difference is
- The actual values that differ
