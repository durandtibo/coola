# Review of `coola.equality`: target state (second pass)

Scope: `src/coola/equality` (54 files, 8.0k lines: 38 handler classes, 30 exported testers, one
registry, one config, one result type), checked against the **Guarantees** table of
`docs/docs/uguide/equality.md` (15 rows). Breaking changes are allowed.

Method: read every module of `config`, `interface`, `result`, `tolerance`, `tester/*` (the chains
and the registry) and `handler/{base,mixin,utils,native,scalar,sequence,mapping,set,object,equal,difference,tolerant,data,dataclass,allclose,shape,dtype,numpy,torch,jax,pandas,polars,pyarrow,pydantic}`; ran 6 probe scripts, a
benchmark and a 4 000-case fuzz against `main` (`e33c41b8`, Python 3.14, numpy 2.5, torch 2.14, pandas 3.0,
polars 2.0, pyarrow 26, jax 0.12, xarray 2026.9, pydantic 2.14); ran the code blocks of the three
equality guides with `doctest`. `tester/{xarray,pydantic}` were read for their chains only. Items marked **[verified]** were reproduced on `main`;
**[reading]** come from the code only. Nothing was changed in `src`, `tests` or `docs`.

This replaces the first-pass review. Its items E1 to E10 were implemented (table in section 1);
this pass found 20 new points, 4 of them wrong results or a crash (F1 to F4).

## 0. Summary

| ID  | P   | Finding                                                                                                                            |
| --- | --- | ---------------------------------------------------------------------------------------------------------------------------------- |
| F1  | P1  | `objects_are_allclose(np.timedelta64(1, "s"), np.timedelta64(1, "s"))` raises `TypeError` (default tolerance) **[verified]**       |
| F2  | P2  | NumPy, torch, JAX arrays do not follow the documented elementwise rule: `[100, 101.005]` vs `[101.005, 100]` **[verified]**        |
| F3  | P2  | polars ignores the tolerance for integers; pandas and polars use another rule than everything else **[verified]**                  |
| F4  | P2  | A pandas frame with `None`, `pd.NA` or `NaT` is not equal to its own copy; polars and pyarrow nulls are **[verified]**             |
| F5  | P2  | The default `max_depth=100` does not protect: 91 nested dicts raise a raw `RecursionError` **[verified]**                          |
| F6  | P2  | `show_difference=True` prints nothing by default, costs 5x when it prints nothing, and only exists on two functions **[verified]** |
| F7  | P2  | `compare` formats the whole objects before truncating: 243 ms instead of 11 ms for a list of 1M floats **[verified]**              |
| F8  | P3  | The guides contradict the code in four places, although CI runs their examples **[verified]**                                      |
| F9  | P3  | Input validation accepts `max_depth=True`, `max_depth=1.5`, `atol=True` **[verified]**                                             |
| F10 | P3  | `bool` has a tolerance for Python (`True` close to `False` with `atol=1`), not for NumPy **[verified]**                            |
| F11 | P3  | NaN dict keys and NumPy `NaT` scalars ignore `equal_nan`; sets honour it **[verified]**                                            |
| F12 | P3  | Durations: Python `timedelta` and `pd.Timedelta` have a tolerance, `timedelta64` arrays do not **[verified]**                      |
| F13 | P3  | Masked arrays: values under the mask are compared **[verified]**; decision needed                                                  |
| F14 | P3  | A difference in a data frame is not located (no column, no row); sets report the element as if it were an index **[verified]**     |
| A1  | P2  | The handler chain is a mutable linked list that recurses (2 frames per handler) and needs its own cycle and equality machinery     |
| A2  | P3  | Identity and type checks are copied into all 29 chain testers; the dispatcher could own them                                       |
| A3  | P2  | `EqualityConfig` still mixes settings and state; `registry` is passed twice; two ways to report                                    |
| A4  | P2  | Each library handler delegates to the testing module of the library, so six semantics coexist (F2, F3, F4, F12)                    |
| A5  | P3  | Public surface: 48 exports in `handler`, 35 in `tester`; library handlers are public API                                           |
| T1  | P2  | No equality property test for any array or frame type, no cross-container consistency test                                         |

Priority as in `suggestions.md`: P1 wrong result or crash on realistic input, P2 inconsistent or
surprising behaviour, P3 cleanup or design.

## 1. Where the first pass stands, and what holds

| First pass | Delivered                                                                                                                           | What is left                                                 |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| E1         | `Tolerance`, `compare(tolerance=)`, defaults in the table (#2308)                                                                   | `EqualityConfig` still holds two floats (A3)                 |
| E2         | `report_difference` without `logger` parameter (#2311)                                                                              | Two sinks remain (logger and `reporter`), see F6, F7         |
| E3         | Public methods on `EqualityConfig` (`mark_warned`, `visiting`, ...) (#2310)                                                         | Settings and state are still one mutable dataclass (A3)      |
| E4         | One formula for Python numbers, `Decimal`, `Fraction`, `timedelta` (#2309)                                                          | Arrays do not follow it (F2), pandas and polars neither (F3) |
| E5         | Not done                                                                                                                            | A5                                                           |
| E6 to E10  | Export of the errors, `max_depth` documented once, alias removed, `run_comparison` public, monotonicity properties (#2312 to #2316) | Nothing                                                      |

**Keep** (checked, matches the table): strict type at every level; order rules (`OrderedDict`);
signed zero in all 8 containers (`0.0` equals `-0.0`); time zones for `datetime`; unregistered types
fall back to `==`; cycles are co-inductive (`l = [l]` equals `m = [m]`, `[1, l]` differs from
`[2, m]`); `compare` is thread-safe (8 threads x 200 calls, no error); the depth counter is restored
after an exception; **hash consistency holds**: 15 equal pairs (`0.0/-0.0`, `Decimal("1")/Decimal("1.0")`,
`Fraction`, UTC against `+01:00`, C against F order, dict and set order, tensors with other strides, NaN
with `equal_nan`) have the same `hash_object`.

The dispatch by MRO, the handler composition as an idea, lazy per-library registration, `Tolerance`
as a value object, and the early exit of the fast path for lists of `int`/`str` (0.9 ms for 20 000
items) are all sound.

## 2. Conformance with the Guarantees table

| Row                     | Holds? | Evidence                                                                                                                 |
| ----------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------ |
| Strict type             | yes    | `{1: "a"}` vs `{True: "a"}` is `False`. **The guide says `True`** (F8)                                                   |
| Recursive               | yes    |                                                                                                                          |
| Order rules             | yes    |                                                                                                                          |
| Value semantics         | yes    | `defaultdict(int)` equals `defaultdict(list)` with the same items (same as `==`, P4)                                     |
| NaN                     | **no** | pandas treats `None`, `pd.NA`, `NaT` as NaN (F4); NumPy `NaT` scalars ignore `equal_nan` (F11)                           |
| Signed zero             | yes    | numpy, torch, jax, pandas, polars, pyarrow, xarray, list                                                                 |
| Time zones              | yes    | for `datetime`; a pandas series in `UTC` differs from the same instants in `Europe/Paris` (dtype), not stated in the row |
| Tolerance               | **no** | F2, F3, F10, F12                                                                                                         |
| Unregistered types      | yes    |                                                                                                                          |
| Reflexive               | **no** | `eq(df, df.copy())` is `False` for a column of `str` with a `None` (F4). The row only excepts "a distinct NaN"           |
| Symmetric               | yes    | 4 000 random float cases, 7 containers, 0 asymmetric answers (but see F2: symmetric and wrong)                           |
| Not transitive          | yes    |                                                                                                                          |
| Pure                    | yes    | a `LazyFrame` is collected, so its query runs on each comparison (documented)                                            |
| Failure modes           | **no** | "Unsupported types never raise" is false: `==` returning an array raises `TypeError` (documented two sections below), F1 |
| Consistent with hashing | yes    | see section 1                                                                                                            |

## 3. Findings

### F1. `np.timedelta64` scalars crash `objects_are_allclose` **[verified]**

`np.timedelta64` is a subclass of `np.signedinteger`, so `np.integer: ScalarEqualityTester()`
(`tester/numpy.py`) catches it, and `number_equal` calls `float(number)` on a `datetime.timedelta`.

```
objects_are_allclose(np.timedelta64(1, "s"), np.timedelta64(1, "s"))   # TypeError: float() argument ... 'datetime.timedelta'
objects_are_allclose([np.timedelta64(1, "s")], [np.timedelta64(1, "s")])  # same, inside a list
objects_are_equal(np.timedelta64("NaT"), np.timedelta64("NaT"), equal_nan=True)  # TypeError ... 'NoneType'
objects_are_equal(np.timedelta64(1, "s"), np.timedelta64(1, "s"))        # True (no tolerance, no equal_nan)
```

It is a crash with the default arguments of the most used function of the package, on a value that
appears in any dict built from a NumPy time series. `datetime64` scalars are not affected (they take
the `==` fallback), but then ignore `equal_nan` (F11).

**Target**: durations and instants are not integers. Register `np.timedelta64` and `np.datetime64` with
their own tester (see F12 and F11 for the semantics) and make `number_equal` total: it must never
call `float()` on something that is not a `numbers.Real`. Regression test for the three calls above.

### F2. Arrays are "symmetric" by trying both directions, which is not the elementwise rule **[verified]**

`handler/numpy.py`, `torch.py`, `jax.py` make the NumPy rule symmetric with
`allclose(a, b) or allclose(b, a)`. `allclose` is `all(...)`, so this is "all elements pass with the
tolerance scaled by `b`, **or** all pass with the tolerance scaled by `a`". The documented rule is
`|a - b| <= atol + rtol * max(|a|, |b|)` **per element**. They differ as soon as the elements need
different directions:

```python
x, y = [100.0, 101.005], [101.005, 100.0]          # rtol=0.01, atol=0
objects_are_allclose(x, y, rtol=0.01, atol=0.0)     # True   list (max rule holds for each pair)
objects_are_allclose(np.array(x), np.array(y), ...) # False  numpy
                     torch (float64) False, jax False, pyarrow False, xarray False
                     pandas True, polars True
```

The same numbers give three answers depending on the container. The fuzz of section 2 did not hit
this (n <= 3 and few values near the boundary); it needs two elements near the boundary, one in each
direction. The NumPy properties of `tests/properties/test_equality_types.py` use NumPy scalars, not
arrays.

**Target**: one kernel, section 4.5, `close = (a == b) | (finite & |a-b| <= atol + rtol * max(|a|,|b|))`,
elementwise, `& equal_nan` handled once. Symmetric by construction, no second pass.

### F3. polars: integers ignore the tolerance; pandas and polars use `max(atol, rtol * ...)` **[verified]**

```
ac(pl.Series([100]), pl.Series([101]), rtol=.01, atol=0)   # False   pandas, pyarrow, numpy, list: True
ac(pl.Series([1]),   pl.Series([2]),   atol=2)             # False   pandas True
ac(pd.Series([0.0]), pd.Series([0.9]), rtol=.5, atol=.5)   # False   numpy True, list True, polars False
```

`polars.testing` applies the tolerance to floats only, so the "Honored" cell of the polars row of the
"NaN and tolerance" table is wrong for integer columns. The pandas and polars rule (`math.isclose`)
is documented as a deliberate, stricter exception; it is still one more semantics to remember and
it makes `objects_are_allclose(frame, frame2)` and `objects_are_allclose(frame.to_numpy(), ...)`
disagree. See A4 for the target.

### F4. Missing values: pandas copies are not equal, polars and pyarrow copies are **[verified]**

| Object                                                        | `eq(x, copy)` | `eq(x, copy, equal_nan=True)` |
| ------------------------------------------------------------- | ------------- | ----------------------------- |
| `[None]`                                                      | `True`        | `True`                        |
| `pd.DataFrame({"a": ["x", None]})`                            | **`False`**   | `True`                        |
| `pd.Series(["a", None], dtype="string")`                      | **`False`**   | `True`                        |
| `pd.DataFrame({"a": pd.array([1, None], dtype="Int64")})`     | **`False`**   | `True`                        |
| `pd.DataFrame({"a": pd.to_datetime(["2020", None])})` (`NaT`) | **`False`**   | `True`                        |
| `pl.Series([1, None])`, `pa.array([1, None])`                 | `True`        | `True`                        |
| `pd.NaT`, `pd.NA` as scalars                                  | `True`        | `True`                        |
| `np.array(["NaT"], "M8[s]")`                                  | `False`       | `True`                        |
| `np.datetime64("NaT")` (scalar)                               | `False`       | **`False`** (F11)             |

The Guarantees table says "Reflexive: also for a copy of `x`. A distinct NaN is the exception".
`None` and `pd.NA` are not NaN, and a user who compares a frame with a text column to its copy
gets `False` with no hint (the message is the full dump of both frames). The scalar `pd.NA` and a
`None` in a list are equal to themselves; only the frame handlers decide otherwise, because pandas
cannot tell a missing float from a NaN and the handler took the conservative side for every dtype
(`if not config.equal_nan and isna().any(): return False`).

**Target** (section 4.2): two kinds of missing value, decided by the dtype.

- _NaN-like_ (float or complex `NaN`, `Decimal("NaN")`, NumPy `NaT`): governed by `equal_nan`.
- _Null_ (`None`, `pd.NA`, `pd.NaT` in a pandas non-float column, polars or Arrow null): equal to a
  null, always, whatever `equal_nan`.

In a pandas float column, the missing marker is a `NaN` (pandas has no other), so it stays NaN-like.
Behaviour change: frames with nulls in non-float columns become equal to their copy.

### F5. `max_depth=100` does not protect dictionaries **[verified]**

The guide says: "The default `max_depth` is below that [the interpreter limit], so a
`MaxDepthExceededError` with an explanatory message is raised instead of a `RecursionError`."

```
nesting at which the first error appears (sys.getrecursionlimit() == 1000):
list, max_depth=100   101 levels   MaxDepthExceededError   (as documented)
list, max_depth=1000  111 levels   RecursionError           (about 9 frames per level)
dict, max_depth=100    91 levels   RecursionError           (about 11 frames per level)
```

The cause is structural (A1): a handler chain recurses, each handler costs `handle` +
`_handle_next` (2 frames), and a mapping goes through 5 of them plus the tester and the registry. The
caller's own stack (a test runner, a web framework) is subtracted from the same 1000 frames, so the
limit comes sooner in practice. The error is still a `RecursionError` with advice, so nothing is
wrong silently; the claim of the guide and the default are.

**Target**: 3 frames per level independent of the chain length (A1), a test that nests 100 dicts,
lists, dataclasses and pydantic models under the default limit from a 100-frame-deep caller, and the
default `max_depth` stays 100.

### F6. `show_difference`: invisible, costly, inconsistent **[verified]**

- With the default logging configuration nothing is printed: `eq([1], [2], show_difference=True)`
  returns `False` and is silent (the logger is at `INFO`, the root at `WARNING`). The guide knows it
  and says to call `logging.basicConfig(level=logging.INFO)`.
- It is on `objects_are_equal` and `objects_are_allclose` only; `compare` hard-codes `True`, the
  `assert_*` functions do not have it, and the table "Common options" does not list it.
- The handlers format the message **before** the logger can drop it: for
  `W(x=list_of_1M_floats)` (dataclass) differing at index 0, `eq(...)` takes 43 ms and
  `eq(..., show_difference=True)` takes 240 ms with the logger disabled (`safe_str(value)` of the
  whole field is an argument of `report_difference`).
- Each level reports its own message (`numbers are different`, `sequences have different values at
index 2`, ...), and `compare` has to reassemble the path from them in reverse order
  (`for _, location in reversed(reports)`). A handler that reports twice, or not at all, breaks the path
  silently (the pyarrow handler gives `objects are not equal at actual` + a dump).

**Target** (section 4.4): one first-failure record produced by the context; reasons are built lazily,
only if somebody listens; `show_difference` is removed (decision D1) because `compare` and
`assert_*` already do the job and print something by default.

### F7. The cost of the failure path **[verified]**

```
eq(a, b)       a, b: 1M floats, differ at index 0     11.3 ms
compare(a, b)                                          243 ms      (safe_repr of both lists, then truncate to 1000)
compare(dataclass holding a dataclass holding the list)  702 ms
eq(a, b) 2M ints, differ at index 0                    110 ms      (the fast path scans the types of both lists first)
compare(a, b)  2M ints                                 176 ms
```

`ComparisonResult.actual/expected` are `truncate_str(safe_repr(obj), max_len=1000)`: the full `repr`
is built, then cut. The parent levels do the same with `safe_str(child)`. On the failure path of a
large test fixture, the assertion is 20x slower than the comparison (243 ms against 11 ms). `compare` on **equal** data is
also 1.8x slower than `objects_are_equal` (75 ms vs 42 ms for a dict of 5 000 entries; not
investigated).

**Target**: a bounded repr (`reprlib.Repr` with `maxlist`, `maxdict`, `maxstring`, `maxother`, and
the array libraries' own summarisation, which already truncates) used for the result and the reasons,
never the full `repr`. A benchmark of `compare` on 1M elements with a threshold (section 5).

### F8. The guides contradict the code, and CI does not run them **[verified]**

`python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE docs/docs/uguide/equality.md` runs 300
examples and fails 1: line 358, `objects_are_equal({1: "a"}, {True: "a"})` is documented as `True`
and returns `False` (the key handler compares `(type, key)` since #2088, which is what the Strict type
row asks for; the example was added by #2294). The text around it ("`{1: "a"}` and `{True: "a"}` are
still considered equal") is wrong too. `inv test.doctest` runs `python -m doctest` on every Markdown
file, so CI should report it; the recent `main` runs were still `queued` when this was written, so I
could not see whether it does. The two other guides pass. Other statements that do not match:

- Failure modes row: "Unsupported types never raise: they fall back to `==`". A class whose `==`
  returns an array raises `TypeError` (documented in "Types that work with caveats"), and a `==` that
  raises `KeyError` propagates unchanged.
- `Tolerance` docstring: "`|actual - expected| <= atol + rtol * |expected|`, as in `numpy.isclose`".
  The code and the guide use `max(|actual|, |expected|)`.
- `objects_are_equal` docstring: the **Note** is copied from `objects_are_allclose` ("even with a
  tolerance ... `True` and `False` are close with `atol=1`") in a function without tolerance.
- "Controlling Recursion Depth" says the default protects against the interpreter limit (F5).

**Target**: fix the four statements, and check that the Markdown doctest job fails on this example
(it is the only guard of the guides).

### F9. Validation accepts the wrong types **[verified]**

`max_depth=True`, `max_depth=1.5` and `atol=True` are accepted (`validate_positive` and
`validate_non_negative` compare to 0, `Tolerance(atol="a")` is rejected only because `"a" < 0`
raises). Target: `max_depth` must be an `int` that is not a `bool`; `rtol`/`atol` an `int` or a
`float`, not a `bool`.

### F10. `bool` has a tolerance in Python, not in NumPy **[verified]**

`ac(True, False, atol=1)` is `True`, `ac(np.bool_(True), np.bool_(False), atol=1)` is `False`, and
`ac(1, True, atol=1)` is `False` (type). The Note of `objects_are_allclose` says it. A boolean is a
label, not a quantity. **Target**: `bool` is compared exactly (`True` is close to `True` only);
numpy and torch boolean arrays currently accept a tolerance (`ac(np.array([True]), np.array([False]),
atol=1)` is `True`): same fix. Behaviour change.

### F11. `NaN` keys and `NaT` scalars ignore `equal_nan` **[verified]**

`eq({nan1: 1}, {nan2: 1}, equal_nan=True)` is `False` (dict keys are matched by `(type, key)`),
while `eq({nan1}, {nan2}, equal_nan=True)` is `True` (the set handler replaces a NaN by a marker).
`np.datetime64("NaT")` scalars fall to `==` and ignore `equal_nan`, their arrays honour it.
**Target**: reuse the `_key` function of the set handler for the keys of `MappingSameKeysHandler`, and
put `np.datetime64`/`np.timedelta64` on a tester that knows `NaT` (F1).

### F12. Durations **[verified]**

| Value                                               | tolerance               |
| --------------------------------------------------- | ----------------------- |
| `datetime.timedelta`, `pd.Timedelta`                | yes (`atol` in seconds) |
| `np.timedelta64` scalar                             | crash (F1)              |
| `np.array(..., "m8[s]")`, `pd.Series` of timedeltas | no (exact)              |
| `datetime`, `pd.Timestamp`, `datetime64`            | no                      |

The guide documents each row, but not the reason for the split. **Recommendation** (decision D4):
durations are quantities, instants are not. The tolerance applies to all duration types (scalar,
array, series), `atol` in seconds, `rtol` relative; instants stay exact. Cheapest: convert to
integer nanoseconds and use the kernel. The alternative is to remove the tolerance of `timedelta`
and `Decimal`-like durations, which is a regression for people who rely on it (#2244).

### F13. Masked arrays **[verified]**

`eq(ma.array([1, 2], mask=[0, 1]), ma.array([1, 3], mask=[0, 1]))` is `False`: the data under the
mask is compared (`SameDataHandler`), and so is `fill_value`. The data under a mask is unspecified (it
is what the array held before masking), so two arrays that print identically differ. **Recommendation**
(decision D3): compare the mask and the unmasked values only; keep `fill_value` as is, because it is part
of the visible state of the object (a result of `filled()`).

### F14. Differences in frames and sets are not located **[verified]**

- `compare(pa.table({"a": [1]}), pa.table({"a": [2]}))` gives `objects are not equal at actual` and a
  dump of both tables; pandas and polars give the same. A user comparing two frames with 200 columns
  must find the column alone. **Target**: path `['a'][3]` (column, row position), reason
  `values differ: 1 vs 2`; same for `Series` (`[3]`), arrays (`[2, 1]`, multi-index of the first
  difference), xarray (`['var'][...]`).
- `compare({1, 2}, {1, 3}).format_message()` is `objects are not equal at actual[2]`: the path component
  is the **element**, rendered like an index. A set has no position. **Target**: a path part type
  `Element(2)` rendered as `{2}`, next to `AttributeName`.

### A1. The handler chain: a recursion, a mutable list, a lot of guards

`BaseEqualityHandler` holds `_next_handler`; each `handle` ends with `self._handle_next(...)`. Consequences:

- Two frames per handler: a mapping level costs about 11 frames, a list level about 9 (F5).
- The chain is mutable after construction, hence `chain`, `chain_all`, `set_next_handler`,
  `_verify_next_handler` (cycle walk), `validate_chain`, `HandlerCycleError` (with a `RuntimeError`
  base kept "for compatibility"), `get_chain_length`, `visualize_chain`, and `handlers_are_equal` +
  `HandlerEqualityMixin` + `equality_attrs` + a `Protocol` (`EqualityHandler`) to compare chains by
  value. Most of `base.py` (370 lines) and `mixin.py` (123 lines) and part of `utils.py` is this
  machinery, none of it comparison logic.
- A terminal handler (`TrueHandler`, `ObjectEqualHandler`, library handlers) must not call
  `_handle_next`, a filter must; the base class cannot check it. Forgetting `TrueHandler` at the end of
  a chain raises `RuntimeError("next handler is not defined")` at comparison time, not at construction.
- All 29 chain testers start with the same two handlers (A2).

**Target**: a tester owns an immutable tuple of steps. A step answers `True` (equal, stop), `False`
(different, stop) or `None` (no verdict, next step):

```python
class Step(Protocol[T]):
    def check(self, actual: T, expected: T, ctx: ComparisonContext) -> bool | None: ...


@dataclass(frozen=True, slots=True)
class PipelineEqualityTester(BaseEqualityTester[T]):
    steps: tuple[Step[T], ...]  # frozen dataclass: repr, ==, hash, no mixin

    def objects_are_equal(self, actual: T, expected: T, ctx: ComparisonContext) -> bool:
        for step in self.steps:
            verdict = step.check(actual, expected, ctx)
            if verdict is not None:
                return verdict
        return True  # nothing objected: no TrueHandler
```

`SameLengthStep`, `MappingSameKeysStep`, ... return `None` or `False`; `SameObject` returns `True`
or `None`; terminal steps return a bool. A loop replaces the recursion: 3 frames per level (registry,
tester, step) instead of 9 to 11. `visualize_chain` is `"\n".join(map(str, steps))`; equality of
testers is the dataclass equality. Deleted: everything listed above plus `FalseHandler` / `TrueHandler`
(a tester without verdict is `True`; use a `Never` step for the rare always-false case).

An explicit stack (the container steps return the pairs to compare instead of recursing) would remove
`RecursionError` altogether and make `max_depth` optional. It is a bigger change for custom testers
and is **not** recommended now (decision D5); the pipeline leaves the door open because a step never
calls the next one itself.

### A2. Identity and type are tester business, but identical everywhere

All 29 chain-based testers (of the 30 exported ones) begin `SameObjectHandler(), SameTypeHandler()`. The registry already picks the
tester from `type(actual)`; if `type(expected)` differs the answer is `False` before any tester
runs, for every tester. Moving both checks to the dispatcher makes the Strict type and Reflexive rows
true **by construction** for custom testers too (today a custom tester that forgets `SameTypeHandler`
silently breaks the "Strict type" row and, through `type(actual)` dispatch, the "Symmetric" row), and
removes two steps from every comparison (`_handle_next` is the top entry of a `cProfile` of a dict of
5 000 entries: 45 005 calls).

```python
def compare_values(self, actual, expected) -> bool:  # ComparisonContext
    if actual is expected:
        return True
    if type(actual) is not type(expected):
        self.report_type_difference(actual, expected)
        return False
    return self.config.registry.find(type(actual)).objects_are_equal(actual, expected, self)
```

A tester that must accept two types (none today) sets `same_type_only = False`.

### A3. `EqualityConfig`: settings, state, and two registries

```python
@dataclass(slots=True)
class EqualityConfig:
    registry, equal_nan, atol, rtol, show_difference, max_depth, reporter  # settings
    _current_depth, _warned, _in_progress  # state of one run
```

- The docstring says "not thread-safe"; `compare` mutates `config.reporter` after construction.
- The registry is passed twice: `run_comparison(registry, actual, expected, config)` uses the first for the
  root and `config.registry` for every child. Passing different ones gives a half-custom comparison.
- `atol` and `rtol` are floats on the config, a `Tolerance` on `compare`, converted back and forth.
- Reporting has two sinks (`reporter`, else the logger) selected by a flag the handler must test
  (`if config.show_difference:`) before every `report_difference` (38 guards in `handler/`).

**Target**:

```python
@dataclass(frozen=True, slots=True)
class EqualityConfig:  # shareable, hashable, thread-safe
    registry: EqualityTesterRegistry
    tolerance: Tolerance = Tolerance()
    equal_nan: bool = False
    max_depth: int = 100


class ComparisonContext:  # one per call, never shared
    config: EqualityConfig

    def compare(
        self, actual, expected, part: PathPart | None = None
    ) -> bool: ...  # identity, type, depth, cycle, path
    def report(self, reason: str | Callable[[], str], *, part: PathPart | None = None) -> None: ...
    @property
    def explaining(self) -> bool: ...
    def warn_once(self, name: str, message: str) -> None: ...
```

`ctx.compare(child_a, child_e, part)` is the one way to recurse: it runs A2, increments the
depth, tracks the cycle and pushes `part` on the path. A container step cannot forget the depth check
or the cycle check (today a custom container handler that forgets `check_recursion_depth` or
`track_comparison` is a silent stack-overflow or infinite-loop bug).

### A4. Six semantics, one per library

| Library   | Values compared by              | Rule                          | Ints   | Missing values                  | Object cells                      |
| --------- | ------------------------------- | ----------------------------- | ------ | ------------------------------- | --------------------------------- |
| python    | `number_equal`                  | `atol + rtol*max`             | yes    | `NaN` by `equal_nan`            | n/a                               |
| numpy     | `np.allclose` both ways, `all`  | `atol + rtol*b`, `or` swapped | yes    | `NaN`, `NaT` by `equal_nan`     | through the registry              |
| torch/jax | `allclose` both ways, `all`     | same as numpy                 | yes    | `NaN` by `equal_nan`            | n/a                               |
| pandas    | `pd.testing.assert_*_equal`     | `max(atol, rtol*max)`         | yes    | all as NaN                      | by pandas (no `equal_nan` inside) |
| polars    | `polars.testing.assert_*_equal` | `max(atol, rtol*max)`         | **no** | NaN by `equal_nan`, null always | by polars                         |
| pyarrow   | `equals`, then NumPy per column | numpy                         | yes    | NaN by `equal_nan`, null always | exact                             |

(`pandas` object cells: `eq(pd.DataFrame({"a": [[nan]]}), same, equal_nan=True)` is `False`; the cells are
compared by pandas, not by the registry.) Every difference in this table was found by a probe, and
each is a place where the same data in another container answers differently.

**Target**: coola owns the semantics; the library is used for _structure_ only.

1. A private kernel `coola.equality._numeric` (section 4.5) with one function per backend
   (NumPy, torch, JAX) that return the same boolean array for the same numbers.
2. Frame handlers (`pandas`, `polars`, `pyarrow`) check the structure with the library
   (`shape`, `columns`, `index`, `dtypes`, `names`, `attrs` policy), then compare the **columns**:
   numeric columns through the kernel (nullable columns: compare the null masks first, F4), other
   columns exactly, object columns cell by cell through the registry. pandas and polars drop their
   private rule; the guide's "NaN and Tolerance" table loses its pandas/polars footnote.
3. The location of the first difference (column, row) comes for free (F14).

### A5. Public surface

`coola.equality.handler.__all__` has 48 names (38 handler classes); `tester.__all__` 35. The library
handlers (`PandasDataFrameEqualHandler`, `TorchTensorSameDeviceHandler`, ...) are documented API,
which means a change in how pandas frames are compared is a breaking change for people who composed
them. Carried over from the first pass and still valid:

- Public and stable: the five functions, `ComparisonResult`, `Tolerance`, `EqualityConfig`,
  `ComparisonContext`, `MaxDepthExceededError`, `AttributeName` and the other path parts,
  `BaseEqualityTester`, `PipelineEqualityTester`, `EqualityTesterRegistry`, `register_equality_testers`,
  `get_default_registry`, the `Step` protocol and the generic steps (`SameLength`, `SameShape`,
  `SameDType`, `SameAttribute`, `ObjectEqual`, `Mapping*`, `Sequence*`, `Set*`), and the named testers
  of the guide (`ObjectAttributesEqualityTester`, `EqualEqualityTester`, `SequenceEqualityTester`, ...).
  About 40 names instead of 96 (`handler` 48, `tester` 35, `equality` 9, `config` 1, `interface` 3).
- Private (`coola.equality._steps.<lib>`, no re-export): steps and testers of numpy, torch, jax, pandas,
  polars, pyarrow, xarray, pydantic. They are reached through the registry.
- `coola.io`, `coola.hashing.registry`, `coola.registry.base`, `coola.nested` import
  `coola.equality.interface.objects_are_equal` (private module path): import from `coola.equality`.

### T1. Tests

- `tests/properties` has 100+ properties for Python scalars and dataclasses and **none** for arrays or
  frames: `test_library_types.py` is about hashing and `summarize`. F2, F3, F4 were all invisible
  to it. Missing: for any `list[float]`, `objects_are_allclose` agrees between a list, a NumPy array, a
  float64 tensor, a pandas series, a polars series, a pyarrow array and an xarray array, for random `rtol`,
  `atol`, `equal_nan` (the property that catches F2 and F3 in one run); reflexivity for a copy for each
  library type, with and without nulls (F4); symmetry for each library type.
- The guides are executed by CI, but F8 shows an example that fails (check that the job reports it).
- Nobody tests the depth guarantee under a deep caller stack (F5).
- Mutation testing of `handler/` is still open (`suggestions.md`).

## 4. Target state

### 4.1 The semantics, as the Guarantees table should read

Unchanged rows: Strict type, Recursive, Order rules, Value semantics, Signed zero, Time zones,
Unregistered types, Pure, Not transitive, Consistent with hashing. Changed or added:

| Row            | Statement                                                                                                                                                                                                                                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Closeness      | Two numbers are close if `abs(a - b) <= atol + rtol * max(abs(a), abs(b))`, **per element**, for every numeric type and container (list, NumPy, torch, JAX, pandas, polars, pyarrow, xarray). Integers use exact arithmetic. Two different values are never close if one is infinite. `bool` has no tolerance. |
| Missing values | A _null_ (`None`, `pd.NA`, `pd.NaT` in a non-float pandas column, a polars or Arrow null) equals a null, always. A _NaN-like_ value (float or complex `NaN`, `Decimal("NaN")`, NumPy `NaT`) equals another only with `equal_nan=True` or if it is the same object.                                             |
| Reflexive      | `objects_are_equal(x, x)` is `True` for every `x`, and also for a copy of `x`, **unless `x` contains a NaN-like value** and `equal_nan` is `False`.                                                                                                                                                            |
| Durations      | The tolerance applies to durations (`timedelta`, `timedelta64`, `pd.Timedelta`; `atol` in seconds). Instants are always exact.                                                                                                                                                                                 |
| Failure modes  | Cycle: equal. Nesting above `max_depth` (default 100, guaranteed to be reached before the interpreter limit): `MaxDepthExceededError`. `==` that raises `TypeError`/`ValueError` or returns a non-bool: `TypeError` that names the type. Any other exception of `==` propagates.                               |
| Location       | `compare` and `assert_*` give the path of the first difference: keys, indexes, attributes, **column and row** of a frame, **element** of a set.                                                                                                                                                                |

### 4.2 Missing values by type

| Type                               | NaN-like (by `equal_nan`)      | Null (always equal)    |
| ---------------------------------- | ------------------------------ | ---------------------- |
| `float`, `complex`, NumPy floating | `NaN`                          |                        |
| `Decimal`                          | `NaN`, `sNaN`                  |                        |
| NumPy `datetime64`/`timedelta64`   | `NaT`                          |                        |
| NumPy `object` array               | through the registry, per cell |                        |
| pandas float column                | the missing marker             |                        |
| pandas other columns               |                                | `None`, `pd.NA`, `NaT` |
| pandas nullable `Float64`          | `NaN`                          | `pd.NA`                |
| polars                             | float `NaN`                    | `null`                 |
| pyarrow                            | float `NaN`                    | `null`                 |
| xarray                             | `NaN`, `NaT`                   |                        |

### 4.3 Dispatch, steps, context (A1, A2, A3)

```
objects_are_equal(a, e, *, equal_nan, max_depth, registry)   -> context without reporter -> bool
objects_are_allclose(a, e, *, rtol, atol, equal_nan, ...)     -> same, with a tolerance
compare(a, e, *, tolerance, equal_nan, max_depth, max_repr_chars, registry) -> ComparisonResult
assert_objects_equal / assert_objects_allclose                -> compare + AssertionError

ComparisonContext.compare(a, e, part)
   identity -> type -> depth -> cycle -> registry.find(type(a)) -> tester.objects_are_equal(a, e, ctx)
   PipelineEqualityTester: for step in steps: verdict = step.check(a, e, ctx)
```

### 4.4 Reporting (F6, F7, F14)

- `ctx.report(reason, part=None)` records the **first** failure only: its reason and the path
  accumulated by `ctx.compare` at that moment. No list of reports, no reversal, no reasons from parent
  levels (the path says where).
- `reason` may be a `str` or a zero-argument callable. It is called only if `ctx.explaining`, which is
  `False` for `objects_are_equal` and `objects_are_allclose`. Nothing is formatted then.
- Values in reasons and in `ComparisonResult.actual/expected` go through one bounded formatter.
- `show_difference` and the logger sink are removed (D1). `compare` / `assert_*` are the way to
  explain. `logging` is used by nobody in the package; a user who wants a log line writes
  `log.info(compare(a, b).format_message())`.
- Path parts: `AttributeName("x")` -> `.x`, `Element(2)` -> `{2}`, keys and positions -> `[k]`; a frame
  reports `("a", 3)` (column, row position).

### 4.5 The numeric kernel (F2, F3, F10, F12)

```python
# coola/equality/_numeric.py (private)
def close(a, b, tolerance: Tolerance, equal_nan: bool, xp) -> "bool array":
    eq = a == b
    finite = xp.isfinite(a) & xp.isfinite(b)
    near = finite & (
        xp.abs(a - b) <= tolerance.atol + tolerance.rtol * xp.maximum(xp.abs(a), xp.abs(b))
    )
    out = eq | near
    return out | (xp.isnan(a) & xp.isnan(b)) if equal_nan else out
```

`xp` is `numpy`, `torch` or `jax.numpy` (they share these functions; torch needs `torch.maximum`).
Booleans are compared with `==`. Integer dtypes are cast to `float64` before the subtraction (an
unsigned `100 - 200` would wrap around), which is exact below 2^53; above, the kernel falls back to the
exact path that `int_isclose` already implements for Python integers. Durations are converted to integer nanoseconds first. pandas,
polars and pyarrow columns are converted with `to_numpy` (nullable: the null masks are compared first,
then `fill_null(0)`, as the pyarrow handler already does). Python scalars use the same formula
(`number_equal` already does).

## 5. Migration plan

One PR per row, as `CLAUDE.md` asks. "B" marks a breaking change.

| Step | Content                                                                                                                                                                | Closes         | B   | Depends on |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | --- | ---------- |
| 1    | `np.timedelta64`/`np.datetime64` testers; `number_equal` never calls `float()` on a non-number                                                                         | F1             |     |            |
| 2    | Elementwise symmetric kernel for numpy, torch, jax; cross-container property test (list vs array, float inputs)                                                        | F2, part of T1 |     |            |
| 3    | Guides: fix the four statements; find out why the Markdown doctest job did not stop the failing example                                                                | F8             |     |            |
| 4    | Type validation of `max_depth`, `rtol`, `atol` (no `bool`, no `float` depth)                                                                                           | F9             | B   |            |
| 5    | `bool` without tolerance (Python, NumPy, torch)                                                                                                                        | F10            | B   | 2          |
| 6    | Missing-value policy: null equal to null in pandas non-float columns; NaN keys and `NaT` scalars honour `equal_nan`                                                    | F4, F11        | B   | 1          |
| 7    | Durations: tolerance for `timedelta64` arrays/series (or removal, D4)                                                                                                  | F12            | B   | 2          |
| 8    | Masked arrays: ignore the data under the mask (D3)                                                                                                                     | F13            | B   |            |
| 9    | `PipelineEqualityTester` and `Step`; identity and type check in the dispatcher; remove the chain machinery; depth test under a deep caller                             | A1, A2, F5     | B   | 3          |
| 10   | `EqualityConfig` frozen + `ComparisonContext` with `compare`/`report`/`warn_once`; first-failure record; lazy bounded reasons; remove `show_difference` and the logger | A3, F6, F7     | B   | 9          |
| 11   | Library frames on the kernel with structure checks; polars integers; path to the column and row; `Element` path part                                                   | A4, F3, F14    | B   | 2, 6, 10   |
| 12   | Private library steps; `__all__` of `handler` and `tester`; imports of `coola.io` and others from `coola.equality`                                                     | A5             | B   | 9, 11      |
| 13   | Property tests per library type (reflexive for a copy, symmetric, equals the list rule); benchmark gates; mutation testing of `equality/`                              | T1             |     | 11         |

Steps 1 to 5 are small, independent and fix wrong answers: do them first. Steps 9 to 12 break every
custom handler (`handle(actual, expected, config)` becomes `check(actual, expected, ctx)`), so they
share one release with a migration note (`docs/docs/migration`) showing the 3-line change of a
handler.

Benchmark gates to add in `tests/benchmarks/test_equality_benchmark.py` before step 9, from the numbers
measured here: 20 000 floats in a list 9.4 ms; dict of 5 000 x `{float, [int, int]}` 41.8 ms; 20 000 small
arrays 36.6 ms (`allclose`: 200 ms); `compare` of 1M floats differing at 0, 243 ms (target < 30 ms).
Fail on a regression above 10%.

## 6. Decisions for the maintainers

| ID  | Question                                                                          | Recommendation                                                             |
| --- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| D1  | Remove `show_difference` and the logger sink?                                     | Yes. `compare` and `assert_*` cover it; README (28.4) to be updated        |
| D2  | Nulls of non-float columns always equal (F4)?                                     | Yes: a copy of a frame must equal the frame                                |
| D3  | Masked arrays: ignore the data under the mask (F13)?                              | Yes; keep `fill_value` compared                                            |
| D4  | Tolerance for all durations, or for none (F12)?                                   | All, `atol` in seconds                                                     |
| D5  | Recursion or explicit stack?                                                      | Recursion with 3 frames per level; revisit if depth > 300 is a requirement |
| D6  | Is the break of the custom-handler API (`Step`, `ctx`) acceptable in one release? | Yes, with a migration note in `docs/docs/migration`                        |

## 7. Not covered

- Per-library numerical corner cases beyond the probes: `float16`/`bfloat16` rounding in the kernel,
  `int64` differences above 2^53 (a probe of `-2**63` against `2**63 - 1` with `atol=1` gave the right
  answer, but the NumPy path goes through `float64`), sparse and quantized torch tensors (compared,
  semantics not checked), nested tensors (raise a `RuntimeError` from torch), dask-backed xarray.
- The thread safety of the registry (`find` cache), fork behaviour, GPU and Windows.
- Why `compare` is 1.8x slower than `objects_are_equal` on equal data (not profiled; do it before
  step 10).
- `pydantic` private attributes and `attrs` of pandas frames are ignored by design (documented); not
  re-evaluated.
