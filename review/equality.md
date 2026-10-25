# Review of `coola.equality`: target state (third pass)

Scope: `src/coola/equality` (54 files, 8.6k lines: 50 exported names in `handler`, 36 in `tester`, one
registry, one config, one result type), checked against the **Guarantees** table of
`docs/docs/uguide/equality.md` (15 rows) and the rest of the three equality guides. Breaking changes
are allowed; the future state matters more than the past one.

Method: read every module of the package (`config`, `interface`, `result`, `tolerance`, `tester/*`,
`handler/*`); ran 5 probe scripts (about 250 calls), a benchmark of the failure paths, a profile and a
**prototype of the target dispatcher** outside `src` (section 5.8) against `main` (`bb94c6fd`, Python
3.14, numpy 2.5, torch 2.14, pandas 3.0, polars 2.0, pyarrow 26, jax 0.12, xarray 2026.9, pydantic
2.14); ran the code blocks of the three equality guides with `doctest`. Items marked **[verified]**
were reproduced on `main`, **[reading]** come from the code only. Nothing was changed in `src`,
`tests` or `docs`.

This replaces the second-pass review (still in `git log`, commit `c107887b`). Of its 14 findings, 10
are merged and verified (section 1). This pass found 9 new points (G1 to G9), 2 of them wrong
answers on numbers (G1, G8), and re-measures the open ones.

## 0. Summary

The semantics are now close to the Guarantees table. What is left falls in three groups:

1. **Six places where the same data gets another answer** than in a list, a NumPy array or a Python
   number (G1 to G4, G6, G8). All of them come from one cause: a library still decides part of the
   semantics (pandas, polars and pyarrow for non-numeric columns and for the structure; NumPy's dtype
   for big integers and small floats).
2. **The architecture costs most of the code, the speed and the diagnostics.** The handler chain
   (a mutable linked list of 50 classes) needs 500 lines of its own machinery, copies the same two
   checks into 29 testers, uses 6.4 Python frames per nesting level, reports a difference through
   two sinks, and cannot say where the difference is in an array or a frame. A plain-method tester
   with one context object does the same job 4 to 7 times faster, with 1.8 frames per level
   (prototype, section 5.8).
3. **Documentation drift**: the guide still says that a pandas frame with a missing value is never
   equal to its copy, and `extending.md` fails its doctest.

| ID  | P   | Finding                                                                                                                         |
| --- | --- | ------------------------------------------------------------------------------------------------------------------------------- |
| G1  | P2  | Integers beyond 2^53 are compared as `float64` in arrays, tensors and columns: `2**62` is close to `2**62 + 1` with `atol=0.5`  |
| G2  | P2  | `IntEnum` and `IntFlag` members are numbers: two different members are "close" with a large tolerance                           |
| G3  | P2  | pandas booleans (`Series`, `DataFrame`, `Index`, `"boolean"` arrays) have a tolerance: `True` is close to `False` with `atol=1` |
| G4  | P2  | Nested values (lists, structs) in frames: polars applies the tolerance, pyarrow ignores it, pandas decides cell by cell         |
| G5  | P2  | A **failing** exact pandas comparison is 2 000x slower than a passing one: 5.9 s for 2M floats that differ at index 0           |
| G6  | P2  | pandas `object` columns break two rows: NaN is equal to NaN without `equal_nan`, and `None` equals `NaN` but hashes differently |
| G7  | P3  | Virtual subclasses of `Mapping` and `Sequence` (`MappingProxyType`, `array.array`) fall to `==`: no tolerance, arrays raise     |
| G8  | P3  | `float16`/`float32` arrays compute in their own dtype: overflow warning and a wrong `True` at extreme values                    |
| G9  | P3  | Docs: stale pandas paragraph, `show_difference` still a design principle, `extending.md` import error, copied docstring note    |
| F6  | P2  | _(open)_ `show_difference=True` is silent by default, on two functions only, and formats before it knows anybody listens        |
| F7  | P2  | _(open)_ `compare` on a failure builds the `repr` of everything: 231 ms against 10 ms for a list of 1M floats                   |
| F13 | P3  | _(open)_ Masked arrays: the data under the mask is compared, although the hasher ignores it                                     |
| F14 | P3  | _(open)_ A difference inside an array, a frame or a set is not located (path is empty, or the element is shown as an index)     |
| A1  | P2  | _(open)_ The handler chain: 6.4 frames per level, 830 000 calls for 5 000 entries, 500 lines of chain machinery                 |
| A2  | P3  | _(open)_ Identity and type checks are copied into 29 testers; the dispatcher should own them                                    |
| A3  | P2  | _(open)_ `EqualityConfig` mixes settings and state; `registry` is passed twice; `compare` has a different tolerance signature   |
| A4  | P2  | _(partly done)_ Numbers follow one rule, but pandas, polars and pyarrow still own structure and non-numeric columns (G4 to G6)  |
| A5  | P3  | _(open)_ Public surface: 50 handler names and 36 tester names, library handlers included                                        |
| T1  | P2  | _(open)_ No property test compares a list, an array, a tensor, a series and an Arrow array on the same random numbers           |

Priority as in `suggestions.md`: P1 wrong result or crash on realistic input, P2 inconsistent or
surprising behaviour, P3 cleanup or design. There is no P1 left.

## 1. What the second pass delivered, and what holds

| Second pass | Merged                                                                     | Verified on `main`                                                    |
| ----------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| F1          | `np.timedelta64` scalars no longer crash (#2318)                           | yes                                                                   |
| F2          | One elementwise symmetric kernel `handler/elementwise.py` (#2325)          | yes: `[100, 101.005]` against `[101.005, 100]` agrees in 7 containers |
| F3          | pandas and polars integers and floats use the shared rule (#2327)          | yes, except for pandas booleans (G3) and ints above 2^53 (G1)         |
| F4          | pandas nulls equal nulls, only a float `NaN` follows `equal_nan` (#2326)   | yes for frames; **not** for `object` columns (G6)                     |
| F5          | `max_depth` protects (#2328)                                               | yes: 99 nested dicts from a 150-frame caller; fails from 300 (see A1) |
| F8          | Four statements of the guide fixed (#2319)                                 | the example passes; other statements went stale since (G9)            |
| F9          | `max_depth`, `rtol`, `atol` validated by type (#2320)                      | yes                                                                   |
| F10         | `bool` compared exactly (#2321)                                            | Python, NumPy, torch, JAX, polars yes; **pandas no** (G3)             |
| F11         | NaN dictionary keys honour `equal_nan` (#2322)                             | yes                                                                   |
| F12         | Durations: tolerance for `timedelta64` arrays and pandas durations (#2324) | yes (`1000 ms` against `1500 ms`, `atol=1`: close)                    |
| 23.1        | Equal pydantic models with a read `cached_property`                        | yes: equal                                                            |

**Keep** (checked again, matches the table): strict type at every level; order rules; signed zero
(NumPy, torch, JAX, pandas, polars, pyarrow); time zones; unregistered types fall back to `==`; cycles
are co-inductive; `compare` is thread-safe (8 threads x 200 calls, no error); a pandas frame with
`None`, `pd.NA` or `NaT` equals its copy and has the same hash; `compare` on equal data now costs the
same as `objects_are_equal` (43.8 ms and 45.2 ms for a dict of 5 000 entries; the 1.8x of the second
pass is gone).

The dispatch by MRO, lazy per-library registration, `Tolerance` as a value object, the early exit of
the fast path for lists of `int` and `str`, and the one kernel `elementwise_close` are sound and stay.

## 2. Conformance with the Guarantees table

| Row                     | Holds? | Evidence                                                                                                                 |
| ----------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------ |
| Strict type             | yes    | `{1: "a"}` against `{True: "a"}`, `np.float64(1)` against `1.0`, a subclass: all `False`                                 |
| Recursive               | yes    |                                                                                                                          |
| Order rules             | yes    |                                                                                                                          |
| Value semantics         | yes    |                                                                                                                          |
| NaN                     | **no** | pandas `object` column: two distinct `NaN` objects are equal without `equal_nan` (G6)                                    |
| Signed zero             | yes    |                                                                                                                          |
| Time zones              | yes    |                                                                                                                          |
| Tolerance               | **no** | G1 (big ints), G2 (enums), G3 (pandas `bool`), G4 (nested), G8 (small floats). "Applies only to numbers" is broken by G2 |
| Unregistered types      | yes    |                                                                                                                          |
| Reflexive               | yes    | a pandas frame with `None`, `NaT`, `pd.NA` equals its copy                                                               |
| Symmetric               | yes    | `tests/integration/equality/test_allclose_symmetry.py` (13 tests) and the 4 000-case fuzz of the second pass             |
| Not transitive          | yes    |                                                                                                                          |
| Pure                    | yes    | a `LazyFrame` is collected on each comparison (documented)                                                               |
| Failure modes           | **no** | `eq(MappingProxyType({"a": np.ones(2)}), MappingProxyType({"a": np.ones(2)}))` raises `TypeError` (G7)                   |
| Consistent with hashing | **no** | `pd.Series([None, 1.0], dtype=object)` equals `pd.Series([nan, 1.0], dtype=object)`, hashes differ (G6)                  |

## 3. Findings

### G1. Integers beyond 2^53 are rounded in arrays **[verified]**

`handler/numpy.py`, `torch.py`, `jax.py` cast the integers to `float64` before the subtraction ("an
unsigned subtraction wraps around"). Above 2^53 the cast rounds, so `2**62` and `2**62 + 1` become the
same number:

```
ac(2**62, 2**62 + 1, atol=0.5, rtol=0)                              False  Python int (exact, Fraction)
ac([2**62], [2**62 + 1], ...)                                       False  list
ac(np.array([2**62]), np.array([2**62 + 1]), ...)                   True   numpy
ac(torch.tensor([2**62]), torch.tensor([2**62 + 1]), ...)           True   torch
ac(pd.Series([2**62]), ...)  ac(pl.Series([2**62]), ...)  ac(pa.array([2**62]), ...)   True   (they call the NumPy handler)
ac(np.array([2**64 - 1], "u8"), np.array([2**64 - 2], "u8"), ...)   True
```

The Tolerance row says "integers use exact arithmetic" in the table of `equality.md` (`exact for large int`),
and a list gives the right answer, so the same ids in an array give another one. The second pass
listed it as "not covered"; it is now cheap to fix because there is one kernel.

**Target** (section 5.5): the difference of two integers is computed exactly (as `uint64`: for a
signed pair the wrap-around of the unsigned subtraction is the exact absolute difference when the
larger value is first), then compared with the tolerance in `float64`.

### G2. `IntEnum` and `IntFlag` are numbers **[verified]**

```
class C(enum.IntEnum): A = 1; B = 2
ac(C.A, C.B, atol=5)   True      ac(E.A, E.B, atol=5)   False   (plain Enum)
ac(F.X, F.Y, atol=5)   True      (IntFlag)
```

`IntEnum` is a subclass of `int`, and the registry walks the MRO (`IntEnum, int, ReprEnum, Enum,
object`), so `int` wins over `Enum`. An enum member is a label, as a `bool` is. Two members that
differ are different whatever the tolerance. **Target**: the number tester compares exactly when
`isinstance(actual, (bool, enum.Enum))` (a registration cannot do it: any `class X(int, Enum)` would
escape).

### G3. pandas booleans have a tolerance **[verified]**

F10 fixed `bool`, `np.bool_`, boolean arrays and tensors, polars. pandas was missed because the
handler hands `bool` columns to `pd.testing`:

```
ac(pd.Series([True]), pd.Series([False]), atol=1)                                   True
ac(pd.DataFrame({"a": [True]}), pd.DataFrame({"a": [False]}), atol=1)               True
ac(pd.Index([True]), pd.Index([False]), atol=1)                                     True
ac(pd.array([True], dtype="boolean"), pd.array([False], dtype="boolean"), atol=1)   True
ac(pl.Series([True]), pl.Series([False]), atol=1)                                   False   (correct)
```

`_is_numeric_pair` selects integer and float dtypes, so booleans go to `assert_*_equal(atol=...)`.
**Target**: a column kind "exact" (section 5.7) that never sees the tolerance.

### G4. Nested values in frames follow three rules **[verified]**

```
ac(pl.Series([[1.0]]), pl.Series([[1.0 + 1e-9]]))      True    polars applies the tolerance inside the list
ac(pa.array([[1.0]]),  pa.array([[1.0 + 1e-9]]))       False   pyarrow: exact (the guide says so)
ac(pd.Series([[1.0]], dtype=object), ...)              True    pandas: by pandas, not by coola
eq(pd.DataFrame({"a": [[nan]]}), same, equal_nan=True) False   cells are compared by pandas: equal_nan is not applied
```

The Tolerance row promises one rule for "numbers, arrays, tensors and data frames". A list cell in a
frame is the same list as in a Python list, where `ac([[1.0]], [[1.0 + 1e-9]])` is `True`.
**Target**: nested and `object` cells are compared one by one through the registry (`to_list()`,
`to_pylist()`), so they follow the rules of their Python equivalent. It is slower than a vectorised
call, so the exact path with no `NaN` and no tolerance keeps the library's `equals`.

### G5. A failing exact pandas comparison is 2 000x slower than a passing one **[verified]**

```
                    equal copy   differ at index 0      (2M floats; frames: 200k x 10)
numpy                  4.6 ms           1.1 ms
torch                  1.7 ms           0.3 ms
polars Series          8.2 ms           1.6 ms
pyarrow                1.0 ms           0.7 ms
pandas Series          2.9 ms       5 897.7 ms
pandas DataFrame       3.0 ms         538.2 ms           (polars DataFrame: 0.9 ms)
```

`series_equal` and `frame_equal` call `pd.testing.assert_*_equal` and catch the `AssertionError`.
When the data differ, pandas builds its message (the percentage of different values, the formatted
operands) before raising, and coola throws it away. A test suite that compares large frames pays it
on every failure, and `objects_are_equal` used as a predicate (a cache key check, a "did it
change?" loop) hangs. **Target**: no use of `pandas.testing` (section 5.7); a column comparison is a
boolean array and `all()`.

### G6. pandas `object` columns break the NaN and the hashing rows **[verified]**

```
eq(pd.Series([1.0, nan1], dtype=object), pd.Series([1.0, nan2], dtype=object))      True   # equal_nan=False, two distinct NaN
eq(pd.Series([None, 1.0], dtype=object), pd.Series([nan, 1.0], dtype=object))       True   # None equals NaN, but hash_object differs
eq(pd.Series([1.0, nan1]), pd.Series([1.0, nan2]))                                  False  # the float64 column follows the rule
```

pandas treats `None` and `NaN` as the same missing value in an `object` column, and its comparison is
strict about neither. The NaN row says "NaN is not equal to NaN, unless `equal_nan=True` or both are
the same object", and "Consistent with hashing" says that equal objects have the same hash. The same
data in a `float64` column and in an `object` column give two answers. **Target**: the `object` cells
go through the registry (G4): `None` against `nan` differ by type, and a `NaN` follows `equal_nan`.

### G7. Virtual subclasses of `Mapping` and `Sequence` fall to `==` **[verified]**

The registry walks `type.__mro__`. `types.MappingProxyType` and `array.array` are registered with
`collections.abc` (virtual subclasses), which the MRO does not show:

```
ac(MappingProxyType({"a": 1.0}), MappingProxyType({"a": 1.0 + 1e-9}))      False   no tolerance
eq(MappingProxyType({"a": nan}), MappingProxyType({"a": nan}), equal_nan=True)   False
eq(MappingProxyType({"a": np.ones(2)}), MappingProxyType({"a": np.ones(2)}))     TypeError: `==` ... cannot be converted to a bool
eq({"a": np.ones(2)}.items(), {"a": np.ones(2)}.items())                        TypeError (same)
ac(array.array("d", [1.0]), array.array("d", [1.0 + 1e-9]))                     False
```

A `mappingproxy` holding arrays is how a class namespace or a read-only config is often exposed. The
documented caveat ("a class that compares arrays raises `TypeError`") is right, but it should not
apply to a type that is a mapping. **Target**: register `types.MappingProxyType` with the mapping
tester and `array.array` with the sequence tester (two lines, no new mechanism: changing the
registry to follow ABC registrations would also change `hashing` and `summary`, which share it).
`dict_items` and `dict_keys` stay on `==` (documented).

### G8. Small floats are computed in their own dtype **[verified]**

```
ac(np.array([60000], "f2"), np.array([-60000], "f2"), rtol=1.5, atol=0)     True + RuntimeWarning: overflow
ac(np.array([3e38], "f4"), np.array([-3e38], "f4"), rtol=1.5, atol=0)       True + RuntimeWarning: overflow
ac(3e38, -3e38, rtol=1.5, atol=0)                                            False   (Python float)
ac(np.array([1e308 + 1e308j]), np.array([-1e308 - 1e308j]), atol=1)          RuntimeWarning: overflow in subtract
```

`elementwise_close` computes `rtol * max(|a|, |b|)` and `|a - b|` in the dtype of the arrays; the
NumPy caller silences `invalid` only. At the edge of the range the tolerance or the difference
becomes `inf` and `inf <= inf` is `True`. It is an edge case, but it is a wrong answer plus a warning
in a library whose job is to be exact, and the fix is the same as G1 (one cast in the kernel).
**Target**: `float16`, `bfloat16`, `float32` are cast to `float64` (and `complex64` to `complex128`)
before the kernel; the NumPy caller silences `over` as well as `invalid`.

### F6. `show_difference` (still open) **[verified]**

Unchanged since the second pass: silent with the default logging configuration, on
`objects_are_equal` and `objects_are_allclose` only, absent from `assert_*`, and `compare` hard-codes
it to `True`. The messages are built before the logger can drop them; there are 50 occurrences of
`show_difference` in `src` and 39 `report_difference` calls, each behind an `if config.show_difference`.
The guides still present it as the way to find a difference ("Finding a Difference", the Design
Principles, the README). `compare` and `assert_*` already do the job, print by default, and are tested.
**Target** (section 5.4): remove it and the logger sink (decision D1).

### F7. The cost of the failure path (still open) **[verified]**

```
eq(a, b)         1M floats, differ at index 0     10.3 ms
compare(a, b)                                    230.7 ms     (safe_repr of both lists, then truncated to 1000 characters)
```

`ComparisonResult.actual` and `.expected` are `truncate_str(safe_repr(root), max_len=1000)`: the whole
`repr` is built and cut; the parent levels do the same with `safe_str(child)`. Note that these two
fields repeat the root objects the caller already holds; the useful value is the one **where the
difference is** (F14). **Target**: the reason holds a bounded representation of the two differing
values only (`reprlib`, array libraries' own summaries), and the two root fields are dropped
(decision D4).

### F13. Masked arrays (still open) **[verified]**

`eq(ma.array([1, 2], mask=[0, 1]), ma.array([1, 3], mask=[0, 1]))` is `False`. New evidence for the
decision: `hash_object` of the two arrays is **equal**, so hashing already treats the data under the
mask as not part of the value, and equality is the stricter one. **Target** (D3): compare the mask
and the unmasked values; keep `fill_value` compared.

### F14. A difference is not located (still open) **[verified]**

```
compare(np.array([1, 2]), np.array([1, 3])).path                     ()
compare(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 3]})).path   ()   message: the dump of both frames
compare(pa.table({"a": [1]}), pa.table({"a": [2]})).path             ()
compare({1, 2}, {1, 3}).path                                         (2,)   "objects are not equal at actual[2]": the element, shown as an index
```

A user who compares two frames of 200 columns has to find the column alone. All the information is
available: the kernel returns a boolean array, and the first `False` is the location, found only when
somebody asks for the explanation. **Target**: path `['a'][1]` for a frame (column, row position),
`[1][2]` for an array, `{2}` for a set element (a path part `Element`).

### G9. Documentation drift **[verified]**

`inv test.doctest` runs `python -m doctest` on each guide. `equality.md`, `equality_handler.md` and
`equality_tester.md` pass; these statements are wrong or stale:

- `equality.md` ("Behavior by Type", details of pandas): "with `equal_nan=False`, a data frame or a
  series that contains any missing value is never equal to another one, **even to a copy of itself**".
  F4 made the opposite true (the table above the paragraph says so).
- "Design Principles": `show_difference` is listed as a configuration parameter.
- `extending.md:139`: `from coola.equality import get_default_registry` raises `ImportError`
  (the function is in `coola.equality.tester`); `python -m doctest docs/docs/uguide/extending.md`
  fails with 1 failure. Check why the Markdown doctest job did not stop on it.
- `objects_are_equal` docstring: the **Note** is the one of `objects_are_allclose` ("even with a
  tolerance", "`atol=1`") in a function that has no tolerance.
- `equality_handler.md` (654 lines) documents the chain API that section 5 removes.

## 4. Architecture assessment

### A1. The handler chain **[verified, reading]**

A tester owns the head of a linked list of handlers; each `handle` ends with
`self._handle_next(...)`.

- **Frames**: 6.4 per level for a dict, 5.5 for a list, 4.5 for a dataclass (the largest nesting that
  fits in 1 000 frames is 141, 165 and 198). `max_depth=100` is reached before the interpreter limit
  only if the caller is less than about 250 frames deep: a dict of 99 levels passes from a 150-frame
  caller and fails with `RecursionError` from a 300-frame one (a test runner and a web framework use
  that much).
- **Calls**: a dict of 5 000 entries (`{float, [int, int]}`) makes 830 177 function calls (166 per
  entry) and creates 30 003 `contextmanager` objects (`check_recursion_depth`, `track_comparison`
  for each container); the top of the profile is `handle` and the context managers.
- **Machinery that is not comparison logic**: `handler/base.py` (375 lines: `chain`, `chain_all`,
  `set_next_handler`, `_verify_next_handler` with a cycle walk, `validate_chain`, `HandlerCycleError`
  with two base classes, `visualize_chain`), `handler/mixin.py` (123), `handlers_are_equal`,
  `equality_attrs`, the `EqualityHandler` protocol, `tester/chain.py`, `tester/handler.py`. A
  tester equality (`equal(other)`) is the equality of two chains.
- **Fragility**: a chain must end in a terminal handler (`TrueHandler`, `ObjectEqualHandler`) or
  raises `RuntimeError("next handler is not defined")` at comparison time; a terminal handler must
  not call the next one, a filter must, and nothing checks it. A container handler must call
  `check_recursion_depth` and `track_comparison` itself: a custom one that forgets either is a
  silent stack overflow or an infinite loop.
- **The abstraction is mostly a wrapper**: every library handler is a 15-line class that calls a
  function (`array_equal`, `frame_equal`, ...) and then `report_difference`; 25 of the 29 chain
  testers differ from each other by one handler.

### A2. Identity and type are tester business, but identical everywhere

29 of the 30 exported chain testers start with `SameObjectHandler(), SameTypeHandler()`. The registry
already picks the tester from `type(actual)`; if `type(expected)` differs the answer is `False` for
every tester. Moving both to the dispatcher makes the Strict type and Reflexive rows true **by
construction**: today a custom tester that forgets `SameTypeHandler` silently breaks them (and,
through the dispatch on `type(actual)`, the Symmetric row).

### A3. `EqualityConfig`: settings, state, two tolerance shapes

`EqualityConfig` holds the settings (`registry`, `equal_nan`, `atol`, `rtol`, `show_difference`,
`max_depth`, `reporter`) **and** the state of one run (`_current_depth`, `_warned`, `_in_progress`):
it is "not thread-safe" and `compare` mutates `config.reporter` after the construction. The registry
is passed twice (`run_comparison(registry, actual, expected, config)` uses the first for the root and
`config.registry` for every child). The entry points take `rtol`/`atol` (three of them) or a
`Tolerance` (`compare`), and the config stores two floats: three shapes for one concept.

### A4. Numbers follow one rule; structure and non-numeric values do not

After F2 and F3 the numeric part is unified, through the kernel and `array_equal` (the frame handlers
call the NumPy handler for their numeric columns). What still belongs to a library:

| Library | Structure                                              | Non-numeric values                       | Consequence    |
| ------- | ------------------------------------------------------ | ---------------------------------------- | -------------- |
| pandas  | `pd.testing.assert_*_equal` (dtype, index, flags, ...) | `pd.testing`: booleans, objects, nested  | G3, G4, G5, G6 |
| polars  | `polars.testing.assert_*_equal`                        | `polars.testing`: nested get a tolerance | G4             |
| pyarrow | `equals`, then per-column NumPy                        | `equals`: nested exact                   | G4             |

The pandas handler is 503 lines, most of them workarounds to feed `pd.testing` something it does not
change (copying frames, `isetitem` of the already-compared columns, `dt.total_seconds()` to turn
durations into floats, `_index_has_nan` for a `MultiIndex`). The polars one calls `lru_cache` for the
name of a keyword (`is_new_naming`) and `is_nan` on the nested columns by hand.

### A5. Public surface

`coola.equality.handler` exports 50 names, `coola.equality.tester` 36, `coola.equality` 9. The
library handlers (`PandasDataFrameEqualHandler`, `TorchTensorSameDeviceHandler`, ...) are documented
API, so a change in how a frame is compared is a breaking change for people who composed them;
`coola.io`, `coola.hashing.registry`, `coola.registry.base` and `coola.nested` import
`coola.equality.interface` (a private module path).

### T1. Tests

- `tests/properties` has equality properties for Python scalars, containers and dataclasses
  (`test_equality.py`, `test_equality_types.py`); the library types appear only in hashing and
  summary properties. The symmetry of arrays has 13 example-based tests; there is **no** property that
  compares the same random numbers in a list, a NumPy array, a float64 tensor, a pandas series, a
  polars series, a pyarrow array and an xarray array: G1, G3, G4 and G6 are all invisible to the
  suite. This property is the safety net of the migration of section 6, so it comes first.
- There is no test of a pandas or polars **failing** path at a realistic size (G5), and
  `tests/benchmarks/test_equality_benchmark.py` has five benchmarks, none for a failure or a library
  type.
- `tests/unit/equality` has 20.5k lines (the source has 8.6k): most of them test handlers one by
  one, so they follow the architecture of section 5 and have to be moved with it.

## 5. Target state

### 5.1 The design in one page

```
objects_are_equal / objects_are_allclose / compare / assert_objects_*        (5 functions, same keywords)
        |  build EqualityConfig (frozen) and ComparisonContext (one per call)
        v
ComparisonContext.compare(actual, expected, *path)          identity -> type -> depth -> cycle -> tester
        |  registry.find(type(actual))  (cached for the call)
        v
BaseEqualityTester.objects_are_equal(actual, expected, ctx)  plain Python: checks, ctx.compare(child), ctx.fail(reason)
        |  numbers and arrays: coola.equality._numeric.close(...)   one kernel for NumPy, torch, JAX
        v
library modules (_numpy, _torch, _jax, _pandas, _polars, _pyarrow, _xarray, _pydantic): private
```

Three rules keep it simple:

1. **One extension point**: the tester, a class with one method. No chain, no steps, no handler
   classes. Shared checks (same length, same shape, same dtype) are small functions.
2. **The dispatcher owns the invariants**: identity, type, depth, cycles, path. A tester cannot
   forget them.
3. **coola owns the semantics, the libraries own the structure**: every number goes through one
   kernel; a library says whether two frames have the same columns, never whether `1` is close to `2`.

### 5.2 The public API

```python
# the five entry points: same keywords, same meaning, same order
objects_are_equal(actual, expected, *, equal_nan=False, max_depth=100, registry=None) -> bool
objects_are_allclose(actual, expected, *, rtol=1e-5, atol=1e-8, equal_nan=False, max_depth=100, registry=None) -> bool
compare(actual, expected, *, rtol=0.0, atol=0.0, equal_nan=False, max_depth=100, registry=None) -> ComparisonResult
assert_objects_equal(actual, expected, *, equal_nan=False, max_depth=100, registry=None, root="actual") -> None
assert_objects_allclose(actual, expected, *, rtol=1e-5, atol=1e-8, equal_nan=False, max_depth=100, registry=None, root="actual") -> None
```

Removed: `show_difference`, `compare(tolerance=)` (the others never took it; a `Tolerance` stays as the
value stored in the config), `compare(max_repr_chars=)`. `compare` is exact unless `rtol`/`atol` are
given, as today.

```python
@dataclass(frozen=True, slots=True)
class EqualityConfig:  # shareable, hashable, thread-safe
    registry: EqualityTesterRegistry
    tolerance: Tolerance = Tolerance()
    equal_nan: bool = False
    max_depth: int = 100


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    equal: bool
    path: tuple[object, ...] = ()  # keys, positions, AttributeName, Element
    reason: str | None = None

    def format_message(self, root: str = "actual") -> str: ...
```

Exported from `coola.equality` (about 25 names instead of 95): the five functions,
`ComparisonResult`, `EqualityConfig`, `Tolerance`, `ComparisonContext`, `MaxDepthExceededError`,
`AttributeName`, `Element`, `BaseEqualityTester`, `EqualityTesterRegistry`, `get_default_registry`,
`register_equality_testers`, and the testers a user composes or registers for their own classes:
`ObjectAttributesEqualityTester`, `EqualEqualityTester`, `OrderedMappingEqualityTester`,
`SequenceEqualityTester`, `MappingEqualityTester`. Everything per library is private
(`coola.equality._numpy`, ...) and reached through the registry.

### 5.3 The tester and the context

```python
class BaseEqualityTester(ABC, Generic[T]):
    recursive: ClassVar[bool] = False  # True if objects_are_equal calls ctx.compare

    @abstractmethod
    def objects_are_equal(self, actual: T, expected: T, ctx: ComparisonContext) -> bool: ...


class ComparisonContext:
    config: EqualityConfig
    explaining: bool  # False for objects_are_*: nothing is formatted

    def compare(self, actual, expected, *path) -> bool: ...  # the one way to recurse
    def fail(self, reason: str | Callable[[], str], *path) -> Literal[False]: ...
    def warn_once(self, key: str, message: str) -> None: ...
```

`ctx.compare` does, in this order: `actual is expected` (equal); `type(actual) is not type(expected)`
(different, with the reason built by the context); the tester lookup (a plain dict filled from the
registry, so the 300 ns lock of `find_equality_tester` is paid once per type and call); for a
`recursive` tester the depth check (`MaxDepthExceededError`) and the cycle check (a pair already
being compared is equal); the call; the pop of the path. A tester that forgets `recursive = True`
still cannot overflow the stack, because the depth is counted there, but it will not detect cycles.

`ctx.fail(reason, *path)` records the **first** failure only: its reason and the path accumulated by
`ctx.compare` plus `path`; it returns `False`, so a tester writes `return ctx.fail(...)`. `reason`
may be a callable; it is called only if `ctx.explaining`.

The three testers that cover most of the package, in full:

```python
class SequenceEqualityTester(BaseEqualityTester[Sequence]):
    recursive = True

    def objects_are_equal(self, actual, expected, ctx):
        if len(actual) != len(expected):
            return ctx.fail(lambda: f"lengths differ: {len(actual)} vs {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            if not ctx.compare(a, e, i):
                return False
        return True


class MappingEqualityTester(BaseEqualityTester[Mapping]):
    recursive = True

    def objects_are_equal(self, actual, expected, ctx):
        keys_a = {comparison_key(k, ctx.config): k for k in actual}
        keys_e = {comparison_key(k, ctx.config): k for k in expected}
        if keys_a.keys() != keys_e.keys():
            return ctx.fail(lambda: describe_key_difference(keys_a, keys_e))
        for key, value in actual.items():
            if not ctx.compare(value, expected[keys_e[comparison_key(key, ctx.config)]], key):
                return False
        return True


class NumpyArrayEqualityTester(BaseEqualityTester[np.ndarray]):
    def objects_are_equal(self, actual, expected, ctx):
        if actual.dtype != expected.dtype:
            return ctx.fail(lambda: f"dtypes differ: {actual.dtype} vs {expected.dtype}")
        if actual.shape != expected.shape:
            return ctx.fail(lambda: f"shapes differ: {actual.shape} vs {expected.shape}")
        close = close_numpy(actual, expected, ctx)  # boolean array; `object` cells go through ctx.compare
        if close.all():
            return True
        index = np.unravel_index(np.argmin(close), close.shape)  # found only when somebody asks
        return ctx.fail(lambda: describe_values(actual[index], expected[index]), *index)
```

A user tester for their own class changes by one parameter type and one call
(`config.registry.objects_are_equal(a, b, config)` becomes `ctx.compare(a, b, "name")`):

```python
class PointEqualityTester(BaseEqualityTester[Point]):
    recursive = True

    def objects_are_equal(self, actual, expected, ctx):
        return ctx.compare(actual.coords, expected.coords, AttributeName("coords"))
```

The "wrapper that can be inspected and compared" role of the chain is a `repr` and an `__eq__` on the
base class (same type and same `vars`).

### 5.4 Reporting

- One record, the first failure: `(path, reason)`. No list of reports, no reversal in `compare`, no
  message per parent level (the path says where).
- Nothing is formatted unless `ctx.explaining` (`compare`, `assert_*`): `objects_are_equal` pays for
  no message, so the 5x of F6 and the 20x of F7 go away. The values in a reason go through one
  bounded formatter (`reprlib.Repr` with limits; NumPy, torch and pandas already summarise).
- `show_difference` and the logger sink are removed. A user who wants a log line writes
  `log.info(compare(a, b).format_message())` (decision D1).
- Path parts: a key or a position is itself (`['a']`, `[3]`); `AttributeName("x")` is `.x`;
  `Element(2)` is `{2}`. A frame reports `("a", 3)` (column, row position), an array `(1, 2)`.

```
objects are not equal at actual['prices'][3]                    # DataFrame, column "prices", row 3
values differ: 4.0 vs 5.0 (|diff| 1.0 > atol 1e-08 + rtol 1e-05 * 5.0)

objects are not equal at actual.weights[1][2]                   # dataclass field, array element
values differ: 0.5 vs 0.25

objects are not equal at actual{2}                              # set
element has no match in expected: 2

objects are not equal at actual['a']
types differ: int vs float
```

### 5.5 The numeric kernel

`handler/elementwise.py` becomes `coola/equality/_numeric.py` and keeps its formula, extended by the
casts of G1 and G8:

```
close = (a == b) | (finite & (|a - b| <= atol + rtol * max(|a|, |b|)))     | (isnan(a) & isnan(b) if equal_nan)
```

| Dtype                                | Preparation                                                             | Rule                                     |
| ------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------- |
| `bool`                               | none                                                                    | exact (`==`): no tolerance, no NaN       |
| signed and unsigned integers         | exact absolute difference as `uint64` (wrap-around trick); max in float | kernel                                   |
| `float16`, `bfloat16`, `float32`     | cast to `float64`                                                       | kernel                                   |
| `float64`, `complex128`              | none (`complex64` cast up)                                              | kernel (`abs` of the complex difference) |
| `timedelta64`                        | `int64` nanoseconds, `atol * 1e9`; `NaT` mask apart                     | kernel; `NaT` follows `equal_nan`        |
| `datetime64`, `str`, `bytes`, `void` | none                                                                    | exact; `NaT` follows `equal_nan`         |
| `object`                             | none                                                                    | cell by cell through `ctx.compare`       |

`xp` is `numpy`, `torch` or `jax.numpy`; the torch variant needs `torch.maximum`. The integer
difference is exact; the final comparison to a float tolerance is made in `float64`, so a difference
of exactly `atol` above 2^53 can still round the other way than the `Fraction` of
`int_isclose` (a measure-zero case, documented in the kernel docstring). Python scalars use the same
formula (`number_equal` already does), `Decimal`, `Fraction` and `timedelta` keep `difference_close`.

### 5.6 The semantics, as the Guarantees table should read

Unchanged rows: Strict type, Recursive, Order rules, Value semantics, Signed zero, Time zones,
Unregistered types, Reflexive, Symmetric, Not transitive, Pure, Consistent with hashing. Changed or
added:

| Row           | Statement                                                                                                                                                                                                                                                                                                   |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NaN           | Two NaN-like values (float or complex `NaN`, `Decimal("NaN")`, NumPy `NaT`) are equal only with `equal_nan=True` or if they are the same object, **in every container** (including a pandas `object` column and a nested cell). A null (`None`, `pd.NA`, a polars or Arrow null) equals a null, always.     |
| Tolerance     | `abs(a - b) <= atol + rtol * max(abs(a), abs(b))`, per element, for every number in every container (list, NumPy, torch, JAX, pandas, polars, pyarrow, xarray), computed exactly for integers. `bool`, enum members, instants, strings and the structure are exact. Nested cells follow the rule of a list. |
| Durations     | The tolerance applies to durations (`timedelta`, `timedelta64`, `pd.Timedelta`; `atol` in seconds). Instants are always exact.                                                                                                                                                                              |
| Failure modes | Cycle: equal. Nesting above `max_depth` (default 100, reached before the interpreter limit for a caller less than 600 frames deep): `MaxDepthExceededError`. `==` that raises `TypeError`/`ValueError` or returns a non-bool: `TypeError` that names the type. Any other exception of `==` propagates.      |
| Location      | `compare` and `assert_*` give the path of the first difference: keys, positions, attributes, the **column and row** of a frame, the **index** in an array, the **element** of a set.                                                                                                                        |

### 5.7 The libraries: structure by the library, values by coola

A frame is compared in two steps: the structure (cheap, by the library), then the columns, each by
its **kind**. A kind is decided by the dtype, once per column.

| Kind       | Dtypes                                                                | Comparison                                                                                     |
| ---------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `numeric`  | integers, floats, nullable `Int64`/`Float64`, Arrow/polars numerics   | null masks equal, then the kernel on the values (`fill 0` under the null)                      |
| `duration` | `timedelta64`, polars `Duration`, Arrow `duration`                    | kernel on integer nanoseconds                                                                  |
| `exact`    | booleans, strings, categories, instants, periods, intervals, decimals | library `equals` (a null equals a null); never sees the tolerance                              |
| `nested`   | polars `List`/`Array`/`Struct`, Arrow nested, pandas `object`         | `to_list()` and `ctx.compare` per cell; library `equals` as the fast path when there is no NaN |

| Library | Structure checked by the library                                                     | Not used any more                                      |
| ------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| pandas  | type, shape, `index.equals` (values, dtype, names), `columns.equals`, `dtypes` equal | `pd.testing`; the copy of frames; `dt.total_seconds()` |
| polars  | `schema` equal (names, order, dtypes), height; a `LazyFrame` is collected            | `polars.testing`, `is_new_naming`                      |
| pyarrow | `type`, `len`; for a table `schema.equals(check_metadata=False)`, `num_rows`         | `equals` followed by the NumPy fallback                |
| xarray  | dims, coords, attrs, name (through `ctx.compare` on the parts, as now)               |                                                        |

pandas `attrs`, `flags` and Arrow schema metadata stay ignored by design (documented). The dtype
includes the time resolution (`timedelta64[s]` against `timedelta64[ns]` differ, as in NumPy): a
pandas 3 user who builds the same instant two ways has to cast (decision D7). Every comparison is a
boolean array and `all()`, so the failing path costs what the passing one does (G5), and the first
`False` gives the column and the row (F14).

### 5.8 Evidence: a prototype of the dispatcher

`/private/tmp/claude-501/-Users-thibaut-workspace-code-2022-coola/4ef7429f-5a8e-4465-ab31-51d3c00554ec/scratchpad/proto.py`
(about 70 lines: `Ctx.compare`, `Ctx.fail`, a scalar tester, a sequence tester, a mapping tester, the
default tester; no library, no `Decimal`). It is not production code and the registry is a `dict`, but
it uses the same loop as section 5.3 with the identity check, the type check, the depth counter and
the cycle set. Best of 7 runs, equal but distinct objects (a shared object would hit the identity
shortcut):

| Case                                | `main`  | Prototype, dict lookup | Prototype, real registry lookup per element | Prototype, lookup cached per call |
| ----------------------------------- | ------- | ---------------------- | ------------------------------------------- | --------------------------------- |
| 20 000 floats, exact                | 15.5 ms | 3.4 ms                 | 9.4 ms                                      | 3.3 ms                            |
| 20 000 floats, `rtol`/`atol`        | 31.2 ms | 5.3 ms                 | n/a                                         | n/a                               |
| dict of 5 000 `{float, [int, int]}` | 55.9 ms | 7.8 ms                 | 15.6 ms                                     | 7.5 ms                            |
| same dict, `explaining=True`        | n/a     | 8.4 ms                 | n/a                                         | n/a                               |
| frames per nesting level (dict)     | 6.4     | 1.8                    | 1.8                                         | 1.8                               |
| nesting that fits in 1 000 frames   | 141     | 497                    | 497                                         | 497                               |

The real registry lookup costs about 300 ns (a lock and the cache); that is why the context caches
the lookup per call. Explaining costs 7% (a path stack). The default `max_depth=100` is then reached
before the interpreter limit for a caller up to about 600 frames deep, instead of 250. The gain is
the removal of the generators of `contextmanager`, the `_handle_next` property and the repeated
`SameObject`/`SameType` handlers, not a micro-optimisation.

### 5.9 What goes, what stays

| Removed                                                                                                                                        | Replaced by                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `BaseEqualityHandler`, `HandlerEqualityMixin`, `EqualityHandler`, `HandlerCycleError`, `create_chain`, `handlers_are_equal`, `visualize_chain` | the tester method                                      |
| 38 handler classes (`Same*Handler`, `*EqualHandler`, `TrueHandler`, `FalseHandler`, ...)                                                       | statements in 20 testers and 8 shared helper functions |
| `BaseChainEqualityTester`, `HandlerEqualityTester`, `tester.equal(other)`                                                                      | `BaseEqualityTester` with `__eq__`                     |
| `check_recursion_depth`, `track_comparison`, `report_difference`, `EqualityConfig` state and `reporter`                                        | `ComparisonContext.compare`, `.fail`                   |
| `show_difference`, the logger sink, `compare(tolerance=)`, `max_repr_chars`, `ComparisonResult.actual/expected`                                | `compare` and `assert_*`                               |
| `pd.testing`, `polars.testing` wrappers, `is_new_naming`                                                                                       | column kinds (5.7)                                     |

Kept as they are: the registry and its lazy per-library registration, `Tolerance`, `comparison_key`
(the set and the mapping testers), `get_object_attributes`, `difference_close`, `number_equal` and
`int_isclose` (moved to `_numeric`), `MaxDepthExceededError`, `AttributeName`.

An estimate, not a measurement: the package goes from 8.6k to about 4k lines, `tests/unit/equality`
from 20.5k to about 12k (the handler tests become tester tests), and `equality_handler.md` (654
lines) disappears.

## 6. Migration plan

One PR per row, as `CLAUDE.md` asks. "B" marks a breaking change. The plan keeps `main` green:
rows 1 to 8 do not touch the handler API, rows 9 to 14 move the architecture one family at a time.

| Step | Content                                                                                                                                                                                                                                                                                                                                                      | Closes         | B   | Depends on |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- | --- | ---------- |
| 1    | **Cross-container property tests**: the same random `list[float]`/`list[int]` in list, NumPy, torch (float64, int64), pandas, polars, pyarrow, xarray: same answer for random `rtol`, `atol`, `equal_nan`; copy is equal; swapped is symmetric. `xfail` for G1, G3, G4, G6 until fixed                                                                       | T1             |     |            |
| 2    | Enum members exact (`isinstance(actual, (bool, Enum))` in the number tester); pandas booleans exact (`Series`, `DataFrame`, `Index`, extension arrays)                                                                                                                                                                                                       | G2, G3         | B   | 1          |
| 3    | Kernel: integers exact above 2^53, small floats cast to `float64`, `over` silenced; applies to NumPy, torch, JAX and to the frame numeric columns                                                                                                                                                                                                            | G1, G8         |     | 1          |
| 4    | Register `types.MappingProxyType` and `array.array`; regression tests with arrays inside a `mappingproxy`                                                                                                                                                                                                                                                    | G7             |     |            |
| 5    | Guides: pandas paragraph, Design Principles, `extending.md` import, docstring Note of `objects_are_equal`; find why the Markdown doctest job did not fail on `extending.md`                                                                                                                                                                                  | G9             |     |            |
| 6    | Masked arrays: compare the mask and the unmasked values, keep `fill_value`                                                                                                                                                                                                                                                                                   | F13            | B   |            |
| 7    | pandas on column kinds: no `pd.testing`, `object` and nested cells through the registry, failing-path benchmark (1M floats and a 200k x 10 frame, must be within 2x of the passing path)                                                                                                                                                                     | G4, G5, G6     | B   | 1, 3       |
| 8    | polars and pyarrow on column kinds: nested cells through the registry; Arrow tables and batches                                                                                                                                                                                                                                                              | G4             | B   | 1, 3       |
| 9    | `ComparisonContext` **inside** `EqualityConfig` (methods `compare`, `fail`, `explaining`, path stack); the dispatcher owns identity, type, depth, cycle, a per-call lookup cache; handlers keep working because they receive the same object                                                                                                                 | A2, part of A3 |     | 1          |
| 10   | One reporting mechanism: first failure, lazy bounded reasons; `report_difference` forwards to `ctx.fail`; remove `show_difference`, the logger sink, `compare(tolerance=, max_repr_chars=)` and the two root fields                                                                                                                                          | F6, F7, A3     | B   | 9          |
| 11   | Port the Python testers (object, scalar, sequence, mapping, set, dataclass, object attributes, pydantic) from chains to methods; a port reports a location; delete the handlers no one uses                                                                                                                                                                  | A1             | B   | 10         |
| 12   | Port the array testers (NumPy, masked, torch, JAX, xarray): index in the path, bounded value reasons                                                                                                                                                                                                                                                         | A1, F14        | B   | 11         |
| 13   | Port the frame testers (pandas, polars, pyarrow): column and row in the path; `Element` path part for sets                                                                                                                                                                                                                                                   | A1, F14        | B   | 11, 7, 8   |
| 14   | Delete the chain machinery (`BaseEqualityHandler`, mixin, `create_chain`, `BaseChainEqualityTester`, `HandlerEqualityTester`); freeze `EqualityConfig`; library code to private modules; trim `__all__`; `coola.io` and others import from `coola.equality`; migration note in `docs/docs/migration`; rewrite `equality_handler.md` and `equality_tester.md` | A1, A3, A5     | B   | 12, 13     |

Steps 1 to 6 are small and independent of the architecture: do them first, they fix wrong answers.
Steps 11 to 14 break every custom handler and every custom tester (`handle(actual, expected,
config)` and `objects_are_equal(actual, expected, config)` become `objects_are_equal(actual,
expected, ctx)`), so 11 to 14 share one release with the note showing the three-line change of a
tester.

Benchmark gates to add to `tests/benchmarks/test_equality_benchmark.py` **before step 9**, from the
numbers measured here (fail on a regression above 10%): 20 000 floats in a list 15.5 ms; dict of 5 000
x `{float, [int, int]}` 55.9 ms; `compare` of 1M floats differing at index 0 230 ms (target below
30 ms after step 10); `objects_are_equal` of two pandas series of 2M floats differing at index 0
5 898 ms (target below 10 ms after step 7); a 200k x 10 frame differing in one cell 538 ms (target
below 10 ms). Add the depth test: 100 nested dicts, lists, dataclasses and pydantic models from a
caller 200 frames deep, under the default limit (after step 11).

## 7. Decisions for the maintainers

| ID  | Question                                                                                    | Recommendation                                                                                                                                                                               |
| --- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Remove `show_difference` and the logger sink (F6)?                                          | Yes. `compare` and `assert_*` cover it and print by default. Update README and the three guides                                                                                              |
| D2  | A `Step` pipeline (second pass) or plain tester methods?                                    | **Plain methods.** The pipeline keeps 40 step classes; a method is 10 lines of straight code, reads without the chain, reuses checks by calling functions, and the prototype shows the speed |
| D3  | Masked arrays: ignore the data under the mask (F13)?                                        | Yes: the hasher already does; keep `fill_value` compared                                                                                                                                     |
| D4  | Drop `ComparisonResult.actual/expected` and `max_repr_chars`?                               | Yes: they repeat the roots; the reason holds the two differing values, bounded                                                                                                               |
| D5  | `compare(rtol, atol)` instead of `compare(tolerance=Tolerance)`?                            | Yes: one signature for five functions; `Tolerance` stays as the config value                                                                                                                 |
| D6  | Nested and `object` cells of frames through the registry (G4, G6)?                          | Yes: one rule per value; the exact fast path keeps the cost when nothing needs the registry                                                                                                  |
| D7  | Keep the time resolution in the dtype (`[s]` against `[ns]` differ)?                        | Yes (as NumPy); document it in the pandas section, with the `astype` fix                                                                                                                     |
| D8  | Cast `float16`/`float32` arrays to `float64` in the kernel (memory x2 for the temporaries)? | Yes: correctness first; chunk the temporaries if a benchmark shows a problem                                                                                                                 |
| D9  | Is one release that breaks the custom tester API acceptable (steps 11 to 14)?               | Yes, with the migration note; the change of a user tester is one parameter and one call                                                                                                      |
| D10 | `requires_grad`, pandas `attrs`/`flags`, Arrow metadata stay ignored?                       | Yes: they are not values (documented for the last two, add `requires_grad` to the torch section)                                                                                             |

## 8. Not covered

- Per-library corner cases beyond the probes: sparse and quantized torch tensors (compared, semantics
  not checked), nested tensors (raise a `RuntimeError` from torch), dask-backed xarray, pandas
  `Interval`, `Period` and `Sparse` columns on the `exact` path (they are covered by `Series.equals`,
  which was not probed per dtype).
- The GPU path (`device` is compared; the kernel on CUDA was not run), Windows, fork behaviour.
- The thread safety of a **shared** registry while a type is registered during a comparison: the
  per-call lookup cache of step 9 snapshots the answer, so a registration made during a call is seen
  by the next one only.
- The prototype of section 5.8 has no `Decimal`, no cache invalidation and no library tester: its
  numbers bound the gain of the dispatcher, not the end-to-end gain of the rewrite. The 20-tester
  and 4k-line estimates of section 5.9 are an extrapolation from it.
- Mutation testing of the equality package is still open (`suggestions.md`).
