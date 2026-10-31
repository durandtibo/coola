# Codebase review: suggestions (round 6)

This file **replaces the previous `review/suggestions.md`** (round 5, items 22.1 to 30.1). That version
is in git history (last touched by `7530c55a`).
Section 41 gives the status of its items. Nothing below repeats an item of round 5 or of the
`review/<package>.md` design documents, except where an item is explicitly marked _carried over_.

The numbering **continues** the previous rounds (sections 34 to 42), so that an ID such as `34.1` in a
branch name or a commit message is unambiguous.

## How this review was done

Checkout of `main` at `7cd495e8` (`.venv`, Python 3.14.8, NumPy 2.x, pandas 3.x, torch, polars,
pyarrow, xarray, JAX, pydantic, scipy installed). Baseline, run on the clean checkout:

- `pytest tests/unit tests/integration`: 13411 passed, 34 skipped, 25 xfailed, **1 failed** (39.1
  below), in 35 s, with 100% line and branch coverage (6647 statements, 1560 branches).
- `pytest tests/properties`: all pass (with the markers of 39.2 hiding a defect and a wrong test).
- `ruff check src tests`, `ty check` and `lint-imports` (7 contracts) are clean. `uv build` builds the
  sdist and the wheel.
- The Markdown of `README.md`, `docs/docs/index.md`, `get_started.md` and `uguide/*.md` passes
  `python -m doctest` (315 examples in `uguide/equality.md` alone), so the user documentation is not
  the problem; the findings on the documentation are about `docs/docs/dev`.

What was done:

1. **Read-through.** Every package of `src/coola` (231 modules, 31.7k lines) was surveyed. The
   packages with the most logic (`registry`, `equality` engine, helpers and the NumPy, pandas, torch,
   xarray, pydantic, table and collection testers, `jsonable`, `io`, `identifier`, `random`,
   `validation`, `factory`, `utils`, `formatting`, `display`) were read with the docstrings stripped.
   The hashing, summary, recursive, iterator, nested, reducer and rounding engines were exercised
   rather than read line by line.
2. **Probes.** Several hundred one-line probes (equality of 120 pairs of values, hashing of 60 types,
   `to_jsonable` of 50 values, summaries, rounding, reducers, pandas, polars and pyarrow edge cases,
   a differential fuzz of the three reducers, timing of the traversal helpers on shared structures).
   Items marked **[verified]** were reproduced on this checkout and the repro is given. Items marked
   **[reading]** come from reading the code and were not reproduced (GPU, Windows and CI-only
   behaviour were not run; the CI results were not looked at).
3. **Conventions.** The rules of `CLAUDE.md` that can be checked mechanically (one assertion per test,
   section headers, tests for every function, docstrings) were checked with AST scripts over `src` and
   `tests` (section 39).
4. **Round 5 follow-up.** Every round-5 item was re-checked by running its repro (section 41).

The library is in a very good state: all the round-5 crashes are fixed, and the findings below are
smaller. Four themes stand out: **hashing and equality still disagree** on a few types (section 34),
the **diagnostics of the table testers** are far less useful than those of the array testers
(section 35), **the validators are written for Python `int` and not for NumPy integers** (36.1), and
**the shared sub-objects of a structure cost an exponential time** in every traversal helper but one
(38.1).

Priority: **P1** wrong result, crash on realistic input or a red test on `main`, **P2** inconsistent or
surprising behaviour, **P3** cleanup or design.

## Summary

| ID   | P   | Area        | Finding                                                                                                 |
| ---- | --- | ----------- | ------------------------------------------------------------------------------------------------------- |
| 39.1 | P1  | tests       | `test_real_mark_is_importable_in_a_generated_module` fails on `main`: `docformatter` rewrote its source |
| 34.1 | P2  | hashing     | Equal data frames with a different order of the categories hash differently                             |
| 34.2 | P2  | hashing     | `hash_object(np.timedelta64)` leaks a raw `TypeError`; `np.datetime64` and other types have no hasher   |
| 35.1 | P2  | equality    | A data frame or series difference is reported without cause, row or value                               |
| 35.2 | P2  | equality    | `objects_are_allclose` ignores the tolerance in list/struct columns and in `object` columns             |
| 36.1 | P2  | validation  | `manual_seed(np.int64(3))` and 5 other entry points reject NumPy integers                               |
| 38.1 | P2  | performance | Shared sub-objects are visited once per reference: 2^n time for equality, `recursive_apply`, ...        |
| 39.2 | P2  | tests       | Non-strict `xfail` markers hide a real defect (34.1) and a wrong test (_carried over_ 29.1)             |
| 34.3 | P3  | hashing     | The list of types that are not hashed is stale and omits what changed recently                          |
| 35.3 | P3  | equality    | `compare` and `assert_objects_equal` do not follow the signature of the `objects_are_*` functions       |
| 36.2 | P3  | api         | `dir(coola.<package>)` lists `TYPE_CHECKING` and `lazy_exports`; `coola.formatting` exports nothing     |
| 36.3 | P3  | factory     | `import_object` cannot reach `module.Class.attribute`, `pkgutil.resolve_name` can                       |
| 36.4 | P3  | utils       | 12 modules of the public `coola.utils` are used by nothing in the library                               |
| 37.1 | P3  | jsonable    | Non-finite floats, `longdouble`, `IntEnum` keys, data frames and sparse tensors                         |
| 37.2 | P3  | nested      | `from_flat_dict(to_flat_dict(x))` is lossy for keys that contain the separator (default)                |
| 37.3 | P3  | reducer     | `NativeReducer.mean` raises on `[inf, -inf]` where NumPy and torch return `nan`                         |
| 37.4 | P3  | random      | `random_seed` leaves the CUDA/MPS generators seeded when they are initialised inside the block          |
| 37.5 | P3  | formatting  | Byte sizes stop at TB, `truncate_str` raises for a short limit; summary of sparse/nested tensors        |
| 38.2 | P3  | performance | A list of floats takes 80x the time of `==` (no fast path for floats)                                   |
| 39.3 | P3  | tests       | Convention slips: 103 multi-assert tests, 82 malformed headers, 29 tests with logic (_carried over_)    |
| 39.4 | P3  | tests       | 14 modules without a mirrored test file; about 160 functions never named in a test                      |
| 39.5 | P3  | maintenance | 56 `# pragma: no cover` import guards and 12 copies of the `is_*_available`/`check_*` pair              |
| 40.1 | P3  | docs        | `docs/docs/dev` describes the deleted `DataclassMarker`; `dev` and `review/` duplicate about half       |
| 40.2 | P3  | packaging   | Classifiers, `CITATION.cff`, `all` extra (_carried over_ 30.1)                                          |

## 34. `hashing` and `equality`: the contract "equal objects hash equal"

### 34.1 Equal data frames with a different order of the categories hash differently **[verified]** (P2)

```python
import pandas as pd
from coola.equality import objects_are_equal
from coola.hashing import hash_object

a = pd.DataFrame({"c": pd.Categorical(["a", "b"], categories=["a", "b"])})
b = pd.DataFrame({"c": pd.Categorical(["a", "b"], categories=["b", "a"])})
objects_are_equal(a, b)  # True
hash_object(a) == hash_object(b)  # False
a.c.cat.codes.tolist(), b.c.cat.codes.tolist()  # ([0, 1], [1, 0])
```

With `ordered=True` the two are different for both (the contract holds), and with an extra unused
category they are different for both. Only the order of the categories of an **unordered** categorical
is invisible to equality and visible to the hash. This is the defect that the property test
`test_different_frames_have_different_hashes` is marked `xfail(reason="pandas metadata (14.2)")` for:
run it with `--runxfail` and Hypothesis finds `(None, None, ['a', 'b'], False)` against
`(None, None, ['b', 'a'], False)`, and it fails on `assert not objects_are_equal(...)`, that is on the
equality side, not on the hash. The marker hides it (39.2).

`pandas.testing.assert_frame_equal` compares the categories in order by default
(`check_category_order=True`), so the simplest fix is in the equality: `_same_dtypes`
(`equality/tester/pandas.py:326`) compares `list(df.dtypes)`, and `CategoricalDtype(["a", "b"]) ==
CategoricalDtype(["b", "a"])` is `True` for unordered dtypes. Compare `list(dtype.categories)` for the
category dtypes (also inside a `MultiIndex`), keep the hash as is, and remove the marker. The alternative (sort the categories in the hash) changes every categorical hash and is not recommended. The polars
`Categorical` and the pyarrow dictionary arrays should be checked for the same question (not tested).

### 34.2 `hash_object(np.timedelta64)` raises a `TypeError` that names no type **[verified]** (P2)

```python
import numpy as np
from coola.hashing import hash_object

hash_object(np.timedelta64(1, "s"))
# TypeError: int() argument must be a string, a bytes-like object or a real number, not 'datetime.timedelta'
hash_object(np.datetime64("2020-01-01"))  # TypeNotRegisteredError
```

`np.timedelta64` is a subclass of `np.signedinteger` (`[c.__name__ for c in np.timedelta64.__mro__]` is
`['timedelta64', 'signedinteger', 'integer', ...]`), so the registry resolves it to the `ReprHasher`, and
`_normalize_numpy_scalar` (`hashing/scalar.py:162`, `isinstance(data, np.integer)`) calls `int(value)`.
The user guide says that the scalar is _not hashed yet_ (so a
`TypeNotRegisteredError` is expected); the actual error is a bare `TypeError` that does not name the
type.

Fix: write the two hashers (`datetime64` and `timedelta64`: the unit, the int64 value, one `NaT`),
register them in `hashing/numpy.py` (the type is in the MRO before `np.integer` if it is registered by
its own class), and make the integer branch of the normalization test `dtype.kind in "iu"`, not
`isinstance(data, np.integer)`. The same `isinstance(..., np.integer)` idiom should be searched for in
the other packages (`grep -rn "np.integer\|np.signedinteger" src`).

### 34.3 The list of unhashable types is stale **[verified]** (P3)

`docs/docs/uguide/hashing.md:229` lists the types that `objects_are_equal` supports and `hash_object`
does not. Running `hash_object` on one object of each type gives, in addition to that list:
`scipy.sparse` matrices and arrays (the equality tester was added in #2642), `pd.RangeIndex`,
`pd.MultiIndex` and `pd.array(..., dtype="Int64")` (the pandas extension arrays), `JsonLoader` and the
other `BaseLoader`/`BaseSaver` (they have an equality tester, `EqualEqualityTester`), `np.void`, `type`
objects (`hash_object(int)`), `np.dtype` and `torch.dtype`. `tests/integration/test_support_matrix.py`
has a `HASH_GAPS` set that is the right place to keep the list: it is `xfail(strict=True)`, so it cannot
go stale, but it has no `scipy` row, and the guide is not generated from it.

Suggestions: (a) add the scipy sparse row to the matrix; (b) generate the sentence of the guide from
`HASH_GAPS` (a test that compares the two is enough); (c) close the cheap gaps in one batch, since a
hash format change is needed for each anyway: the NumPy time scalars (34.2), `np.dtype`, `torch.dtype`
and `torch.device` (a string each), `array.array` (typecode and bytes), the dict views (as the list
of the elements), `type` (the qualified name), the pandas `Index`, `RangeIndex`, `MultiIndex` and
`Categorical` (the index hashing already exists inside the frame hasher, and only needs a registration
of its own), and the scipy sparse matrices (format, shape, `data`, `indices`, `indptr`). The
`LazyFrame` should stay out (hashing it would run the query).

## 35. `equality`: diagnostics and tolerance of the table testers

### 35.1 A data frame difference is reported without cause, row or value **[verified]** (P2)

The README promises that `compare` and `assert_objects_equal` "tell where two objects differ", and the
array testers do (`actual['k'][0]`, `values are different: 1.0 vs 0.0`). The `TableEqualityTester`
reports:

```python
a = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
compare(a, a.rename(columns={"b": "c"})).format_message()  # 'tables have a different structure'
compare(a, a.astype({"a": "float64"})).format_message()  # 'tables have a different structure'
compare(a, a.set_axis([5, 6])).format_message()  # 'tables have a different structure'
compare(a, a.iloc[:1]).format_message()  # 'tables have a different structure'
compare(
    a, a.assign(b=[1.0, 3.0])
).format_message()  # "tables have different values: column 'b' differs"
compare(
    pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [1.0]})
).format_message()  # 'tables have a different structure'
```

The same four causes give the same sentence, and a value difference names the column but not the row
nor the two values. For the tests of a data pipeline (the main use of `assert_objects_equal` on data
frames) the message is the first thing the user reads.

Fix, in `equality/tester/table.py`: `same_structure(other, ctx)` returns `bool`; let it return `None`
for "same" or a reason string (or call `ctx.fail(reason)` itself, as the array testers do) so that
each adapter says which part differs: shape (`(2, 2)` vs `(1, 2)`), columns (`['a', 'b']` vs
`['a', 'c']`, the first difference), dtypes (`a: int64` vs `a: float64`), index (the first differing
label, or the name). For the values, `column_close` already has the two arrays (`array_equal` for the
numeric kinds): compute the first mismatch with the existing `array_close` and report the row label
and the two values; for the `exact` kind, report the first position where `~(s1 == s2)` (with the
same null rule) is `True`, bounded as the NumPy reason is (`test_numpy_array_compare_reason_is_bounded`
is the model). Add the row label to the path (`actual['b'][1]`) so that `format_path` is useful.

### 35.2 `objects_are_allclose` ignores the tolerance in nested and object columns **[verified]** (P2)

```python
S = pd.Series
close(S([1.0]), S([1.000001]))  # True
close(pl.Series([[1.0]]), pl.Series([[1.0000001]]))  # False  (list column)
close(pl.Series([{"a": 1.0}]), pl.Series([{"a": 1.0000001}]))  # False  (struct column)
close(pa.array([[1.0]]), pa.array([[1.0000001]]))  # False  (list array)
close(pa.array([{"a": 1.0}]), pa.array([{"a": 1.0000001}]))  # False  (struct array)
close(S([1.0], dtype=object), S([1.000001], dtype=object))  # False  (pandas object column)
```

`_column` in the pandas, polars and pyarrow adapters classifies the data types into `numeric`,
`duration` and `exact`; a list, a struct, an `object` column holding floats, and a decimal fall in
`exact`, which "never sees the tolerance". The docstring of `_column` says so, but the user guide and
the docstring of `objects_are_allclose` do not, and the result (`False` for a difference of `1e-7`
with the default `rtol=1e-5`) is the opposite of what the function name promises. A model whose
predictions are a struct or list column cannot be compared with a tolerance.

Options: (a) document the rule in `uguide/equality.md` (a table of the dtypes that follow the
tolerance), which is the minimum; (b) for `list`/`struct`/`object` columns, fall back to the
engine: yield the values as children (`ctx.run` already compares nested Python values with the
tolerance) when the tolerance is not zero. (b) costs a conversion to Python objects only on the
non-exact path.

### 35.3 `compare` and `assert_objects_equal` do not follow the `objects_are_*` signatures **[verified]** (P3)

`compare(actual, expected, *, rtol=0.0, atol=0.0, ...)` takes a tolerance and defaults to an exact
comparison; `assert_objects_allclose` defaults to `rtol=1e-5, atol=1e-8` (like `objects_are_allclose`);
`assert_objects_equal` has no tolerance. So `compare(a, b)` is the diagnostic twin of
`objects_are_equal` and not of `objects_are_allclose`, which is surprising to anybody who reads the
README ("`compare` ... exact or with a tolerance"). Suggestion: keep `compare` exact, and add the
sibling `compare_close` (or document the default prominently in the first line of the docstring);
this is the moment to do it because `compare` is new in the 2.0 line.

## 36. API, validation and `utils`

### 36.1 The validators reject NumPy integers **[verified]** (P2)

```python
manual_seed(np.int64(3))  # TypeError: seed must be an int, got int64
random_seed(np.int32(3))  # TypeError: seed must be an int, got int32
objects_are_equal(1, 1, max_depth=np.int64(5))  # TypeError: max_depth must be an int, got int64
summarize([1, 2, 3], max_items=np.int64(2))  # TypeError: max_items must be an int, got int64
SnowflakeIdGenerator(worker_id=np.int64(1))  # TypeError: worker_id must be an int, got int64
generate_ulid(timestamp_ms=np.int64(1700000000000))  # TypeError: timestamp_ms must be an int, ...
round_values(1.234, decimals=np.int64(1))  # works (inconsistent with the others)
```

`validate_int` (`validation/type.py`) is `isinstance(value, int) and not isinstance(value, bool)`. A
library whose reason d'être is NumPy and torch data meets NumPy integers all the time: the seed comes
from `rng.integers(...)` or from a dataframe row, a timestamp from `df["t"].iloc[0]`. The error message
("must be an int, got int64") is also hard to understand.

Fix: accept `numbers.Integral` (NumPy registers its integer types, and `bool` is excluded explicitly
as now) and have the callers store `operator.index(value)` so that the rest of the code sees a Python
`int` (for the `Tolerance` and `EqualityConfig` dataclasses, `__post_init__` can use
`object.__setattr__` on the frozen instance, or the interface functions can convert before building
them). Add a parametrized regression test over the 6 entry points above. `validate_number` already
accepts `np.float32` (it is a `numbers.Real`), so only the integer validator needs the change.

### 36.2 `dir()` of the packages leaks names; `coola.formatting` exports nothing **[verified]** (P3)

`lazy_exports` returns `dir_` as `sorted({*namespace, *exports})`. For the 10 packages that use it
(`io`, `reducer`, `hashing`, `equality`, `equality.tester`, `display`, `display.rich`, `random`,
`nested`, `summary`) `dir(coola.nested)` therefore contains `TYPE_CHECKING` and `lazy_exports` (and
`annotations`), next to the public names. Tab completion and the `help()` index show them. Return
`sorted({*namespace.get("__all__", ()), *exports})` (`namespace` is the `globals()` of the package, and the packages define `__all__`).

`coola.formatting.__init__` re-exports nothing (`dir(coola.formatting)` is `['annotations']`), unlike
every other package: the functions are reached through `coola.formatting.quantity`, `.text`,
`.secret`, `.diff` and `.structure`. Either export the public names, or write in the module docstring
that the submodules are the API, as `coola/__init__.py` does.

### 36.3 `import_object` cannot reach an attribute of a class **[verified]** (P3)

```python
import_object("os.path.join")  # ok
import_object("collections.OrderedDict.fromkeys")  # ImportError: The target object does not exist
pkgutil.resolve_name("collections.OrderedDict.fromkeys")  # <built-in method fromkeys ...>
```

`import_object` imports `rpartition(".")[0]` as a module. The `_init_` option of `instantiate_object`
is the supported way to call an alternative constructor, so this has a workaround, but a `_target_`
such as `torch.optim.lr_scheduler.LRScheduler.step` or a nested class or a class constant
(`enum` members) fails. The registry already uses `pkgutil.resolve_name` for the lazy groups; the same
function (with the `allowed_prefixes` check done on the string first, as now) gives the same
behaviour and removes the hand-written error handling.

### 36.4 `coola.utils` contains 12 modules and a function that the library does not use **[verified]** (P3)

By a search of `src` outside `utils`, nothing imports (directly or by name) `password`, `git`,
`env_vars`, `timing`, `date`, `file_size`, `batching`, `array`, `logging` (`CaptureHandler`), `lru`
(`LRUCache`), `bloom_filter`, `recursion` or `path.find_root_package_parent`. They are documented in
`uguide/utils.md` (348 lines) and tested, so they are maintained: each of them has to follow the
conventions of the repository (a docstring with a doctest, a test per function, keyword-only options),
and each is a public API that has to be kept in a 2.x line. `get_password` (an interactive prompt),
`get_today_date`, `get_current_branch`, `temp_env_vars` and `timeblock` have nothing to do with
comparing, hashing or summarizing.

`docs/docs/dev/utils.md` (decision D2) keeps `password` and `recursion` "for now and re-examine them later".
The alpha (`2.0.0a1`) is the moment to re-examine: either delete them, or move them to a separate
distribution (`coola-utils`) and keep `coola.utils` for what the other packages import (`fork`,
`singleton`, `imports`, `introspection`, `path.add_uuid_suffix`/`sanitize_path`, `safe`, `integer`,
`selection`, `stats`, `mapping`, `version`, `tensor`). Without the unused modules `coola.utils` shrinks from 23 modules (and the `imports` and `fallback` sub-packages) to 11.

## 37. Smaller findings by package

### 37.1 `jsonable` **[verified]** (P3)

```python
json.dumps(
    to_jsonable({"a": float("nan")}), allow_nan=False
)  # ValueError: Out of range float values ...
json.dumps(to_jsonable(np.array([np.inf])), allow_nan=False)  # ValueError
to_jsonable(np.longdouble(1.5))
# TypeError: The converter of longdouble returned a value of the same type: it would be converted without end
to_jsonable(np.array([1.5], dtype=np.longdouble))  # the same
to_jsonable(
    IntEnum_member
)  # the member itself (it is an int), and a StrEnum member is kept as a dict key
to_jsonable(
    pd.DataFrame({"a": [1]})
)  # TypeError: Cannot convert an object of type DataFrame to JSON
to_jsonable(
    torch.tensor([[1.0, 0]]).to_sparse()
)  # RuntimeError: Cannot access data pointer of Tensor ...
```

1. **Non-finite floats** are returned as `nan`/`inf`, which `json.dumps` writes as `NaN`/`Infinity`,
   invalid in strict JSON (`JSON.parse` of a browser, PostgreSQL `jsonb`). The name is `to_jsonable`.
   Add `nan="keep" | "null" | "raise" | "string"` (default `"keep"` to keep the current behaviour, or
   `"null"` as the safe default in a 2.0) and apply it in the float leaf and in the NumPy/torch/pandas
   fast path (`tolist()` followed by a check is the cost; `np.isfinite(array).all()` on a float array
   is cheaper than the `tolist()`).
2. **`np.longdouble`**: `scalar.item()` returns the same type, which the `ConvertTransformer` guard
   reports as an infinite loop. Convert with `float(scalar)` (documented as lossy) or `str(scalar)`.
   `dtype.char in "efd"` in `NDArrayTransformer` excludes `g`, which sends the array through the guard.
3. **`IntEnum` / `StrEnum`**: the MRO finds `int`/`str` before `Enum`, so the member is returned
   unchanged (a member that is an `int`), while a plain `Enum` gives its `value`. A dict key
   `{SE.A: 1}` stays an enum member. `json.dumps` handles both, but `yaml.safe_dump` and strict encoders do not. The registry resolves the MRO of the type, and `int`/`str` come before `Enum`: register a transformer for `IntEnum` and `StrEnum` explicitly (the `value`).
4. **`pandas.DataFrame`** has no transformer while `Series` and `Index` do. The obvious conversion is
   `to_dict(orient="records")` or `"list"`; make it an option or document that it must be registered.
5. **Sparse and nested tensors** raise a raw `RuntimeError`; check `tensor.layout` and raise the same
   `TypeError` as the other unsupported objects (or use `to_dense()` for the sparse COO/CSR).
6. `JsonSaver` (`io/json.py`) calls `json.dump` and so cannot save the objects that `to_jsonable` can.
   Since both exist in the package, an option `JsonSaver(convert=True)` (call `to_jsonable` first)
   closes the loop; it is one line and the docs can then describe one route from a nested object to a
   file.

### 37.2 `nested`: the default of `escape` loses information **[verified]** (P3)

```python
from_flat_dict(
    to_flat_dict({"a.b": 1, "c": {"d": 2}})
)  # {'a': {'b': 1}, 'c': {'d': 2}}  (the key 'a.b' is gone)
to_flat_dict({"a.b": 1, "a": {"b": 2}})  # ValueError: Key collision: ... 'a.b' (good)
```

`to_flat_dict` and `from_flat_dict` have `escape=False`. With the default, a key that contains the
separator silently changes the structure on the way back; the collision check only catches the case
where the two flattened keys are equal. Suggestions (breaking, fine for 2.0): default `escape=True`; or,
if the unescaped form is wanted for readability, make `to_flat_dict` raise `ValueError` when a key
contains the separator and `escape=False`.

### 37.3 `NativeReducer.mean` raises where the other reducers return `nan` **[verified]** (P3)

A differential fuzz of the three reducers (`mean`, `median`, `std`, `min`, `max`, `sort`, about 1,800 inputs over `0.0, -0.0, 1.0, 2.5, -3.0, inf, -inf, nan, 1e308, -1e308, 1e-320`) agrees everywhere except:

- `NativeReducer().mean([inf, -inf])` raises `ValueError: -inf + inf in fsum`; NumPy and torch return
  `nan`. 161 of the inputs. `mean([10**400, 10**400])` raises `OverflowError`.
- `std` on values near the limits: NumPy and torch overflow or underflow (`std([0, 1e308])` is `inf`,
  `std([0.0, 1e-320])` is `0.0`) where `NativeReducer` is right (`7.07e+307`, `7.07e-321`). This is not
  a defect of coola but is worth a sentence in the docstring of `BaseReducer.std`.
- `NumpyReducer().mean([1e308, 1.0, -1e308])` is `0.0` and torch gives `0.333...`: the naive sum
  loses the `1.0`. Same remark.

Fix for the first: catch the `ValueError` of `math.fsum`, or test `math.isinf` of the values first and
return `nan` when both signs are present, so that the three reducers agree; add the case to
`tests/properties/test_reducer.py` as a cross-reducer property.

### 37.4 `random_seed` and a CUDA/MPS generator that is initialised inside the block **[reading]** (P3)

`TorchRandomManager.get_rng_state` saves the CUDA state only `if torch.cuda.is_initialized()` and the
MPS state only if MPS is available. `manual_seed` calls `torch.manual_seed`, which seeds CUDA lazily
(it queues the seed if CUDA is not initialised). So

```python
with random_seed(0):  # CUDA not initialised: the state of 'torch.cuda' is not saved
    x = torch.randn(1, device="cuda")  # CUDA is initialised here with the seed derived from 0
# after the block, set_rng_state restores the CPU state, the CUDA generators keep the seed of 0
```

Every later CUDA random number is then deterministic although the user asked for a scoped seed. Not
reproduced (no GPU). Fix: in `get_rng_state`, when CUDA is available but not initialised, record that
fact (`state["torch.cuda"] = None`) and, in `set_rng_state`, re-seed the CUDA generators from fresh
entropy (`torch.cuda.seed_all()` would do) when the saved value is `None` and CUDA is now initialised;
or document the limit in `random_seed` and in `uguide/random.md`.

### 37.5 `formatting` and `summary` **[verified]** (P3)

- `str_human_byte_size` (`formatting/quantity.py`) has no unit above TB: `1024**5` is `'1,024.00 TB'`,
  `10**30` is `'909,494,701,772,928,256.00 TB'`. Add `PB`, `EB`, `ZB`, `YB`. The units are powers of
  1024 but are written `KB`, `MB`: `KiB` is the IEC name, or switch to powers of 1000; say which in the
  docstring. `str_human_byte_size(True)` returns `'1.00 B'` (a `bool` is accepted).
- `truncate_str('abc', 0)` and `truncate_str('abcdef', 2)` raise a `ValueError` ("max_len (2) must be greater than or equal to the suffix length (3)"). A truncation helper that is used to build a message
  should not raise because the limit is small: return `text[:max_len]` when the suffix does not fit.
- `char_diff_summary('abc', 'abd')` is `'3 -> 3 chars (+0 chars, +0.0%).'`: it counts lengths, not
  character differences; the name suggests the second.
- `summarize(torch.sparse_coo_tensor)` shows no `layout` (the shape and dtype of a sparse tensor look
  like a dense one), and `summarize(torch.nested.nested_tensor(...))` returns
  `<summary failed: RuntimeError>` instead of the fields that can be read (`dtype`, `device` do not raise). Add `layout=` to the tensor summary when it is not `torch.strided`, and catch the `RuntimeError` per field.

## 38. Performance

### 38.1 A shared sub-object is visited once per reference **[verified]** (P2)

`hash_object` memoizes the digest of a shared child (the cost for a DAG is linear, documented in the
hashing guide) and `summarize` is bounded by `max_items`. The other helpers re-visit a shared object
every time it is referenced, so a structure in which each level references the next one twice takes
`2^n`:

```python
def dag(n):
    x = [1]
    for _ in range(n):
        x = [x, x]
    return x


d = dag(20)  # 2 million nodes in the tree view, 20 objects in memory
recursive_apply(d, lambda x: x)  # 4.1 s
objects_are_equal(d, dag(20))  # 1.9 s
sum(1 for _ in bfs_iterate(d))  # 2.4 s
sum(1 for _ in dfs_iterate(d))  # 0.9 s
to_jsonable(d)  # 1.7 s
to_flat_dict(d)  # 3.3 s
hash_object(d)  # 0.0 s
summarize(d)  # 0.0 s
```

`n = 30` would take about a thousand times longer (not run). The same happens, less dramatically, for realistic structures: a config in
which the same large sub-dict is referenced from 1000 places is compared 1000 times, and a list that
contains the same `np.ndarray` object many times compares the array each time. Also,
`recursive_apply([s, s], f)` returns a list whose two items are two distinct copies
(`result[0] is result[1]` is `False`): the sharing is lost, which matters if a node is later mutated.

- **Equality**: `Engine.run` already has `visiting` (the pairs `(id(a), id(b))` on the current path, for
  the cycles). Add a `proven` set of the pairs whose children were all found equal (add the key when the
  child iterator is exhausted at `stack.pop()`), and return `True` for a pair in it. Sound as long as the
  objects are not mutated during the comparison, which the cycle detection already assumes. This is also
  the fix for the repeated array.
- **`recursive_apply`**, **`to_jsonable`**, **`to_flat_dict`**, **`dfs_iterate`**, **`bfs_iterate`**:
  `recursive_apply` is the base of the first three. A memo `id(obj) -> result` (valid only while the
  input is alive, which is true during the call) gives a linear time and preserves the sharing.
  `dfs_iterate`/`bfs_iterate` yield leaves, so a visited set would change what they yield: add a
  keyword `unique=False` (visit each container once) rather than change the default, and document the
  cost in the docstrings.

### 38.2 A list of floats takes 80x the time of `==` **[verified]** (P3)

```
python == on two lists of 1e6 floats      1.7 ms
objects_are_equal(...)                  141.5 ms
objects_are_allclose(...)                84.6 ms
objects_are_equal on two lists of 1e6 ints  39.2 ms     (has a fast path)
```

`SequenceEqualityTester` has a fast path for lists and tuples of exact scalars
(`is_exact_scalar_sequence`: `int`, `str`, `bytes`, `bool`, `None`), and `float` is excluded because of
`NaN` and the tolerance. But `actual == expected` being `True` is sound for floats in every
configuration (every pair is `==`, hence equal with any tolerance and any `equal_nan`; `-0.0 == 0.0`
too), so the fast path can accept `float` for the `True` answer: `if actual == expected: return True`
before the check of the types (the types are the same if both lists contain only floats; keep the
`list(map(type, ...))` comparison). Only the `False` answer needs the slow path when `equal_nan` or a
tolerance is set (already how the code is structured). The same shortcut applies to `complex`.
A single list of floats is the common shape of a metric history or a loss curve.

## 39. Tests, CI and maintenance

### 39.1 A test fails on `main`: `docformatter` rewrote the source of a generated module **[verified]** (P1)

```
$ .venv/bin/pytest tests/integration/testing/test_marks.py::test_real_mark_is_importable_in_a_generated_module
E   File ".../test_real_mark_is_importable_in_a_generated_module.py", line 1
E     From coola.testing.marks import numpy_available,
E          ^^^^^
E   SyntaxError: invalid syntax
```

The test passes the source of a module to `pytester.makepyfile("""...""")`. The commit `8b53d01b` ("Fix format (#2534)") changed it to:

```python
    pytester.makepyfile(
        """From coola.testing.marks import numpy_available,
        numpy_not_available.

        @numpy_available
```

The cause is `docformatter` (pre-commit hook, `types: python`, no `files`): it takes the triple-quoted string that is the only argument of the call for a docstring, capitalizes the first word, wraps the first line, adds a period and re-indents the rest. This is reproducible: running `docformatter --config pyproject.toml --in-place` on the version of the file before `8b53d01b` gives exactly the text above. The failure is deterministic and comes from the formatter, not from what the commit was about. I did not look at whether CI reports it (`ci-test.yaml` runs `inv test.integration`, which includes the test). The other `makepyfile` calls of the file are not affected because their source is an expression (`HEADER + """..."""`) and not the only argument.

Fix, in order of preference: (a) restrict the hook to the code that has docstrings with
`files: ^src/` (the `tests/**` per-file-ignore of ruff already disables the docstring rules `D`); (b) move the sources to module-level constants
(`_MARK_MODULE = "from coola.testing.marks import ..."`, as `test_fixtures.py` does with `TEST_FILE`),
or build them with `textwrap.dedent`; (c) both. Restore the source of the generated module. The pre-commit job (`ci-pre-commit.yaml`) is where a hook that rewrites a file must fail the build; check that it does so for `docformatter` (the commit is titled "Fix format", so the rewrite was merged as a formatting change and nobody ran the integration tests on it).

### 39.2 The non-strict `xfail` markers hide a defect and a wrong test **[verified]** (P2, _carried over_ 29.1)

`tests/properties/test_library_types.py` has three `xfail(strict=False)`. Running them with `--runxfail`:

| Test                                                     | Marker says              | Actual state                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_equal_float_arrays_have_the_same_hash`             | NaN bit patterns (14.3)  | The defect is fixed (`hash_object(np.array([nan])) == hash_object(np.array([-nan]))`, also for a payload NaN). The test fails for another reason: `floats` is `sampled_from([0.0, -0.0, 1.0, -1.0, nan, -nan, inf])` and the test flips the sign of every item, so `[1.0]` against `[-1.0]` is a legitimate difference and the first `assert objects_are_equal(first, second, equal_nan=True)` fails. |
| `test_different_structured_arrays_have_different_hashes` | structured dtypes (14.1) | **Passes**: the defect is fixed and the marker is stale (non-strict, so XPASS is silent).                                                                                                                                                                                                                                                                                                             |
| `test_different_frames_have_different_hashes`            | pandas metadata (14.2)   | The metadata (names, attrs) is fixed. The remaining failure is the real defect 34.1 (category order).                                                                                                                                                                                                                                                                                                 |

Fix: remove the three markers; restrict the sample of the first test to the values that are equal to
their sign flip (`0.0`, `-0.0`, `nan`, `-nan`; for `1.0`, `-1.0` and `inf` do not flip); after 34.1 the third passes. Make
`xfail_strict = true` apply (it is set in `pyproject.toml` but `strict=False` on the marker overrides it;
`grep -rn "strict=False" tests` should return nothing), and consider a `scripts/check_xfail.py` that
runs the suite with `--runxfail` weekly to catch the reasons that change.

### 39.3 Convention slips in the unit tests **[verified]** (P3, _carried over_ 29.2)

Counted with an AST script over `tests/unit` (6289 tests):

- **103** tests with more than one `assert` or `pytest.raises` (CLAUDE.md: one behaviour, one
  assertion). Most in `utils` (18), `factory` (11), `hashing` (11), `registry` (11), `equality` (8),
  `rounding` (8), `validation` (8), `reducer` (7). Examples: `unit/equality/test_engine.py:374-390`
  (`test_warn_once_*` has a `pytest.warns` and an `assert` that both check the result),
  `unit/factory/test_instantiation.py:102`.
- **56** tests without an assertion or a `pytest.raises` (25 in `validation`, which is the "does not
  raise" idiom, 9 in `display`). The `display/rich/test_print.py` tests (`print_markdown`,
  `print_pretty`) call the function and assert nothing: capture the output with `capsys`/`Console(file=...)`
  and assert it contains the title and the message, or they cannot fail. `test_installs_rich_handler`
  ends with a comment (`# raises if not found`).
- **82** section headers whose lines do not match the rule (the second line is `#`, 5 spaces, the name,
  5 spaces, `#`, and the other two repeat `#` to the same length): `unit/equality/test_engine.py:204`
  (`#     Tests cycle   #`, 3 spaces), `unit/display/rich/test_print.py:22` (the `#` lines are longer than
  the title line), `unit/equality/_helpers/test_numpy.py:260`, `unit/display/test_pydantic.py:675`, etc.
  A 15-line script can check and fix them; run it as a pre-commit hook (`local` repo hook).
- **29** tests whose body contains a loop or a conditional (some of them in a local helper generator, for instance `unit/equality/test_engine.py:163`; others are real logic: `unit/nested/test_flat.py:737` and `:744` unroll a nesting with `for _ in range(DEEP_NESTING_DEPTH)`, and `unit/hashing/test_engine.py` builds a DAG in a loop). Use a helper function with its own test, `parametrize`, or move the test to `tests/integration`.
- `time.sleep` is used in two integration tests (`integration/registry/test_type.py:409,600`: "encourage race conditions" and "let the thread miss the fast path and block on the lock"). It is allowed in `integration`, but the second test depends on the speed of the machine; a `threading.Event`/`Barrier` that the thread sets when it is about to block is deterministic.

### 39.4 Modules without a mirrored test file and functions never named in a test **[verified]** (P3)

`tests/unit/<package>/test_<module>.py` should exist for each module (CLAUDE.md). Missing:
`display/rich/print_.py`, `equality/tester/_array.py`, `equality/tester/base.py`,
`equality/tester/object.py`, `factory/constants.py`, `hashing/{jax,numpy,pandas,polars,pyarrow,torch}.py`
(their cases are presumably in `test_array_libraries.py`, which does not say which module a test covers),
`iterator/base.py`, `recursive/base.py`, `summary/object_fields.py`. The coverage is 100%, so the code is
executed; the point is that a test that fails does not point to the module.

CLAUDE.md also says that every function, private ones included, has its own unit test. A script that
lists the functions of `src` whose name appears nowhere in `tests/` gives 162 names. Most are private
helpers that are only covered through the public function, for instance
`equality/_helpers/elementwise.py` (`_is_torch`, `_is_integer`, `_itemsize`, `_cast`, `_to_double`,
`_integer_close`), `equality/_helpers/numpy.py` (`_uses_tolerance`, `_is_nan_like`, `_durations_close`,
`_supports_nan`), `equality/engine.py` (`_fail_types`, `_check_depth`, `_record_failure`),
`equality/tester/pandas.py` (`_same_index`, `_index_equals`, `_same_dtypes`, `_column`, `_series_equals`,
`_has_nan_dtype`, `_same_label`), `io/base.py` (`_fsync_file`, `_fsync_directory`, `_link`),
`nested/flat.py` (`_freeze`, `_split_key`, `_escape_key`, `_set_nested`, `_set_leaf`, `_build_key`),
`registry/type.py` (`_encode_type`, `_decode_type`, `_load_groups`, `_resolve`, `_resolve_uncached`),
`summary/engine.py` (`_cut`, `_repr`, `_repr_list`, ..., `_visit`), `reducer/base.py`
(`_prepare_input`, `_check_quantiles`), and also public ones: `hashing/numpy.py:canonicalize_floats`
and `hash_array_content`, `utils/fallback/factory.py:make_fake_module`,
`utils/imports/optional.py:check_colorlog`, `check_rich`, `check_scipy`, `utils/introspection.py:get_type_constructor`,
`equality/_helpers/torch.py:tensor_close`. The list is reproducible with the script of the appendix. Do it package by package, with the existing helpers; the elementwise kernel in particular
deserves its own tests (the dtype pairs int8/uint64/float16/bfloat16 and the overflow cases are where an
error would be), because three libraries share it.

### 39.5 Boilerplate: 56 import guards and 12 `is_*`/`check_*` pairs **[verified]** (P3)

- `grep -rn "pragma: no" src` gives 65 lines; 56 of them are the `else:  # pragma: no cover` of the optional import guard
  `if TYPE_CHECKING or is_numpy_available(): import numpy as np / else: from coola.utils.fallback.numpy import numpy as np`.
  CLAUDE.md forbids `# pragma: no cover` on a line that can be tested; these lines can be (a job runs the package without its optional libraries), but the coverage of that job is not combined with the main one as far as I can see (`ci-test.yaml` uploads a single cell), so the pragma is the practical choice. Combining the coverage of the two jobs (`coverage combine`) and removing the pragmas would make the 100% mean something again.
- `utils/imports/optional.py` (530 lines) has one `@availability_cache def is_X_available()` and one
  `def check_X()` per library (12 pairs), each with a docstring and a doctest, which are the same code
  with a different name. `check_rich`, `check_scipy` and `check_colorlog` are never referenced by a test
  (39.4). A small factory `make_optional("numpy") -> (is_available, check)` removes 300 lines. The
  explicit functions are kept for the type checker; assigning the factory results at module level
  (`is_numpy_available, check_numpy = make_optional("numpy")`) types fine with `Callable[[], bool]` and
  `Callable[[], None]`.
- `module_available` (used by every `is_*_available`) imports the module, while `package_available`
  uses `find_spec`; the two names do not say it. A docstring line in each is enough.

## 40. Documentation and packaging

### 40.1 `docs/docs/dev` describes removed code; `dev` and `review/` duplicate each other **[verified]** (P3)

- `docs/docs/dev/iterator.md:37,156` and `docs/docs/dev/recursive.md:41,150` give `DataclassMarker` as the
  registration of the dataclass child finder and transformer. `DataclassMarker` was deleted in #2472;
  the code registers `registry.register_fallback(dataclasses.is_dataclass, DataclassChildFinder())`
  (`iterator/interface.py:250`). `dev/summary.md` and `dev/registry.md` mention it too. The mentions of
  `convert_to_jsonable`, `resolve_or`, `enter_dataclasses` and `coola.tools` in `dev/nested.md`,
  `dev/registry.md`, `dev/recursive.md` and `dev/utils.md` are in the history paragraphs (acceptable if
  they say so).
- `docs/docs/dev` has 7436 lines in 22 files and `review/` 11608 lines. About half of the lines of a
  `review/<package>.md` also appear in `docs/docs/dev/<package>.md` (338 of 862 for hashing, 309 of 716
  for summary, 208 of 608 for registry), and the published pages contain the decision logs ("Decided",
  "Recommended": 18 lines in `dev/utils.md`, 16 in `dev/identifier.md`, 10 in `dev/hashing.md`, ...). The reader of a published
  architecture page wants the current design, the contracts and the extension points, not the options
  that were weighed.
  Suggestion: once a design is implemented, keep **one** current page per package in `docs/docs/dev`
  (architecture and contracts, with the `pycon` examples that are doctested) and move the decision
  record to `docs/adr/NNNN-title.md` (or delete `review/<package>.md`, whose content is in git
  history), and add a check that greps `docs/docs/dev` for the names that `src` no longer defines
  (a 20-line test: collect the backticked identifiers of the docs that look like `CamelCase` or
  `snake_case()` and check that they are importable from `coola`, with an allow-list for the history
  paragraphs).

### 40.2 Packaging metadata **[verified]** (P3, _carried over_ 30.1)

- `Development Status :: 5 - Production/Stable` is the classifier of a `2.0.0a1` alpha; use
  `3 - Alpha` (then `4 - Beta`, `5` at 2.0.0).
- `license = "BSD-3-Clause"` (an SPDX expression, PEP 639) **and** the classifier
  `License :: OSI Approved :: BSD License`: PEP 639 says to use one or the other, and setuptools-based consumers of the metadata reject the combination. `uv build` (hatchling) accepts it, and the wheel metadata carries both (`License-Expression` and `Classifier`). Remove the classifier.
- `CITATION.cff` still says `title: "COOLA: A Python library to check if two complex/nested objects are
equal or not."`, `date-released: 2023-01-01` and has no `version`, `abstract` or `keywords`, while the
  library is now "compare, hash and summarize". Update the title, the version (`2.0.0a1`, which a
  release script can fill) and the date.
- The `all` extra installs `pytest` (the `testing` extra is part of it) in an environment where nobody
  runs the tests; exclude `testing` from `all`.

## 41. Status of round 5

Checked on `main` at `7cd495e8` by running the repro of each item.

| Round-5 ID                         | Status     | Notes                                                                                                                                                     |
| ---------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 22.1                               | Fixed      | `boolean`, `string`, tz-aware, `Period` and `Interval` series with a missing value hash.                                                                  |
| 22.2, 22.3, 22.5, 22.6, 22.7, 22.8 | Fixed      | bfloat16 `-0.0` hashes like `0.0`; `hash_object(pa.array(np.arange(10**6)))` takes 12 ms; huge integers, lone surrogate, aware `time`.                    |
| 22.4                               | Partly     | `get_stable_registry` is used by `identifier/content.py`; the optional `HASH_FORMAT_VERSION` constant (item 3 of 22.4) is not there.                      |
| 23.1                               | Fixed      | Two models are equal after a `cached_property` was read on one of them.                                                                                   |
| 23.2                               | **Open**   | `objects_are_allclose({1.0}, {1.0000001}, atol=1e-3)` is still `False`: the elements of a set are compared with `==`.                                     |
| 23.3                               | **Open**   | Plain objects, exceptions and `functools.partial` compare `False` even if all attributes are equal (`ObjectAttributesEqualityTester` is not the default). |
| 24.1, 24.2, 24.3                   | Fixed      | `track_fork`, `temporary_default_registries`, `COOLA_DISABLE_PLUGINS`/`get_loaded_plugins`.                                                               |
| 25.1, 25.2, 25.3                   | Fixed      | Cycles and any depth with an explicit stack; deterministic summaries of sets (checked under 3 `PYTHONHASHSEED`); dataclasses are entered.                 |
| 26.1, 26.2                         | Fixed      | `DataclassTransformer`; the `TypeError` of `flatten_mapping` names the key.                                                                               |
| 27.1                               | Documented | A nested `_target_` is not resolved (`resolve_object({... "a": {"_target_": ...}})` returns the inner dict unchanged); the guide says so.                 |
| 27.2                               | Fixed      | `is_object_config(..., import_target=False)` does not import.                                                                                             |
| 28.1                               | Won't fix  | `coola/__init__.py` states the choice (only `__version__`).                                                                                               |
| 28.2                               | Fixed      | Options of `summarize`, `recursive_apply`, `dfs_iterate`, `bfs_iterate` are keyword-only.                                                                 |
| 28.3                               | **Open**   | `max_depth` still raises above the limit in `equality` and is a display depth in `summary` (now documented as such).                                      |
| 28.4                               | Fixed      | The README quick start shows `assert_objects_equal` with its path and mentions `compare` (but see 35.1 and 35.3).                                         |
| 29.1                               | **Open**   | See 39.2: one marker is stale (the test passes), one hides a wrong test, one hides 34.1.                                                                  |
| 29.2                               | **Open**   | See 39.3 and 39.4.                                                                                                                                        |
| 30.1                               | **Open**   | See 40.2.                                                                                                                                                 |
| Items of the `review/<package>.md` | Fixed      | As listed in the table of round 5 (kept in git history); no regression seen in the probes of this round.                                                  |

## 42. Suggested order

1. **Unblock `main`:** 39.1 (restore the test and stop the formatter from rewriting `tests/`).
   One small PR; the rest of the list is easier to review once the suite is green.
2. **Tests first, because they make the next steps visible:** 39.2 (remove the markers, fix the
   strategy), which makes 34.1 an ordinary red test; then 39.3 (the header script and the multi-assert
   tests can be done mechanically, package by package).
3. **Contract fixes, one PR each, with the regression test that fails without the change:** 34.1
   (category order), 34.2 (`timedelta64` and `datetime64`), 36.1 (NumPy integers), 37.3 (`mean` of
   `[inf, -inf]`), 37.2 (default of `escape`).
4. **Diagnostics:** 35.1 (messages of the table testers), then 35.2 (document the rule first, the
   engine fallback second), 35.3.
5. **Hash format batch (the hash values change once; do it with the format version):** 34.3 (the new
   hashers) together with 34.2.
6. **Performance:** 38.2 (the float fast path, five lines), then 38.1 (the memo of the equality engine,
   then `recursive_apply` and the helpers built on it). Add both to `tests/benchmarks`.
7. **API and `utils`:** 36.2, 36.3, 37.1, then the decision of 36.4 (the 2.0 window).
8. **Maintenance:** 39.4, 39.5, 40.1, 40.2 and 37.4/37.5 as time permits.

## Appendix: the script of 39.4

The other checks of section 39 are small AST scripts over `tests/unit` (count the `assert` and
`pytest.raises` per `test_*` function; match the three lines of each header). They are not in the
repository. The three that would be worth keeping in CI are (a) the header checker (39.3), (b) the
report below (39.4), and (c) the check that the identifiers of `docs/docs/dev` exist (40.1).

```python
import ast, pathlib, re

tests = "".join(p.read_text() for p in pathlib.Path("tests").rglob("*.py"))
for p in sorted(pathlib.Path("src/coola").rglob("*.py")):
    if p.name == "__init__.py":
        continue
    rel = p.relative_to("src/coola")
    if not (pathlib.Path("tests/unit") / rel.parent / f"test_{rel.name}").exists():
        print("no mirrored test file:", rel)
    for node in ast.walk(ast.parse(p.read_text())):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("__"):
            if not re.search(rf"\b{re.escape(node.name)}\b", tests):
                print("never named in a test:", f"{rel}:{node.lineno}", node.name)
```

(The methods that implement an abstract method with the same name, such as `expand` or
`objects_are_equal`, are tested through the public function; exclude them.)
