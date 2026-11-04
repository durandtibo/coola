# Codebase review: suggestions (round 7)

This file **replaces the previous `review/suggestions.md`** (round 6, items 34.1 to 42). That version is
in git history (last touched by `7df3c102`). Section 51 gives the status of its items. Nothing below
repeats an item of round 6 or of the `review/<package>.md` design documents, except where an item is
explicitly marked _carried over_ or _follow-up_.

The numbering **continues** the previous rounds (sections 43 to 52), so that an ID such as `43.1` in a
branch name or a commit message is unambiguous.

## How this review was done

Checkout of `main` at `8055c876` (`.venv`, Python 3.14.8, NumPy 2.5.4, pandas 3.1.0, polars 2.1.0,
pyarrow 26.0.0, torch 2.15.0, xarray 2026.9.0, JAX 0.12.0, pydantic 2.14.2, SciPy 1.18.1). Baseline, run
on the clean checkout:

- `pytest tests/unit tests/integration`: 13396 passed, 34 skipped, 24 xfailed, **1 failed** (the
  round-6 item 39.1, still red), in 41 s, with 100% line and branch coverage (6670 statements, 1564
  branches).
- `pytest tests/properties`: 697 passed, 2 xfailed, **1 xpassed** (the non-strict markers of 39.2).
- `ruff check src tests` and `ty check` are clean.
- The Markdown of `README.md`, `docs/docs/index.md`, `get_started.md`, `uguide/*.md` and `migration/*.md`
  passes `python -m doctest` (all 27 files), and the four scripts of `examples/` run.
- 38 commits landed since round 6 (`7cd495e8..8055c876`). They are compatibility fixes for old versions
  of pyarrow, polars, pandas and torch, the removal of `polars.LazyFrame` from `coola.equality` and the
  rewrite of the developer guide. I saw no regression from them.

What was done:

1. **Read-through with the docstrings stripped** (a small AST script) of `registry`, the equality engine
   and the collection/scalar/default/equal/tolerant testers, `identifier`, `nested`, `factory`, `io`,
   `validation`, `random`, `rounding`, `display`, `formatting`, `testing` and most of `utils`.
2. **Probes**, about 700 one-line calls over every package (equality of 150 pairs of values, allclose of
   80, `to_jsonable` of 55, `summarize` of 85, rounding of 60, reducers of 3 x 35, iterators and
   `recursive_apply` of 55, table testers of 45 pairs), listed per item below. Items marked
   **[verified]** were reproduced on this checkout and the repro is given. Items marked **[reading]**
   come from reading the code and were not reproduced.
3. **A differential fuzz** of 301 values (scalars, containers, 14 NumPy dtypes x 6 shapes, torch dtypes,
   layouts and views, 39 pandas, 18 polars, 17 pyarrow, 10 xarray, 4 JAX, dataclasses): for the 45 150
   pairs, `objects_are_equal(a, b) == objects_are_equal(b, a)`, equal objects have the same hash, equal
   objects are `objects_are_allclose`, and a copy of a value is equal to it. Result: **no violation
   except the round-6 item 34.1** (categories), and 11 NaN-bearing values that are not equal to their
   copy, as documented. The hashes are also identical under `PYTHONHASHSEED=1,2,3` for 8 composite
   objects (sets of strings, sets of sets, dicts with tuple keys, dataclass with a set, Enum members).
4. **The support matrix** of `tests/integration/test_support_matrix.py` was extended in a scratch script
   to `objects_are_allclose`, `to_jsonable`, `round_values`, `recursive_apply` and `dfs_iterate`
   (section 45.2).
5. **Scale tests**: 1e6-element lists, 1e6-key dicts, 1e7-element arrays and tensors, 1e6 x 5 data
   frames, 1e5 nested dicts, deep structures (up to 1e4 levels), a sparse tensor of shape 20000 x 20000.
6. **Conventions**: the rules of `CLAUDE.md` that can be checked mechanically, with AST scripts over `src`
   and `tests` (section 49).

The library is in a very good state, and the fuzz found no new contradiction between equality, hashing
and allclose. The new findings are narrower: **a summary that is not deterministic** (43.1), **sparse
tensors that are densified** (43.2), **`jsonable` and `round_values` that do not cover the types the
other modules cover** (43.4, 45.2), **diagnostics of the table testers** (44.2) and **one dependency
cycle between `equality` and `io`** (45.1). The round-6 defects are all still open (section 51), and
39.1 still turns the CI red.

Priority: **P1** wrong result, crash on realistic input or a red test on `main`, **P2** inconsistent or
surprising behaviour, **P3** cleanup or design.

## Summary

| ID   | P   | Area         | Finding                                                                                                         |
| ---- | --- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| 51   | P1  | tests        | _Carried over_ 39.1: `test_real_mark_is_importable_in_a_generated_module` is still red on `main`                |
| 43.1 | P2  | summary      | The summary of a set with items that cannot be sorted depends on `PYTHONHASHSEED`                               |
| 43.2 | P2  | equality     | A sparse tensor is converted to a dense one to be compared or hashed: 1.2 GB and 1 s for 2 non-zero values      |
| 43.3 | P2  | rounding     | `round_values(..., recursive=True)` leaves JAX, xarray, `Decimal`, complex values unrounded without a word      |
| 45.2 | P2  | jsonable     | 29 of the 74 types of the support matrix cannot be converted; sparse and meta tensors raise raw errors          |
| 44.2 | P2  | equality     | "tables have a different structure" does not say which part, and a value difference has no row or value         |
| 43.4 | P3  | utils        | `clear_availability_caches` does not reset the CUDA/MPS checks; the MPS check allocates a tensor                |
| 43.5 | P3  | summary      | `max_characters` counts the header, the result is longer than the limit, and `truncate_str` does otherwise      |
| 43.6 | P3  | registry     | `bool(registry)` imports every optional library                                                                 |
| 43.7 | P3  | utils        | `temp_env_vars({"X": None})` sets `"None"`; `get_file_size` follows a link or not by the argument               |
| 44.1 | P3  | equality     | Identity short-circuits NaN: `{"a": np.nan}` equals itself, `arr` equals `arr` but not `arr.copy()`             |
| 44.3 | P3  | equality     | _Carried over_ 28.3: `max_depth=100` is the only depth limit of the library and protects nothing                |
| 45.1 | P3  | architecture | `equality` imports `io` (a local import hides the package cycle `equality` <-> `io`)                            |
| 45.3 | P3  | recursive    | `recursive_apply`, `dfs_*` and so `round_values`, `to_flat_dict` treat a pydantic model as a leaf               |
| 45.4 | P3  | features     | Small gaps: negative `decimals`, `Fraction`/complex JSON, `save_json` of NaN, no `io` by extension              |
| 46.1 | P3  | performance  | _Follow-up_ 38.1/38.2: the float list is 25x `==` (not 80x), a shared DAG is still exponential                  |
| 46.2 | P3  | performance  | Hash throughput of containers: 0.7 us per int, 2.4 s for a dict of 1e6 items                                    |
| 47   | P3  | conventions  | Six `raise ValueError` for a range, a finite or a membership check, one `__init__` docstring, `functools.cache` |
| 49   | P3  | packaging    | `Typing :: Typed` classifier, upper bounds of the optional dependencies, examples for 4 of 19 packages          |

## 43. Behaviour of the library

### 43.1 The summary of a set whose items cannot be sorted depends on `PYTHONHASHSEED` **[verified]** (P2)

`summarize` promises a deterministic text (round 5, item 25.2, checked "under 3 `PYTHONHASHSEED`"). That
holds for the sets that can be sorted. With mixed items the order of the set is shown, and the order of
a set of strings changes from one run to the next:

```python
from coola.summary import summarize

print(summarize({1, "a", "b", "c", "d", None}, max_items=3))
```

```text
PYTHONHASHSEED=1   (0): None  (1): 1  (2): d
PYTHONHASHSEED=2   (0): None  (1): 1  (2): b
PYTHONHASHSEED=3   (0): None  (1): 1  (2): c
```

The cause is `coola.utils.selection.smallest_items` (used by `summary/set.py:42` and `summary/engine.py:273`):

```python
try:
    return sorted(data) if limit is None else heapq.nsmallest(limit, data)
except TypeError:
    return list(islice(data, limit))  # the iteration order of the set
```

The summary of a set of mixed items is then different in two runs of the same program, which makes a
logged summary hard to diff, and breaks a test that compares the summary text. `coola.utils.mapping.sort_by_keys`
already has the right three-step fallback (the items, then `(type name, item)`, then `(type name,
repr(item))`). Move it to a public helper (for example `coola.utils.selection.sort_items`) and use it in
`smallest_items`; `nsmallest` accepts `key=`. Add a test that runs `summarize` of the set above in two
subprocesses with different `PYTHONHASHSEED` (it belongs in `tests/integration/summary`), and a unit test
of `smallest_items({1, "a"}, limit=1)` with a set built in a fixed order.

### 43.2 A sparse tensor is converted to a dense one to be compared or hashed **[verified]** (P2)

`equality/_helpers/torch.py:49-53` and `:91-92`, and `hashing/torch.py:58-59`, call `to_dense()` on every
tensor whose layout is not `torch.strided`. A sparse tensor exists to avoid the dense storage, so the
cost is the product of the shape and not the number of stored values:

```python
import torch
from coola.equality import objects_are_equal
from coola.hashing import hash_object

n = 20_000
mk = lambda: torch.sparse_coo_tensor(
    torch.tensor([[0, 5], [1, 7]]), torch.tensor([1.0, 2.0]), (n, n)
)
objects_are_equal(
    mk(), mk()
)  # True after 1.05 s and a peak RSS +1.2 GB (2 stored values, 1.6 GB dense)
hash_object(mk())  # 4.7 s
```

With `n = 1_000_000` (a normal size for an embedding table or a graph) the same call would ask for 4 TB. The same code path is used by `objects_are_allclose`. The SciPy tester
(`equality/tester/scipy.py`) is the model: it compares the positions and the stored values and never
builds a dense array ("the explicitly stored zeros are ignored"). For torch:

- COO: `t.coalesce()`, compare `indices()` and `values()` (the tolerance applies to the values);
- CSR/CSC/BSR/BSC: compare `crow_indices()`/`col_indices()` (or the compressed pair of the layout) and
  `values()`;
- hash: feed the shape, the layout name, the indices and the values (the format changes for the sparse
  tensors only, so it goes in the same release as 34.3).

Until that is done, at least `summarize` and the docs should say that a sparse tensor is compared in its
dense form, and `to_dense()` should be guarded by a size check (`numel() * itemsize` above a limit
raises a `ValueError` that names the shape) so that the process is not killed by the OOM-killer. Note
also that `hash_object(torch.eye(2).to_sparse())` equals the hash of the dense `torch.eye(2)` and of the
CSR tensor, while `objects_are_equal` of the three is `False`: the "hash collision between unequal
objects" is allowed by the contract, but it is one more reason to put the layout in the hash.

### 43.3 `round_values(..., recursive=True)` silently leaves some numeric leaves unrounded **[verified]** (P2)

The docstring says that, with `recursive=True`, "a leaf that is not supported (a string, `None`, a date,
...) is kept as it is instead of raising an error". The leaves that are numbers are in that group too:

```python
import decimal, jax.numpy as jnp, xarray as xr
from coola.rounding import round_values

x = {
    "x": jnp.array([1.2345]),
    "d": decimal.Decimal("1.2345"),
    "c": 1.2345 + 1j,
    "xa": xr.DataArray([1.2345]),
}
round_values(x, decimals=1, recursive=True)
# {'x': Array([1.2345]), 'd': Decimal('1.2345'), 'c': (1.2345+1j), 'xa': <xarray.DataArray ... array([1.2345])>}
round_values(jnp.array([1.2345]), decimals=1)  # UnsupportedTypeError (recursive=False)
```

A user who rounds a result before `objects_are_equal` or before logging a nested dict that contains a JAX
array or an xarray object gets the unrounded values and no signal. `recursive=False` raises, so the two
modes disagree on the same leaf. Options, from the smallest:

1. Add `on_unsupported: Literal["raise", "ignore"] = "ignore"` for the recursive mode (so the default
   does not change) and document that `None`, `str`, `bytes`, `bool`, dates and enums are the only
   leaves that are never an error; `"raise"` raises for any other leaf.
2. Add the missing rounders: JAX (`jax.Array` goes through `round_numpy`), xarray (`DataArray`,
   `Dataset`, `Variable`: round the data and keep the coordinates), `Decimal` (`quantize`) and complex
   (round the two parts). These are the same types that `equality`, `hashing` and `summary` support.
3. Support a negative `decimals` (`round_values(1234.5, decimals=-2)`), which `round` and `numpy.round`
   accept and which `validate_arguments` rejects (`rounding/common.py:91`).

### 43.4 `clear_availability_caches` does not reset the CUDA/MPS checks **[verified]** (P3)

`coola/utils/tensor.py` caches `get_available_devices`, `is_cuda_available` and `is_mps_available` with
`functools.cache`, while `coola.utils.imports.availability_cache` exists "instead of `functools.cache`
on the functions that check if a package is available", and `CLAUDE.md` asks for `LazySingleton` or the
registered cache and not `functools.cache`:

```python
from coola.utils import tensor as T
from coola.utils.imports import clear_availability_caches

T.is_cuda_available()
clear_availability_caches()
T.is_cuda_available.cache_info()  # CacheInfo(hits=0, misses=1, maxsize=None, currsize=1): not cleared
```

A test that monkeypatches `torch.cuda.is_available` and calls `clear_availability_caches()` still reads
the old answer. Replace `@cache` by `@availability_cache` (three lines). `is_mps_available` also
allocates a tensor on the device (`torch.ones(1, device="mps")`, the comment of `testing/marks.py:113`
says "allocates a tensor on the device, which is slow and initializes a GPU context") and catches only
`RuntimeError`; `torch.backends.mps.is_available()` (already used in `random/torch.py:63`) answers without
allocating, and `torch.cuda.is_available()` is the CUDA equivalent. `get_available_devices` returns
`"cuda:0"` and `"mps:0"` even when there are several CUDA devices; document it or return one entry per
`torch.cuda.device_count()`.

### 43.5 `max_characters` counts the header and is exceeded by the ellipsis **[verified]** (P3)

`summary/engine.py:38-53` cuts the whole text of a node to `max_characters` and appends `...`:

```python
summarize("x" * 50, max_characters=10)  # "<class 'st..."      the value is never shown
summarize(np.arange(100), max_characters=20)  # "<class 'numpy.ndarra..."
summarize("x" * 50, max_characters=0)  # "..."
```

- The result has up to `max_characters + 3` characters, while `coola.formatting.text.truncate_str`
  (`max_len=10` gives 10 characters, `suffix` included) and its error ("must be greater than or equal to
  the suffix length") follow the other convention. Pick one for the library; `truncate_str` is the one
  users know (`textwrap.shorten`, `reprlib`).
- The limit is applied to the line that holds the class name, so a limit smaller than the header
  (`<class 'numpy.ndarray'> | shape=...`) hides the value and the metadata. Cut the value and keep the
  header.

None of this matters with the default of 1000, so it is P3, but `max_characters` is a public argument and
its contract ("at most N characters") is not met.

### 43.6 `bool(registry)` imports every optional library **[verified]** (P3)

`Registry.__len__` calls `_load_lazy_all()` (`registry/registry.py:124`), and a class without `__bool__`
uses `__len__` for its truth value:

```python
from coola.hashing import get_default_registry

r = get_default_registry()
# "torch" and "pandas" are not in sys.modules
bool(r)  # True ... and now they are (a second or more)
```

`if registry:`, `assert registry`, `registry or default` and a logging call such as `logger.debug("%s",
registry)` guarded by `if registry` all pay for the import of torch, pandas, polars, pyarrow, JAX, xarray.
`__repr__` has taken care of this ("Displaying a registry must not import the optional libraries"), the
truth value has not. Define `__bool__` as `bool(self._state) or bool(getattr(self, "_lazy", ()))` (a
registry with a pending group is not empty; the lazy groups of `TypeRegistry` are in `_lazy`), and add
`tests/integration/test_lazy_imports.py::test_bool_of_a_registry_does_not_import_the_libraries`.
`__len__`, `__iter__` and `keys()` must keep loading: they list the entries.

### 43.7 Small behaviours of `utils` **[verified]** (P3)

- `temp_env_vars({"X": None})` sets `X` to the string `"None"` (`utils/env_vars.py:182`, `str(value)`).
  `None` is the natural way to say "unset it for the block"; either implement it (restore afterwards) or
  raise a `TypeError` for a value that is not a `str`.
- `get_file_size(path)` follows a symbolic link when the argument is the link (`1000` for a link to a file
  of 1000 bytes) and ignores the links when it walks a directory (a directory with the file and two
  links is `1000`). Document it in one line or make the two cases agree (`lstat` for both).
- `sort_by_keys` falls back to `(type name, key)` and then to `repr`; `sort_by_values` has no fallback
  and raises a `TypeError` for `{"a": 1, "b": "x"}`. Reuse the same helper (43.1).
- `formatting.quantity.str_human_byte_size` divides by 1024 and writes `KB`/`MB`/`GB`/`TB` (an IEC unit is
  `KiB`), and stops at `TB`: `str_human_byte_size(2**50)` is `'1,024.00 TB'`. Add `PB`/`EB`, or write
  `KiB`. (Round 6 mentions the cap, 37.5; the label is new.)

## 44. Equality: semantics and diagnostics

### 44.1 The identity shortcut makes `equal_nan=False` depend on the objects **[verified]** (P3, design)

The engine tests `actual is expected` first (`equality/engine.py:189`). It is documented ("NaN is not
equal to NaN, unless `equal_nan=True` or both are the same object"), and it makes `objects_are_equal(x,
x)` reflexive. The consequence is a trap for a test suite:

```python
import math, numpy as np
from coola.equality import objects_are_equal as eq

eq(float("nan"), float("nan"))  # False
eq(np.nan, np.nan)  # True   (the same object)
eq({"a": np.nan}, {"a": np.nan})  # True
eq([math.nan], [math.nan])  # True
eq({"a": float("nan")}, {"a": float("nan")})  # False
a = np.array([np.nan])
eq(a, a), eq(a, a.copy())  # (True, False)
```

An expected value written with `np.nan` or `math.nan` (the usual way) passes a comparison that a value
computed at run time fails, so a test that "works" can start to fail when the value is computed. Python
has the same quirk for `[nan] == [nan]`, but `coola` is the library that promises to be precise about
NaN. Options: a keyword `identity: bool = True` (and `False` for the strict mode), or, at least, a
warning in the user guide next to the NaN row that names `np.nan`/`math.nan` and arrays. The property
tests (`test_equality.py`) generate fresh NaNs, so they cannot see it.

### 44.2 The messages of the table testers do not say what differs **[verified]** (P2, follow-up of 35.1)

35.1 asked for the cause, the row and the value. The column is now named, but:

```text
pandas  index values / index name / column order / dtype int vs float / dtype int32 vs int64 /
        columns name / tz / datetime unit / categories / MultiIndex   ->  "tables have a different structure"
pandas  series name / series dtype                                    ->  "series have a different structure"
polars  column order / dtype / number of rows                         ->  "tables have a different structure"
pyarrow column type / nullable flag                                   ->  "tables have a different structure"
all     a value in a column of 1000 rows   ->  "tables have different values: column 'a' differs"   path=()
```

while the array and xarray testers say `values are different: actual: 2 expected: 3` with
`path=('variable', 'data', 1)`, and the mapping tester lists the missing and the additional keys.
Proposal, in the order of the cost:

1. Split "different structure" into the part that differs: `index values differ`, `index names differ:
['i'] vs [None]`, `columns differ: ['a', 'b'] vs ['b', 'a']`, `dtypes differ: a: int64 vs float64`,
   `shapes differ: (3, 2) vs (2, 2)`. The code already computes each of them (`_same_index`,
   `_same_dtypes`, `_same_label`); they are folded into a boolean.
2. For a value difference, report the first different row and the two values, and set the `path` to
   `(row_label, column)`: `values are different at row 999 of column 'a': 999 vs 0`. For pandas it is
   `np.flatnonzero(~equal_mask)[0]`, for polars `(a != b).arg_true().head(1)`, for pyarrow
   `pc.not_equal(a, b)`.
3. `path` of the xarray testers leaks the internals (`('variable', 'data', 1)`,
   `('coords.variables', 'x', 'data', 1)`); map it to `('data', 1)` and `('coords', 'x', 1)`.
4. `NaN` as a reason: `numbers are different: actual: nan, expected: nan` should add
   `(NaN is not equal to NaN, use equal_nan=True)`; it is the question that comes first.

`pl.LazyFrame` raises `TypeError: '==' comparison not supported for LazyFrame` since #2681 removed its
support from `coola.equality`; the message could say "collect it first" and `summary` still exports
`PolarsLazyFrameSummarizer`, the one place where a lazy frame is supported.

### 44.3 `max_depth=100` is the only depth limit of the library **[verified]** (P3, carried over 28.3)

```python
def deep(n):
    x = [1]
    for _ in range(n):
        x = [x]
    return x


# n = 150 and n = 10_000
# equality:  MaxDepthExceededError
# hash_object, recursive_apply, to_jsonable, summarize(max_depth=3):  ok
```

The engine has an explicit stack (there is no recursion limit to protect), `hash_object` accepts a depth
of 100 000 (`tests/integration/hashing/test_interface.py`) and `test_engine_compares_100_000_nested_dicts`
passes `max_depth` explicitly. A default of `100` therefore only makes a deep but valid structure fail.
Make the default `None` (no limit), keep the keyword for users who want a guard, and document that
`summary.max_depth` is a display depth, not a limit.

## 45. Architecture, API and features

### 45.1 `equality` imports `io`: a package cycle hidden by a local import **[verified]** (P3)

The package-level import graph (computed from the AST, `TYPE_CHECKING` blocks excluded) has one cycle:

```text
equality -> display, io, registry, utils, validation
io       -> display, equality, factory, utils
```

`equality/tester/interface.py:221-223` imports `coola.io.base` inside `_get_io_equality_testers` (`# noqa:
PLC0415`, "`coola.io` depends on `coola.equality`, so it cannot be imported when this module is imported")
to register `BaseLoader` and `BaseSaver`. The low-level feature depends on a leaf feature to special-case
two of its classes, and no `import-linter` contract covers it (the five contracts of `pyproject.toml`
forbid the core packages to import the feature packages, not the features to import each other). The
simplest way out is not to special-case: `EqualNanEqualityTester` is the generic
"`equal(other, equal_nan=...)`" protocol, so a fallback
`register_fallback(lambda t: supports_methods(t, "equal"), EqualNanEqualityTester())` covers `BaseLoader`,
`BaseSaver` and every user class that follows the protocol, with no import (check first that it does not
change the result for a class that has `equal` and a more specific tester: the fallbacks come after the
MRO lookup, so it does not). The alternative is to let `coola.io` register its two classes when it is
imported.

Then add a contract "`equality`, `hashing`, `summary`, `recursive`, `iterator` must not import `io`" to
`pyproject.toml`.

### 45.2 29 of the 74 types of the support matrix cannot be converted to JSON **[verified]** (P2)

`tests/integration/test_support_matrix.py` compares `equality`, `hashing` and `summary` over the same
types and lists the known gaps of hashing (`HASH_GAPS`, 21 types). `to_jsonable`, `round_values`,
`recursive_apply` and `dfs_iterate` are not in it. Running the same factories through them (scratch
script, rows are the types, `ok` means a result):

```text
type                 eq  allclose  hash   summarize  to_jsonable   round   recursive_apply  dfs
complex              ok  ok        ok     ok         TypeError     ok      ok               ok
Fraction             ok  ok        ok     ok         TypeError     ok      ok               ok
datetime.timezone    ok  ok        ok     ok         TypeError     ok      ok               ok
re.Pattern           ok  ok        TypeError  ok     TypeError     ok      ok               ok
slice, Ellipsis, dict_keys/values/items, array.array     hash TypeError, to_jsonable TypeError
np.datetime64, np.timedelta64, np.void                   hash TypeError, to_jsonable ok
pd.DataFrame, pd.Categorical, pd.Interval, pd.Period     to_jsonable TypeError  (Series, Index, Timestamp ok)
pl.Series, pl.DataFrame, pa.Array, pa.Table, pa.Schema, pa.DataType, pa.Scalar   to_jsonable TypeError
jax.Array, xr.DataArray, xr.Dataset, xr.Variable         to_jsonable TypeError  (NumPy and torch ok)
torch.dtype, torch.device                                hash TypeError, to_jsonable TypeError
torch.meta                                               hash NotImplementedError, to_jsonable NotImplementedError
torch.sparse                                             to_jsonable RuntimeError("Cannot access data pointer of Tensor that doesn't have storage")
```

Three things stand out:

1. **The JSON gaps are not listed anywhere** (the user guide of `jsonable` lists the supported types, not
   the missing ones), and three of them are libraries the project supports everywhere else: JAX arrays
   (NumPy and torch are supported), polars/pyarrow and xarray. A JAX array is one line
   (`np.asarray(x).tolist()` through the NumPy transformer), a polars/pyarrow/xarray object is
   `to_dict()`/`to_pylist()`/`to_dict(data="list")`. Add the transformers, or add the types to a
   `JSON_GAPS` set of the matrix so that the gap is a decision.
2. **A raw library error leaks** for the layouts without data: `RuntimeError` (sparse), `NotImplementedError`
   (meta), `TypeError` from NumPy for a JAX PRNG key (`hash_object`). The registry convention is a
   `TypeNotRegisteredError`/`TypeError` that names the type; wrap these three in a `TypeError("Cannot
convert a sparse tensor to JSON: ...")` (and the same in `hashing/torch.py` for meta).
3. **Extend the matrix**, it is a 40-line change and the best regression net the project has: add columns
   for `allclose`, `to_jsonable`, `round_values`, `recursive_apply` and `dfs_iterate`, each with a
   `*_GAPS` set. Generate the table of "supported types by function" of the user guide from it, so that
   the docs cannot drift from the registries.

`hash_object` of a pandas `Index`, `MultiIndex`, `DatetimeIndex`, `RangeIndex` and `Categorical`, of a
SciPy sparse matrix and of the `pyarrow` scalars is also missing (34.3 lists the first ones); `pd.Index`
is the most likely one to be found in a dict next to a `DataFrame`.

### 45.3 `recursive_apply`, `dfs_iterate` and everything built on them stop at a pydantic model **[verified]** (P3)

```python
from pydantic import BaseModel


class M(BaseModel):
    a: int = 1
    b: list[int] = [1, 2]


recursive_apply(M(), lambda v: v + 1 if isinstance(v, int) else v)  # M(a=1, b=[1, 2]): unchanged
list(dfs_iterate(M()))  # [M(a=1, b=[1, 2])]: one leaf
```

A dataclass is entered (`DataclassTransformer`, `DataclassChildFinder`), a pydantic model is not, while
`equality`, `hashing`, `summary`, `jsonable` and `display` all have a pydantic entry. So
`round_values(model, recursive=True)` and `to_flat_dict(model)` treat a model as one opaque value, and
`recursive_apply` returns it unchanged. Either add a `PydanticModelTransformer`/`PydanticChildFinder`
(read `model_fields` and the extra fields, rebuild with `model_copy(update=...)` so that the validators
are not run again), or state the rule in the docs of `recursive` and `iterator`.

### 45.4 Small gaps in the features **[verified]** (P3)

- `save_json(obj, path)` writes `NaN` and `Infinity` (`json.dump` default `allow_nan=True`), which is not
  JSON: other parsers reject the file. `to_jsonable` returns the `nan` as it is, so `save_json(to_jsonable(x))`
  does the same. Make `allow_nan=False` the default of `JsonSaver` (the keyword can restore it), or map the
  non-finite floats in `to_jsonable` with an option (37.1 of round 6 asks for the option).
- `save_json` does not call `to_jsonable`: `save_json({"a": np.arange(2)}, p)` raises
  `TypeError: Object of type ndarray is not JSON serializable`. A keyword `jsonable=True` (or a
  `JsonableSaver`) would connect the two packages that the docs present together.
- `coola.io` has no `load(path)` / `save(obj, path)` that picks the loader by suffix (`.json`, `.pkl`,
  `.pt`, `.txt`) and no loader or saver for `.npy`/`.npz`, `.parquet` or `.csv`, while the library hashes
  and compares those objects.
- `coola.random` has `get_torch_generator(seed)` and no `get_numpy_generator(seed)`
  (`numpy.random.default_rng(get_random_seed(seed))` is the recipe); a one-liner for symmetry.
- `import_object("module.Class.attribute")` still fails (36.3 of round 6); `pkgutil.resolve_name` is the
  one-line fix.

## 46. Performance

### 46.1 Follow-up of 38.1 and 38.2 **[verified]** (P3)

- 38.1 is unchanged: a chain of shared lists takes `2^n` time in the engine
  (`n=10`: 2 ms, `n=16`: 117 ms, `n=20`: 1.9 s). The engine already keeps `visiting` for the cycles; add
  a `done` set of `(id(a), id(e))` for the pairs that were **proved equal** when the run is not
  explaining (keep the objects alive so that no id is reused: they are alive in the structures being
  compared). `recursive_apply` calls `func` once per reference (`[sh, sh] * 2` calls it 4 times for one
  shared leaf) and does not keep the sharing in the result (`result[0] is result[1]` is `False`); a memo
  by `id` fixes both and is what `copy.deepcopy` does.
- 38.2: the number of round 6 (80x) compared lists that share their float objects, where `==` returns on
  identity (`list(fl) == fl` is 0.9 ms). With distinct floats the real figures are `==` 7.3 ms,
  `objects_are_equal` 187 ms (**25x**) and `objects_are_allclose` 185 ms for a list of 1e6, against 6.4x for
  distinct ints (47.6 ms). The slow path is the per-element dispatch; `is_exact_scalar_sequence` already
  has the fast path for ints and strings, the floats need the NaN rule only (`a == b` is enough when
  `equal_nan` is False and the tolerances are zero, and `array("d", a) == array("d", b)` handles the
  tolerance case in C for the `allclose`). Add the floats and a benchmark with distinct floats.

### 46.2 Throughput of `hash_object` on containers **[verified]** (P3)

```text
list of 1e6 ints      719 ms       dict of 1e6 int items   2442 ms      list of 1e5 small dicts   1307 ms
list of 1e6 floats    730 ms       set of 1e5 ints          73 ms       np.ndarray 1e7 float64     125 ms
```

0.7 us per element is acceptable for a library that is not a hashing library, and the arrays and tables
are fast (`DataFrame` 1e6 x 5: 54 ms). A homogeneous list of ints or floats and a dict with `str`/`int`
keys can use the same approach as the arrays (pack with `struct`/`array.array`, one digest call), which
should bring 719 ms down by an order of magnitude. Worth doing only with the hash-format change of 34.3/43.2, since the digest
of a list would change.

## 47. Conventions of `CLAUDE.md` in `src` **[verified]** (P3)

- `raise ValueError` for a range, a finite check or a membership check, where `coola.validation` has the
  validator: `formatting/quantity.py:54` (`make_bar`, `validate_range(value, 0.0, 1.0)`), `:94`
  (`str_time_human`, `validate_finite`), `:142` (`str_human_byte_size`, `validate_in(unit, BYTE_UNITS)`),
  `reducer/base.py:402` (`quantile`, `validate_range(q, 0, 1)`), `hashing/pydantic.py:121`
  (`on_secret`, `validate_in`), `utils/bloom_filter.py:70` (`fp_rate`, `validate_range(..., low_inclusive=False,
high_inclusive=False)`). Several of the messages would change to the validators' wording, so check
  the tests that match them.
- `utils/bloom_filter.py:51` has a docstring on `__init__` (the rule says the arguments go in the class
  docstring).
- `functools.cache` in `utils/tensor.py` (43.4).
- Everything else that was checked mechanically is in order: every non-dunder function has a docstring
  unless it overrides one (the 8 exceptions are the `PEP 562` module hooks, `CaptureHandler.emit` and
  the three `@overload`s of `resolve_object`), no `type: ignore`, 31 `noqa` for 8 rules, every
  `pytest.raises` has a `match`, no import inside a test, no `eval`/`exec`/`shell=True`.

## 48. Tests (carried over, numbers re-measured)

- **39.2** is open: `tests/properties/test_library_types.py:61,75,90` have `xfail(strict=False)`; the
  run shows `2 xfailed, 1 xpassed`, so one marker is already stale and `xfail_strict = true` in
  `pyproject.toml` is overridden by the explicit `strict=False`.
- **39.3**: 103 tests with more than one `assert`/`raises` (largest groups: `utils` 18, `factory` 11,
  `hashing` 11, `registry` 11), 82 section headers that are not the three-line format, 20 tests with a
  loop or a branch. No `raises` without `match`, no local import.
- **39.4**: 14 source modules without a mirrored test file (`display/rich/print_.py`,
  `equality/tester/_array.py`, `equality/tester/base.py`, `equality/tester/object.py`,
  `factory/constants.py`, `hashing/{jax,numpy,pandas,polars,pyarrow,torch}.py`, `iterator/base.py`,
  `recursive/base.py`, `summary/object_fields.py`) and 157 functions whose name appears in no test.
  Coverage is 100% because the public functions exercise them, and `CLAUDE.md` asks for their own
  tests; the list is in the appendix script of round 6 (still valid).
- **New: add the checks to CI** (the three scripts of round 6 and the import-graph script of 45.1 are
  each under 40 lines); a rule that is not checked drifts, and the counts above did not move in the 38
  commits since round 6.

## 49. Packaging, CI and documentation

- No `Typing :: Typed` classifier although `py.typed` ships (and `ty`/`pyright` are in the CI).
- Upper bounds on optional dependencies: `numpy<3.0`, `pandas<4.0`, `polars<3.0`, `torch<3.0`,
  `pydantic<3.0`, `rich<16.0`, `colorlog<7.0`. For a library that adds no dependency, a cap makes the
  resolver fail (or downgrade the _user's_ pandas) the day a major version is released, before `coola`
  can test it; the nightly workflows that already test the new versions are the safety net. Cap only
  what is known to break.
- `Development Status :: 5 - Production/Stable`, the license classifier next to the SPDX `license`, and
  the stale `CITATION.cff` (title "COOLA: A Python library to check if two complex/nested objects are
  equal or not.", `date-released: 2023-01-01`, no version) are unchanged from 40.2.
- `examples/` has four scripts (equality, hashing, nested, summary) for 19 packages; `ci-examples.yaml`
  runs them nightly, so adding `jsonable.py`, `rounding.py`, `identifier.py` and `io.py` (10 lines each)
  costs nothing and tests the documentation.
- `docs/docs/dev/*.md` no longer describes `DataclassMarker` (only the migration guide names it, as it
  should); the duplication between `review/<package>.md` and `docs/docs/dev/<package>.md` was not
  re-checked. `review/` is 22 files and 11.7k lines in the repository: decide which are still
  design documents (move them to `docs/adr`) and which are finished plans (delete: git keeps them).
- The CI is complete (unit, integration, minimal install, doctest, examples, benchmarks, import cycles,
  dependency matrices, security audit, scorecard, pinned actions). The only gap in what it protects is
  45.1 (cycles between _features_ are not checked, `inv imports.check-cycles` imports each package in
  isolation).

## 50. What was checked and found correct

So that the next round does not redo it.

- **Equality / hashing / allclose contract** (the fuzz of the method section): no violation except 34.1.
  The `hash == hash` collisions of unequal objects are all by design: NumPy scalar and builtin, `int8` and
  `uint8`, `Timestamp` and `datetime`, a masked array that differs only under its mask, a `dict` and a
  `defaultdict`, a `pa.Array` and a one-chunk `ChunkedArray`, a `Parameter` and a tensor.
- **Identifiers**: Crockford encode/decode, the mod-37 checksum (the 5 extra symbols are never emitted),
  the UUIDv7 bit layout (12 + 62 random bits around the version and variant), the monotonic counter of
  `TickSequence` (restarts per tick, borrows the next tick at exhaustion), the fork guard of
  `SnowflakeIdGenerator`.
- **`jsonable`**: key collisions raise (`{1: "a", "1": "b"}`), cycles raise, big integers and `uint64` are
  exact, sets are sorted (also when mixed), `NaT`/`NA` give `None`.
- **`io`**: the temporary file + `os.link` commit, `exist_ok`, a symbolic link target, a directory as
  target, `fsync`, the `weights_only=True` default of `TorchLoader`.
- **`factory`**: the allow-list is checked on the path before the import, an allow-listed `_target_` with
  a nested configuration is refused, the security paragraphs are in every docstring.
- **`random`**: `validate_seed` before any manager is seeded, the state of every manager is restored
  even if one fails (`ExceptionGroup`), NumPy seeds above 2^32 use the two-word form.
- **`rounding`**: the rule is the same for the five libraries (checked on `1.1`, `0.3`, `2.675`, `1e300`,
  halfway cases, `-0.0`, NaN, `bfloat16`, `float16`, nullable pandas, polars nulls, pyarrow chunks).
- **`reducer`**: the three reducers agree on `mean`, `median`, `std`, `quantile`, `sort` and on the
  empty/NaN/invalid inputs (the differences are 37.3 and `[1e308, 1e308]`: native `1e308`, NumPy and torch
  `inf`).
- **`recursive` / `iterator`**: namedtuple, subclasses of `list` and `dict` with attributes,
  `defaultdict`, `Counter`, `deque(maxlen)`, frozen/slots/`init=False` dataclasses, `mappingproxy`,
  `dict_values`; an unbuildable subclass raises an actionable `TypeError`.
- **Import cost**: `import coola.equality` is 192 modules and 50 ms and imports none of the optional
  libraries; `coola.testing` is 58 modules.
- **Scale**: `objects_are_equal` is 6x `==` for a list of 1e6 ints, 17x for a dict of 1e6 items, 25x for
  floats (46.1) and 54x for 1e5 small nested dicts (0.05 to 0.4 s per million elements, which is the
  cost of a Python-level engine); 0.4x to 3.5x of `numpy.array_equal`/`DataFrame.equals` for arrays and
  frames. `summarize` of a dict that holds a list of 5e6 items takes 6 ms, a 50 MB string 9 ms.
- **Security**: `pickle` is used only by the explicit `PickleLoader` (documented, `# noqa: S301`),
  `subprocess` only by `run_git_command` with a resolved executable, no `eval`/`exec`, no `shell=True`.

## 51. Status of round 6

Checked on `main` at `8055c876` by running the repro of each item.

| Round-6 ID | Status        | Notes                                                                                                                                                                                       |
| ---------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 39.1       | **Open** (P1) | `test_real_mark_is_importable_in_a_generated_module` still fails (`1 failed` in the baseline). The generated module is still rewritten by `docformatter`.                                   |
| 34.1       | **Open**      | `objects_are_equal(a, b)` is `True` and `hash_object(a) == hash_object(b)` is `False` for categories in another order (the only violation of the fuzz).                                     |
| 34.2       | **Open**      | `hash_object(np.timedelta64(1, "s"))` raises `TypeError: int() argument must be ... not 'datetime.timedelta'`; `np.datetime64` raises `TypeNotRegisteredError`.                             |
| 34.3       | **Open**      | 21 types in `HASH_GAPS`, plus the SciPy sparse matrices (45.2).                                                                                                                             |
| 35.1       | **Partly**    | The column is named in the reason; the structure, the row and the value are not (44.2).                                                                                                     |
| 35.2       | **Open**      | `objects_are_allclose` is `False` for list/struct columns of polars, list columns of pyarrow, `object` columns of pandas, and for `{1.0}` against `{1.0000001}`.                            |
| 35.3       | **Open**      | `compare` defaults to `rtol=0, atol=0`, `objects_are_allclose` to `1e-5`/`1e-8`.                                                                                                            |
| 36.1       | **Open**      | `manual_seed(np.int64(3))` raises `TypeError: seed must be an int, got int64`; `validate_number` accepts it, `validate_int` does not (`numbers.Integral` would).                            |
| 36.2       | **Open**      | `dir(coola.io)` lists `Any`, `TYPE_CHECKING`, `annotations`, `lazy_exports`; `coola.formatting` exports only the submodule `structure`.                                                     |
| 36.3       | **Open**      | `import_object("collections.OrderedDict.fromkeys")` raises `ImportError`.                                                                                                                   |
| 36.4       | **Open**      | 12 modules of `coola.utils` are used by nothing in `src`: `array`, `batching`, `bloom_filter`, `date`, `env_vars`, `file_size`, `git`, `logging`, `lru`, `password`, `recursion`, `timing`. |
| 37.1       | **Open**      | Non-finite floats, `longdouble` (the message is "would be converted without end"), data frames, sparse tensors: see 45.2.                                                                   |
| 37.2       | **Open**      | `to_flat_dict({"a.b": 1})` and `to_flat_dict({"a": {"b": 1}})` are both `{"a.b": 1}`; `escape=False` is the default.                                                                        |
| 37.3       | **Open**      | `NativeReducer().mean([inf, -inf])` raises `ValueError: -inf + inf in fsum`; NumPy and torch return `nan`.                                                                                  |
| 37.4       | Not re-run    | CUDA/MPS generator seeded inside a `random_seed` block (needs a GPU).                                                                                                                       |
| 37.5       | **Open**      | Byte sizes stop at TB (43.7), `truncate_str` raises for a short limit (43.5), the summary of sparse and nested tensors.                                                                     |
| 38.1       | **Open**      | `n=20` gives 1.9 s (46.1).                                                                                                                                                                  |
| 38.2       | **Open**      | Corrected figure 25x (46.1).                                                                                                                                                                |
| 39.2       | **Open**      | 2 xfailed + 1 xpassed, `strict=False` (48).                                                                                                                                                 |
| 39.3, 39.4 | **Open**      | Numbers unchanged (48).                                                                                                                                                                     |
| 39.5       | **Open**      | 64 `# pragma: no cover` in `src` in total (the import guards and the fallbacks).                                                                                                            |
| 40.1       | **Partly**    | `docs/docs/dev` no longer describes `DataclassMarker`; the duplication with `review/` was not re-checked (49).                                                                              |
| 40.2       | **Open**      | Classifier, CITATION, `all` extra (49).                                                                                                                                                     |

## 52. Suggested order

1. **Unblock `main`:** 39.1 (one small PR; the stack is easier to review once the suite is green), then
   39.2 (strict markers) so that 34.1 is an ordinary red test.
2. **Determinism and robustness, one PR each with the regression test that fails without the change:**
   43.1 (the sort fallback of `smallest_items`, two-subprocess test), 43.2 (sparse tensors: size guard
   first, then the sparse comparison), 43.4 (three lines), 43.6 (`__bool__`).
3. **Contract fixes carried over:** 34.1, 34.2, 36.1, 37.3, 37.2.
4. **Diagnostics:** 44.2 (structure message first, row and value second, xarray path third), then 35.3
   and 44.1 (decide the identity question before documenting more of the NaN behaviour).
5. **Coverage of the type-dispatched modules (one PR):** extend the support matrix (45.2 point 3), then
   fill the gaps it shows: JSON for JAX/polars/pyarrow/xarray/`Fraction`/complex, the three wrapped errors,
   the rounders of 43.3, the pydantic entry of 45.3, `hash_object` of the indexes and the sparse
   matrices. Do the hash additions in the same release as the sparse-tensor format (43.2) so that the
   digest changes once.
6. **Architecture:** 45.1 (the registration moves to `coola.io`, then the import-linter contract).
7. **Performance:** 46.1 (float fast path, five lines; then the memo of the engine and of
   `recursive_apply`), 46.2 with the hash-format change.
8. **Maintenance:** 47, 48 (turn the scripts into CI checks), 49, and the decision of 36.4 (12 unused
   modules in the public `coola.utils`) in the 2.0 window.
