# Codebase review: suggestions (round 5)

This file **replaces the previous `review/suggestions.md`** (round 4, items 12.1 to 19.3). That
version is in git history (last touched by `e2546e4d`). Section 31 gives the status of its items.
Nothing below repeats an item of round 4, `src-coola-review.md`, `noqa-review.md` or
`python-311-modernization.md`, except where an item is explicitly marked _carried over_.

The numbering **continues** the previous rounds (sections 22 to 32), so that an ID such as `22.1` in a
branch name or a commit message is unambiguous.

## How this review was done

Checkout of `main` at `eea4aeab` (`.venv`, Python 3.14.8, NumPy 2.5.4, pandas 3.0.6, torch, polars,
pyarrow, xarray, JAX, pydantic installed). Baseline: `pytest tests/unit tests/integration` gives
12740 passed, 40 skipped, 30 xfailed in 25 s with 99% line coverage (6395 statements, 1 missed, 1
partial branch); `pytest tests/properties` gives 527 passed, 2 xfailed, 2 xpassed in 42 s;
`ruff check`, `ty check`, `lint-imports`, `xdoctest src/coola` (525 doctests) and
`mkdocs build --strict` are clean. The review changed no source file.

1. **Read-through.** Every module of `src/coola` (237 files, 31.6k lines) was read with the docstrings
   stripped, package by package: `registry`, `equality` (handlers and testers), `hashing`, `summary`,
   `identifier`, `utils`, `io`, `iterator`, `recursive`, `nested`, `reducer`, `random`, `display`,
   `factory`, `validation`, `testing`. The tests (323 files, 65k lines), the docs, `pyproject.toml` and
   the CI workflows were surveyed with scripts (structure, conventions, markers), not read line by
   line.
2. **Probes.** Items marked **[verified]** were reproduced on this checkout with a script and the
   repro is given. Items marked **[reading]** or **[lead]** come from reading the code and were not
   reproduced (GPU, Windows and CI-only behaviour were not run).
3. **Round 4 follow-up.** The items of round 4 that were merged (PRs #2238 to #2270) were
   re-checked by running their repros; the ones that are still open are carried over in section 31.

Almost all round-4 items are merged. The new findings are in four places: the **hashing of pandas
objects with missing values** (section 22), the **process-level behaviour of the global registries**
(fork, test isolation, plugins, section 24), the **traversal helpers** that do not follow the
protections of `equality` and `hashing` (section 25), and **property tests whose `xfail` markers
hide the state of the defects** (section 29).

Priority: **P1** wrong result or crash on realistic input, **P2** inconsistent or surprising
behaviour, **P3** cleanup or design.

## Summary

| ID   | P   | Area      | Finding                                                                                       |
| ---- | --- | --------- | --------------------------------------------------------------------------------------------- |
| 22.1 | P1  | hashing   | pandas objects with a missing value in `boolean`, `string`, tz-aware, period, interval dtypes |
| 22.2 | P2  | hashing   | Equal bfloat16 / float8 arrays (`ml_dtypes`, JAX) hash differently (`-0.0`, NaN)              |
| 22.3 | P2  | hashing   | pyarrow arrays are hashed through Python objects: 200x slower than NumPy                      |
| 22.4 | P2  | hashing   | "Stable" IDs depend on the mutable default registry and on installed plugins                  |
| 23.1 | P2  | equality  | Equal pydantic models are different after a `cached_property` was read on one of them         |
| 24.1 | P2  | registry  | A registry lock held by another thread at `fork()` deadlocks the child                        |
| 24.2 | P2  | registry  | No way to copy, scope or reset a default registry: the tests of a user leak state             |
| 25.1 | P2  | recursive | `recursive_apply` has no cycle detection; deep structures give a raw `RecursionError`         |
| 26.1 | P2  | utils     | `to_jsonable` of a dataclass deep-copies it (fails on a lock, copies big arrays)              |
| 28.1 | P2  | api       | `import coola` exposes nothing: the three main functions live in three subpackages            |
| 29.1 | P2  | tests     | Stale non-strict `xfail` markers: a test of a fixed defect still "xfails" for another reason  |
| 22.5 | P3  | hashing   | The hash of an integer with more than 4300 digits depends on `sys.set_int_max_str_digits`     |
| 22.6 | P3  | hashing   | A `str` with a lone surrogate (`os.fsdecode` of a bad file name) cannot be hashed             |
| 22.7 | P3  | hashing   | Equal aware `datetime.time` objects hash differently                                          |
| 22.8 | P3  | hashing   | Merkle hashing costs 10 µs for a three-key dict (format v2)                                   |
| 23.2 | P3  | equality  | `set` and `frozenset` are compared with `==`: `equal_nan` and the tolerance never apply       |
| 23.3 | P3  | equality  | Plain objects, exceptions and `functools.partial` are different even if all attributes match  |
| 24.3 | P3  | registry  | Entry-point plugins cannot be disabled or listed                                              |
| 25.2 | P3  | summary   | Set previews are not deterministic; `RecordBatch` and the mask of masked arrays are missing   |
| 25.3 | P3  | traversal | `recursive_apply`, `dfs_iterate`, `bfs_iterate` and `nested` do not enter dataclasses         |
| 26.2 | P3  | nested    | `flatten_mapping` raises a bare `AttributeError`; `sort_by_keys` fallback can still raise     |
| 27.1 | P3  | factory   | Nested `_target_` configurations are not resolved, and the docs do not say so                 |
| 27.2 | P3  | factory   | `is_object_config` imports the module of an untrusted `_target_`                              |
| 28.2 | P3  | api       | `summarize`, `recursive_apply`, `dfs_iterate` take options positionally                       |
| 28.3 | P3  | api       | `max_depth` means "raise above" in `equality` and "truncate at" in `summary`                  |
| 28.4 | P3  | docs      | The README quick start shows `show_difference` log output and never mentions `compare`        |
| 29.2 | P3  | tests     | Test layout and convention slips (mirror rule, multiple asserts, one `sleep`)                 |
| 30.1 | P3  | packaging | `CITATION.cff` is out of date; classifiers, changelog and README still open (19.1 to 19.3)    |

## 22. `hashing`

### 22.1 pandas objects with a missing value cannot be hashed in several dtypes **[verified]** (P1)

`PandasSeriesHasher` and `PandasDataFrameHasher` hash `column.to_numpy()`. For an extension dtype
with a missing value, or a tz-aware datetime with `NaT`, that is an object array that contains
`pd.NA` or `pd.NaT`, and neither has a hasher:

```python
import pandas as pd
from coola.hashing import hash_object

hash_object(pd.Series([True, None], dtype="boolean"))  # TypeNotRegisteredError: NAType
hash_object(pd.Series(["a", None], dtype="string"))  # TypeNotRegisteredError: NAType
hash_object(pd.Series(["a", None], dtype="string[pyarrow]"))  # TypeNotRegisteredError: NAType
hash_object(
    pd.Series(pd.to_datetime(["2020", None]).tz_localize("UTC"))
)  # ValueError: NaTType does not support utcoffset
hash_object(
    pd.Series(pd.period_range("2020", periods=2, freq="D"))
)  # TypeNotRegisteredError: Period
hash_object(
    pd.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2]))
)  # TypeNotRegisteredError: Interval
```

`Int64`, `Float64`, `category`, `timedelta64`, `Sparse` and the `int64[pyarrow]` dtype work, so the
failure depends on the dtype in a way users cannot guess. `objects_are_equal(a, a.copy(), equal_nan=True)`
is `True` for all of them. This is the realistic case of round 4's 14.4 (which listed
the scalars `pd.NA`, `pd.NaT`, `Period`, `Interval`), and it is the one that users hit: any data frame
read with `convert_dtypes()` or `dtype_backend="pyarrow"` and containing a missing value.

Fix, in `hashing/pandas.py`: for the columns that are not plain NumPy dtypes, hash
`column.to_numpy(dtype=object, na_value=None)` (verified: it gives `None` for `NA` and `NaT`, and
`Timestamp`, `Period` and `Interval` objects otherwise), and add the dtype string already hashed by
`_hash_dtype`. Register hashers for `pd.Period` (`freqstr` and ordinal) and `pd.Interval`
(`left`, `right`, `closed`) so that the objects of the arrays can be hashed, and route `NaTType`
before `DatetimeHasher` (`NaT` is a `datetime` subclass). Add the dtypes above to the support matrix
(`tests/integration/test_support_matrix.py`) and to `tests/properties/test_library_types.py`.

### 22.2 Equal bfloat16 and float8 arrays hash differently **[verified]** (P2)

`canonicalize_floats` (`hashing/numpy.py:88`) normalizes `-0.0` and the NaN bit pattern only for the
NumPy kinds `f` and `c`. The `ml_dtypes` types (`bfloat16`, `float8_*`, which JAX uses and NumPy can
hold) have the kind `V`:

```python
import ml_dtypes, numpy as np
from coola.equality import objects_are_equal
from coola.hashing import hash_object

z, nz = np.array([0.0], dtype=ml_dtypes.bfloat16), np.array([-0.0], dtype=ml_dtypes.bfloat16)
objects_are_equal(z, nz)  # True
hash_object(z) == hash_object(nz)  # False
# the same for the sign of a NaN, and for jnp.array([-0.0], dtype=jnp.bfloat16)
```

This is the 14.3 defect (round 4) for the dtypes that 14.3 did not reach. Fix: when
`dtype.kind == "V"` and `dtype.type.__module__.startswith("ml_dtypes")`, view the array as the
unsigned integer of the same size, and canonicalize with the integer pattern of `0.0` and of the
default NaN (or `astype(np.float32)` for the hash only, which is exact for these types).
The torch hasher has the same problem for float8 (`tensor + 0` raises, 14.5 of round 4): both can
share the "convert to float32 for the normalization, hash the original bytes" strategy.

### 22.3 pyarrow arrays are hashed through Python objects **[verified]** (P2)

`_hash_array` (`hashing/pyarrow.py:24`) calls `array.to_pylist()` and hashes the list item by item.

```python
import numpy as np, pyarrow as pa
from coola.hashing import hash_object

hash_object(pa.array(np.arange(10**6)))  # 1.67 s
hash_object(np.arange(10**6))  # 0.008 s
```

Fix: for the fixed-width types (`pa.types.is_integer`, `is_floating`, `is_boolean`, `is_temporal`,
`is_decimal`), hash the validity bitmap and the values buffer with the NumPy path
(`array.to_numpy(zero_copy_only=False)` after `fill_null(0)` is what the equality handler already
does in `_to_numpy`), and keep `to_pylist()` for the strings, lists and structs. The hash changes for
the types of the fast path, so do it with 22.4 (golden vectors first, then change the format once).

### 22.4 The "stable" IDs depend on global mutable state **[verified]** (P2)

`generate_stable_content_id` and `generate_stable_uuid5` call `hash_object(data, registry=None)`, i.e.
the **default registry**. That registry changes when (a) any code of the process calls
`register_hashers`, or (b) an installed distribution declares a `coola.hashing.hashers` entry point
(plugins are loaded the first time the registry is built, `utils/singleton.py:180`). A library that calls
`register_hashers({int: ...}, exist_ok=True)` changes the IDs of every other library in the process,
and the IDs differ between two environments that differ only by an installed package.

Suggestions, in this order:

1. Golden vectors (14.6, carried): `tests/unit/hashing/test_golden.py` with one fixed value per hasher
   family, so that a change of format is a deliberate edit.
2. A frozen registry for the stable IDs: `get_stable_registry()` (built-in hashers only, no plugin, no
   `register_hashers`) used by `generate_stable_content_id` and `generate_stable_uuid5` when
   `registry=None`. Document that passing another registry opts out of the stability.
3. A `HASH_FORMAT_VERSION` constant, mixed into the UUID5 name, so that the next format change
   (22.3, 22.8) can be requested with `version=1`.

### 22.5 The hash of a huge integer depends on an interpreter setting **[verified]** (P3)

`int_repr` (`utils/integer.py`) uses `int.__repr__` and falls back to hexadecimal only when Python
raises `ValueError` for an integer above `sys.get_int_max_str_digits()` (4300 by default). The same
integer therefore has two representations, and two hashes, depending on `PYTHONINTMAXSTRDIGITS` or
`sys.set_int_max_str_digits`:

```python
import sys
from coola.hashing import hash_object

sys.set_int_max_str_digits(0)
a = hash_object(10**5000)
sys.set_int_max_str_digits(4300)
b = hash_object(10**5000)
a != b  # True
```

Fix: decide by size, not by exception: `value.bit_length() > 14000` (about 4200 digits) is always
hexadecimal. The hash of an integer between 4300 and 14000 digits does not change for the default
setting.

### 22.6 A string with a lone surrogate cannot be hashed **[verified]** (P3)

`hash_string` raises `ValueError: Cannot hash a string that is not valid UTF-8` for `"a\udc80"`. Such
strings are produced by `os.fsdecode` / `os.listdir` for a file name that is not valid UTF-8, so
`hash_object(Path(name))`, `hash_object(os.listdir(...))` and every dict that has such a key raise.
Encode with `errors="surrogatepass"` (deterministic, and the result is unchanged for every string that
can be hashed today).

### 22.7 Equal aware `time` objects hash differently **[verified]** (P3)

`datetime.time(12, tzinfo=UTC) == datetime.time(13, tzinfo=timezone(timedelta(hours=1)))` is `True`
and the hashes are different: `DatetimeHasher` normalizes aware `datetime` objects to UTC but hashes
an aware `time` with its own `isoformat()`. Use the rule of `time.__hash__` (subtract the offset from
the time of day) for aware `time` objects.

### 22.8 Merkle hashing is 10 µs per small container **[lead]** (P3)

`hash_object([{"a": i, "b": 1.0, "c": "x"} for i in range(10**5)])` takes 1.0 s: each node is hashed
with its own `blake2b`, tagged, and the fixed-width digests of the children are concatenated and
hashed again. This is simple and unambiguous, and it makes `hash_object(child)` reusable, but a
streaming encoder (one `blake2b` object, a type tag and a length prefix per node) would do the same
work with one hash call per container. It is a format change, so it belongs with 22.3 and the format
version of 22.4. Not worth doing before the golden vectors exist.

### 22.9 Carried over from round 4

14.4 (hashers for the types that equality supports; the 27 `known gap` hash rows of the support matrix),
14.5 (meta and float8 tensors raise `NotImplementedError`) and 14.7 (`compare=False` fields; the
property test `test_equal_dataclasses_have_the_same_hash` still fails) are unchanged. 14.4 now
has a first priority list: pandas missing values (22.1), then `Index`/`MultiIndex`/`Categorical`,
`np.datetime64`/`np.timedelta64`, `array.array`, `dict_keys`/`dict_values`/`dict_items`,
`MappingProxyType`, `slice`, `Ellipsis`, `re.Pattern`, `torch.dtype`/`torch.device`.

## 23. `equality`

### 23.1 Equal pydantic models are different after a `cached_property` is read **[verified]** (P2)

`PydanticModelEqualityTester` (added in #2242) compares `__dict__` and `__pydantic_extra__`.
`functools.cached_property` stores its value in the instance `__dict__`, so a model whose cached
property was read is "different" from an equal model whose property was not:

```python
import functools
from pydantic import BaseModel
from coola.equality import compare, objects_are_equal
from coola.hashing import hash_object


class M(BaseModel):
    x: int

    @functools.cached_property
    def double(self) -> int:
        return self.x * 2


a, b = M(x=1), M(x=1)
a.double
a == b  # True (pydantic ignores the cache)
objects_are_equal(a, b)  # False: "objects have different lengths: 2 vs 1" at actual.__dict__
hash_object(a) == hash_object(b)  # True (the hasher reads model_fields)
```

Equality, hashing and the summarizer must agree on what a model is. Fix: compare
`{name: getattr(model, name) for name in type(model).model_fields}` (the fields the hasher and the
summarizer use) plus `model_extra`, instead of `__dict__`; a handler such as
`SameFieldsHandler(fields_getter)` keeps the report path (`actual.x`). Private attributes
(`__pydantic_private__`) can be an opt-in, as in pydantic's own `__eq__`. Add the regression test (two equal models, a
`cached_property` read on one of them).

### 23.2 Sets are compared with `==`, so `equal_nan` and the tolerance never apply **[verified]** (P3)

There is no tester for `set` and `frozenset` (the default tester calls `==`). Documented in
`equality.md:818`, but it is the one container with silent behaviour:

```python
objects_are_equal({1.0, float("nan")}, {1.0, float("nan")}, equal_nan=True)  # False
objects_are_allclose({1.0}, {1.0 + 1e-9})  # False
objects_are_equal({(float("nan"), 1)}, {(float("nan"), 1)}, equal_nan=True)  # False
```

Suggestion: a `SetEqualityTester` (same type, same length, then a matching): when every element is of
an exact scalar type (`is_exact_scalar_sequence` logic of 13.7), keep `==`; otherwise match each
element of `actual` to one unmatched element of `expected` through the registry (O(n²) worst case,
bucket by `type(x)` first). Report `sets differ: 1 element of actual has no match` (the element, not
the index, is the path). This closes the family left by 13.3 (Decimal, Fraction, timedelta are PR
#2244).

### 23.3 Objects without value semantics **[verified]** (P3, _carried over_ from 13.1)

`objects_are_equal(C(1), C(1))` for a plain class, `ValueError("a")` vs `ValueError("a")`, and two
`functools.partial(print, 1)` are `False` (identity `==`). That is correct for "equal means `==`", and
surprising next to dataclasses, pydantic models and `attrs`-like classes that are compared field by
field. Options: document the rule in the "Type-specific behavior" table (it says "compared with `==`
unless a tester exists"), and provide an opt-in `ObjectAttributesEqualityTester(attributes="vars")` (vars or
`__slots__`, with a `BaseException` variant comparing `args`) that users register for their classes.
`torch.nested` tensors still raise `RuntimeError` (13.6): compare `unbind()` of both.

## 24. `registry` and process-level behaviour

### 24.1 A registry lock held at `fork()` deadlocks the child **[verified]** (P2)

`BaseRegistry` and `LazySingleton` hold `threading` locks. If a thread of the parent is inside a
registry operation (registration, lazy loading, the first call that builds a default registry) when
the process forks, the child inherits a locked lock whose owner does not exist. The next use of the
registry in the child blocks forever. A typical case is a `torch.utils.data.DataLoader`
with `fork` workers, or a `multiprocessing` pool, created while another thread of the program is
using coola.

```python
# thread holds registry._state._lock, then os.fork(): the child calls registry.register(...)
# -> never returns (the repro is 30 lines; the child was killed by SIGALRM after 5 s)
```

`coola.identifier` already solves this for its generators (`coola.utils.fork`: `track_fork` registers
the object in a `WeakSet`, and `os.register_at_fork(after_in_child=...)` calls `_reset_after_fork`).
Fix: move `track_fork` to `coola.utils` (public, with tests), make `BaseRegistry`, `LazySingleton`,
`HasherRegistry._fallbacks_lock` and `_SAVE_LOCKS_GUARD` fork-aware (`_reset_after_fork` creates new
locks; the registry content is valid because the child has a snapshot of memory), and add the
integration test with `os.fork` (skipped on Windows). The `multiprocessing` `spawn` start method is
not affected.

### 24.2 A default registry can be changed but not copied, scoped or reset **[verified]** (P2)

Every default registry is a process-wide `LazySingleton`. The public API can mutate it
(`register_*`, `clear`, `unregister`) and nothing can restore it:

```python
from coola.equality.tester import get_default_registry

get_default_registry().clear()
objects_are_equal(
    np.ones(2), np.ones(2)
)  # TypeNotRegisteredError: ndarray, for the rest of the process
```

`clear()` also drops the pending lazy groups (`BaseRegistry.clear` clears `_lazy`), so the optional
libraries cannot be loaded again. `LazySingleton.reset()` exists but is reachable only through the
private `_default_registry` of each module. A user who registers a tester in one test (with
`exist_ok=True`, since the default registry may already hold it) leaks it to every other test of
the session.

Suggestions: `BaseRegistry.copy()` (shallow copy of state, lazy groups and fallbacks, new lock);
a context manager `registry.scoped()` / `temporary_registry()` that restores the state on exit; a
public `reset_default_registry()` per package (or one `coola.registry.reset_defaults()`); and a pytest
fixture in `coola.testing` that snapshots and restores the default registries around a test. The
`registry.md` guide gets a "Testing code that registers types" section.

### 24.3 Entry-point plugins cannot be disabled or listed **[reading]** (P3)

`load_registry_plugins` imports and calls every entry point of the group the first time a default
registry is built. Only the failure mode is configurable (`COOLA_STRICT_PLUGINS`). Add
`COOLA_DISABLE_PLUGINS=1` (reproducible runs, debugging a plugin conflict, 22.4), and a function that
lists the plugins that were loaded (`coola.utils.singleton.get_loaded_plugins()`), so that a surprise in
`objects_are_equal` or in a hash can be traced to a package without reading `importlib.metadata`.

## 25. `recursive`, `iterator`, `summary`

### 25.1 The traversal helpers do not have the protections of `equality` and `hashing` **[verified]** (P2)

`objects_are_equal` raises `MaxDepthExceededError` (with the fix to apply), `hash_object` wraps the
`RecursionError` in a `ValueError`, `dfs_iterate`/`bfs_iterate` detect cycles. Three entry points do none of it:

```python
cycle = []
cycle.append(cycle)
recursive_apply(
    cycle, lambda x: x
)  # RecursionError: maximum recursion depth exceeded (no cycle detection)
dfs_iterate(deep)  # RecursionError for a 3000-level structure (bfs_iterate is iterative and works)
summarize(deep, max_depth=5000)  # RecursionError
convert_to_jsonable(deep)  # RecursionError
```

Suggestions: `TransformerRegistry.transform` tracks the ids of the ancestors like `PathRegistry` does for the
iterators (cycle → `ValueError("Cycle detected...")`) and wraps the `RecursionError` in the same
message as `_compare`; `summarize` documents that `max_depth` is the display depth and relies on the
caller for the Python limit, or raises the same wrapped error; `dfs_iterate` could use an explicit
stack (the `IterableIterator` and `MappingIterator` of the registry are the only users of the
recursion). A shared helper `coola.utils.recursion.raise_if_too_deep(exc)` gives one message for all.

### 25.2 Summaries: set previews are random, some types fall back to `str()` **[verified]** (P3)

- `SetSummarizer` previews the first `max_items` of `iter(set)`. For strings the order depends on
  `PYTHONHASHSEED`: `summarize({"b", "a", "c"}, max_depth=2)` printed `a, c, b` here, and a log line
  or an `assert summarize(x) == "..."` differs from one run to the next. Sort when the elements can
  be sorted (`try_sorted_list` of `coola.nested` is the helper), keep the iteration order otherwise.
- `pa.RecordBatch` is registered by equality and hashing, not by summary: it falls to
  `DefaultSummarizer` and prints the Arrow `repr` (cut at 1000 characters) while `pa.Table` gets
  `shape | columns`. `RecordBatch` has `.shape` and `.schema`, so registering it with
  `PyarrowTableSummarizer` is one line. `np.ma.MaskedArray` goes through `NDArraySummarizer`, which
  omits the mask: `summarize(np.ma.masked_array([1, 2], mask=[0, 1]))` shows only the shape and the
  dtype, so two arrays that equality and hashing distinguish have the same summary.

### 25.3 Dataclasses and models are leaves for the traversal helpers **[reading]** (P3)

`recursive_apply`, `dfs_iterate`, `bfs_iterate`, `convert_to_jsonable` and the `nested` helpers treat
a dataclass or a pydantic model as a leaf, while `objects_are_equal`, `hash_object`, `summarize` (15.3)
and `to_jsonable` enter them. `recursive_apply(Config(...), func)` therefore calls `func` on the whole
object. The fallback mechanism of `HasherRegistry` (12.4) would work here too: a
`DataclassTransformer` (rebuild with `dataclasses.replace`) and a `DataclassIterator`, registered
through `register_fallback`, opt-in through a `coola.recursive.enter_dataclasses()` helper so that the
default behaviour does not change.

## 26. `utils` and `nested`

### 26.1 `to_jsonable` of a dataclass deep-copies it **[verified]** (P2)

`to_jsonable` calls `dataclasses.asdict(data)`, which deep-copies every field value that is not a
dataclass, list, tuple or dict:

```python
@dataclass
class HasLock:
    lock: object


to_jsonable(HasLock(threading.Lock()))  # TypeError: cannot pickle '_thread.lock' object
to_jsonable(HasLock(np.zeros(10**7)))  # copies 80 MB before converting the array to a list
```

Fix: `{f.name: getattr(data, f.name) for f in fields(data)}` (the pattern `DataclassHasher` and
`DataclassSummarizer` use) and convert the values with `_to_jsonable_recursive`. The behaviour for
nested dataclasses is the same.

### 26.2 Smaller input-validation slips **[verified]** (P3)

- `flatten_mapping({"a": 1})` raises `AttributeError: 'int' object has no attribute 'items'`. The
  signature says `Mapping[Any, Mapping[Any, Any]]`; raise a `TypeError` that names the outer key.
- `sort_by_keys({(1,): 1, ("a",): 2})` still raises `TypeError`: the fallback sorts by
  `(type name, key)`, which fails for two keys of the same type that cannot be compared. Fall back to
  `(type name, repr(key))` or keep the insertion order.
- `slugify("😀")` returns `""`, an empty file or URL component. The function already hashes non-ASCII
  text that has letters (`slugify("日本")`); the same rule can cover the text without letters.

## 27. `io` and `factory`

### 27.1 Nested `_target_` configurations are not resolved **[verified]** (P3)

```python
resolve_object({"_target_": "collections.Counter", "a": {"_target_": "builtins.int"}})
# Counter({'a': {'_target_': 'builtins.int'}})
```

Hydra users expect the nested configuration to be instantiated. The current behaviour is the safer
one (the allow-list applies to one `_target_` per call), but neither `factory.md` nor the
docstrings say so. Add one sentence and an example in the guide ("`_target_` is resolved at the top
level only; resolve the nested ones with `resolve_object` yourself"), or an opt-in
`recursive=True` that applies `allowed_prefixes` to every level.

### 27.2 `is_object_config` imports an untrusted module **[reading]** (P3, _carried over_ from 16.3)

#2258 made `resolve_object` check the target before calling it. `is_object_config`
(`factory/config.py:74`) still calls `import_object(target)`, which imports the module named by the
configuration just to answer a boolean (import-time side effects of an arbitrary module, or of a
module of `allowed_prefixes`). Without `allowed_prefixes`, check the type without importing: parse the
string, and answer `False` for the targets that are not in `sys.modules` unless the caller opts in
(`import_target=True`).

## 28. API and documentation

### 28.1 `import coola` exposes nothing **[verified]** (P2)

`coola.__init__` only provides `__version__`. The three functions that most users want are in three
subpackages (`coola.equality.objects_are_equal`, `coola.hashing.hash_object`,
`coola.summary.summarize`), and `coola.objects_are_equal` is an `AttributeError`. With
`lazy_exports` (already used in every `__init__`) a flat namespace costs no import time:

```python
# coola/__init__.py
_LAZY_EXPORTS = {
    "objects_are_equal": "coola.equality",
    "objects_are_allclose": "coola.equality",
    "assert_objects_equal": "coola.equality",
    "assert_objects_allclose": "coola.equality",
    "compare": "coola.equality",
    "hash_object": "coola.hashing",
    "summarize": "coola.summary",
}
```

Keep the subpackage paths as the documented ones for everything else. The import-time benchmark
(`ci-benchmark-import.yaml`) guards the cost.

### 28.2 Options are positional in three entry points **[verified]** (P3)

`CLAUDE.md` asks for keyword-only options (`hash_object` and `objects_are_equal` are). Three others
are not: `summarize(data, max_depth=1, registry=None)`, `recursive_apply(data, func, registry=None)`
and `dfs_iterate`/`bfs_iterate(data, registry=None)`. `summarize(data, 2)` and `recursive_apply(data,
f, registry)` are valid today and unreadable; a breaking change is acceptable: make `max_depth` and
`registry` keyword-only (a `DeprecationWarning` for one release, as `hash_object` did).

### 28.3 `max_depth` has two meanings **[reading]** (P3)

`objects_are_equal(max_depth=100)` is a **limit that raises** (`MaxDepthExceededError`), while
`summarize(max_depth=1)` is a **display depth that truncates** (the rest is shown as a string). The
same parameter name for opposite behaviours is easy to confuse (`summarize(data, max_depth=100)` is
legitimate and `objects_are_equal(data, copy_of_data, max_depth=1)` raises for any structure nested
more than one level). Rename in
`summarize` (`depth` or `expand_depth`) with the old name accepted for a release, or say it in the
first line of both docstrings.

### 28.4 The README shows log output and never mentions `compare` **[verified]** (P3)

The "Debug differences easily" example of the README uses `objects_are_equal(...,
show_difference=True)` and a block "Log output". The differences are emitted with `logger.info`, so a
user who copies the example sees only `False` (the root logger shows WARNING and above).
`compare(...)` (a structured `ComparisonResult` with the path) and `assert_objects_equal` /
`assert_objects_allclose` (the pytest use case, with the path in the `AssertionError`) are the better
first examples; neither the README nor `index.md` nor `get_started.md` mention them, they appear
only in `equality.md`. The overview paragraph also still describes a library that "makes it easy to compare
complex and nested data structures" (19.3, carried): lead with the three features (compare, hash,
summarize) and show `assert_objects_equal` first.

## 29. Tests and CI

### 29.1 Stale non-strict `xfail` markers hide the state of the defects **[verified]** (P2)

`tests/properties/test_library_types.py` marks four tests `xfail(strict=False)` with the number of the
defect. The pytest configuration has `xfail_strict = true`, which the explicit `strict=False`
overrides. Current state:

| Test                                                     | Marker says | Reality                                                                                 |
| -------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------- |
| `test_different_structured_arrays_have_different_hashes` | 14.1        | **XPASS**: fixed by #2250; the marker is stale                                          |
| `test_different_frames_have_different_hashes`            | 14.2        | **XPASS**: fixed by #2251; the marker is stale                                          |
| `test_equal_float_arrays_have_the_same_hash`             | 14.3        | **XFAIL, for another reason**: fixed by #2252, but the strategy flips the sign of `inf` |
| `test_equal_dataclasses_have_the_same_hash`              | 14.7        | XFAIL: the defect is open                                                               |

The third one cannot pass: `floats` includes `float("inf")`, `-x if flip else x` turns it into `-inf`, and
`objects_are_equal(first, second, equal_nan=True)` (the first assertion) is `False`. An exhaustive
check of the other six values (`0.0`, `-0.0`, `1.0`, `-1.0`, `nan`, `-nan`) finds no pair of equal arrays
with different hashes, so the test would pass without `inf`. The marker hides that the property is
now true, and would hide a regression. Fix: remove the flip for `inf` (flip only zeros and NaNs), remove the
three stale markers, and use `strict=True` for the one that remains (14.7) so that fixing it
fails the test until the marker is removed. A rule for the repository: `xfail` in a PR always carries `strict=True`.

### 29.2 Test layout and convention slips **[verified]** (P3)

- `CLAUDE.md` says unit tests mirror the source tree. Fifteen modules have no `test_<module>.py`:
  `hashing/{numpy,pandas,polars,pyarrow,torch,jax,base}.py`, `summary/{base,object_fields}.py`,
  `equality/tester/base.py`, `iterator/{bfs,dfs}/base.py`, `recursive/base.py`, `reducer/base.py`,
  `factory/constants.py`. The hashers of the libraries are tested in one file,
  `tests/unit/hashing/test_array_libraries.py` (86 tests), and `tests/unit/hashing/test_xarray.py` has
  one test (`repr`) while the hasher is tested in `test_array_libraries.py`. Split the file per
  library (`test_numpy.py`, `test_pandas.py`, ...), which also makes the missing dtypes of 22.1 visible.
  The abstract `base.py` modules need no file (say so in `TESTING_GUIDE.md`).
- An AST scan of `tests/unit` finds 31 tests with more than one `assert`/`raises` (6 in
  `factory/test_resolve.py`, 4 in `registry/test_type.py`, 3 in `io/test_base.py` and
  `utils/imports/test_torch_numpy.py`), one test with a loop (`utils/test_bloom_filter.py`) and one
  test that sleeps (`identifier/test_snowflake.py::test_snowflake_id_generator_generate_holds_lock_during_busy_wait`),
  against the rules of `CLAUDE.md`. The sleeping test belongs in `tests/integration/`.
- `fail_under` is still not set in `[tool.coverage.report]` (carried from 18.4); the coverage is 99%, so
  `fail_under = 99` costs nothing. Mutation testing of `equality/handler` is still not done.

## 30. Packaging and documentation

### 30.1 Metadata and release notes **[verified]** (P3, _carried over_ from 19.1 to 19.3)

- `CITATION.cff` still says `title: "COOLA: A Python library to check if two complex/nested objects are
equal or not."`, `date-released: 2023-01-01`, and has no `version`. Use the description of
  `pyproject.toml` and add the release date and version to the release steps
  (nothing in `docs/docs/dev/development.md` describes the release steps yet).
- `Development Status :: 5 - Production/Stable` for `1.2.0a2` (19.1), no `CHANGELOG.md` or "What
  changed in 1.2" page (19.2) are unchanged. Items 22.1 to 22.5 and 23.1 change hash values or
  equality results, so the notes are now needed: each of them has a "behaviour change" line to copy.
- The 19.3 docstring slip (`objects_are_equal` documents `RecursionError`, the class is
  `MaxDepthExceededError`) is unchanged (`equality/interface.py:87`).

## 31. Status of round 4

Checked against `git log 0287a61a..HEAD` and by re-running the repros.

| Round-4 ID                 | Status      | Notes                                                                                                     |
| -------------------------- | ----------- | --------------------------------------------------------------------------------------------------------- |
| 12.1, 12.2, 12.4           | Fixed       | #2238, #2239, #2241                                                                                       |
| 12.3                       | Documented  | #2245 (decision documented, the locks are still held during the comparison)                               |
| 13.1                       | Fixed       | #2242 (pydantic tester and `==` that raises); the generic attributes tester is 23.3, and 23.1 is new      |
| 13.2, 13.3                 | **PR open** | #2243 (symmetric tolerance), #2244 (Decimal, Fraction, timedelta); sets are 23.2                          |
| 13.4, 13.5, 13.6, 13.7     | Fixed       | #2246, #2247, #2248, #2249 (`torch.nested` still raises: 23.3)                                            |
| 14.1, 14.2, 14.3           | Fixed       | #2250, #2251, #2252 (the same family remains for bfloat16/float8: 22.2; pandas missing values: 22.1)      |
| 14.4, 14.5, 14.6, 14.7     | **Open**    | Reproduced; see 22.9 and 22.4                                                                             |
| 15.1, 15.2, 15.3           | Fixed       | #2253, #2254, #2255                                                                                       |
| 16.1 to 16.5               | Fixed       | #2256 to #2260 (the second half of 16.3, `is_object_config`, is 27.2)                                     |
| 17.1 to 17.3, 17.5 to 17.7 | Fixed       | #2261 to #2266 (17.2 as #2262)                                                                            |
| 17.4                       | Fixed       | `utils/imports` is `package.py`, `optional.py` and `lazy.py`; the fallbacks take the package name (#2572) |
| 18.1, 18.2, 18.3           | Fixed       | #2267, #2268, #2269 (but see 29.1 for the markers of 18.2)                                                |
| 18.4                       | Partly      | Type diagnostics of the tests in CI (#2270); `fail_under`, mutation testing, pytest plugin: open          |
| testing.md                 | Fixed       | #2526 to #2530, and the guide PR (`review/testing.md`)                                                    |
| 19.1, 19.2, 19.3           | **Open**    | See 30.1 and 28.4                                                                                         |

## 32. Suggested order

1. **Crash on realistic input, one PR each, with the regression test that would have caught it:**
   22.1 (pandas missing values), 26.1 (`to_jsonable` of a dataclass), 23.1 (pydantic
   `cached_property`).
2. **Test hygiene, first, because it makes the next steps visible:** 29.1 (markers and strategy).
3. **Hash format, in one batch (the hash values change once):** golden vectors (14.6/22.4) first, then
   22.2, 22.3, 22.5, 22.6, 22.7, 14.7, 14.5, and the new hashers of 14.4. Add the frozen registry and
   `HASH_FORMAT_VERSION` in the same release and write the notes (30.1).
4. **Robustness of the process:** 24.1 (fork), 24.2 (copy, scope, reset), 25.1 (cycles and depth of
   the traversal helpers), 24.3 (plugins).
5. **API:** 28.1 (flat namespace), 28.2, 28.3 (breaking), 28.4 (README), 27.1.
6. **Cleanup:** 22.8, 23.2, 23.3, 25.2, 25.3, 26.2, 27.2, 29.2, 17.4 (carried).

## 33. Status of the items fixed in round 5

| ID                                 | Status | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 25.1                               | Fixed  | `recursive_apply`: cycles and any depth with an explicit stack (#2375). `dfs_iterate` and `bfs_iterate` too (#2395).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 25.3                               | Fixed  | `recursive_apply`, `round_values` (#2377), `dfs_iterate` and `bfs_iterate` enter dataclasses by default; `enter_dataclasses` is removed (#2381).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 28.2                               | Fixed  | `recursive_apply`: `registry` is keyword-only, the deprecated paths are removed (#2373), as `dfs_iterate` and `bfs_iterate` (#2395), and `summarize` (#2441).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| F1 to F10 (`review/factory.md`)    | Fixed  | Steps 1 to 8 of the factory design: guide (#2405), one import error (#2406), allow-list and collisions (#2407), `_init_` rule (#2408), `builds` (#2409), dict rule and one import (#2410), integration and property tests (#2411), migration page (this PR).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| F1 to F13 (`review/summary.md`)    | Fixed  | Steps 1 to 7 of the summary design: guide and properties (#2438), `UserString` and `MappingProxyType` (#2439), keyword-only options and validated `max_depth` (#2441), engine, `Node`, `SummaryConfig` and options of `summarize` (#2443), `field(repr=False)` (#2444), documentation (this PR). Also 25.1 for `summarize` (no `RecursionError`) and 25.2 (deterministic sets).                                                                                                                                                                                                                                                                                                                                                                |
| F1 to F13 (`review/reducer.md`)    | Fixed  | Steps 1 to 6 of the reducer design: guide and properties (#2451), `auto_reducer` (#2452), `BaseReducer` template (#2453), input rule and errors (#2454), `NativeReducer.mean` (#2455), migration page (#2456).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| F1 to F13 (`review/registry.md`)   | Fixed  | Steps 1 to 7 of the registry design: guide and properties (#2463), plain `dict` cache (#2464), fallbacks in `TypeRegistry` (#2465) then equality (#2466), summary (#2467), iterator (#2468), recursive and nested (#2469), hashing (#2471), deletion of `DataclassMarker` and `resolve_or` (#2472), one storage class and replaced state (#2473), lazy groups as a tuple (#2475), pickle (#2476), documentation (this PR). Also 24.1 for the registries (a lock held at `fork()`).                                                                                                                                                                                                                                                             |
| F1 to F15 (`review/hashing.md`)    | Fixed  | Steps 1 to 8 of the hashing design: benchmark (#2481), documentation (#2483), equality-consistency and type-sensitivity (#2484), engine and `encode` contract with `__coola_hash__` (#2485), secrets as types with a hasher (#2486), dead and deprecated code (#2487), user guide, architecture and migration page (this PR). Also 22.8 (cost of a small container), 14.7 (`compare=False` fields) and, for `MappingProxyType`, 14.4. The prefix property of `length`, any depth and a linear cost for a DAG are documented. Step 10 (inlined leaves, this PR's successor) makes flat containers about 2x faster than before; the gate is still not met on a scalar, an empty list and deep nesting (see the appendix of `review/hashing.md`). |
| F1 to F13 (`review/io.md`)         | Fixed  | Steps 1 to 7 of the io design: guide (#2499), honest types and strict `TextSaver` (#2500), `_load` template and shared `equal` (#2501), `add_uuid_suffix` to `coola.utils.path` (#2502), save lock removed (#2503), property tests and test layout (#2504), architecture and migration page. F6 (async) is out of scope.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| F1 to F12 (`review/identifier.md`) | Fixed  | Steps 1 to 8 of the identifier design: guide and properties (#2510), the four wrappers removed (#2511), one Crockford decoder and strict inputs (#2512), `TickSequence` (#2513), `SnowflakeIdGenerator(worker_id=...)` (#2515), one content module (#2516), checksummed identifiers (#2517), tests and documentation (this PR).                                                                                                                                                                                                                                                                                                                                                                                                                |
| F1 to F13 (`review/nested.md`)     | Fixed  | Steps 0 to 9 of the nested design: `dfs_items` (#2538), properties (#2539), keyword-only options and private helpers (#2540), keys on the engine (#2541), `from_flat_dict` without recursion (#2542), deterministic `convert_to_jsonable` (#2543), `to_flat_dict` on `dfs_items` (#2544), mappings and records (#2545), `flatten_frame` (#2546), documentation (#2547), benchmark gate (#2549), property tests (this PR). Also 25.3 for `nested` (dataclasses are entered), and 26.2 for `flatten_mapping` (a `TypeError` that names the outer key; `sort_by_keys` and `slugify` are still open).                                                                                                                                              |
| F1 to F13 (`review/utils.md`)      | Fixed  | Steps 1 to 10 of the utils design: three defects (#2567), documentation (#2568), `coola.tools` (#2569, moved back into `coola.utils` afterwards) and `coola.formatting` (#2570), `working_directory` removed (#2571), `imports` rewritten (#2572), keyword-only options and validation (#2573), one qualified name (#2574), `to_jsonable` in `nested` (#2575), `singleton` split (#2576), documentation (this PR). Also 26.2: `sort_by_keys` falls back to `(type name, repr)` and `slugify("😀")` returns a hash.                                                                                                                                                                                                                             |
