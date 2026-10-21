# Codebase review: new suggestions

Scope: every module in `src/coola` (~26k lines, 18 subpackages), plus `pyproject.toml`, the
pre-commit config, the CI workflows and the test layout. I read the source with docstrings
stripped, so docstring quality was not reviewed.

This file **replaces the previous `review/suggestions.md`** (697 lines, every item marked as
handled). That version is still in git history (last commit touching it: `2b558481`). The items below
are new: none of them repeats an item of that review, `src-coola-review.md`, `noqa-review.md` or
`python-311-modernization.md`.

Items marked **[verified]** were reproduced on this checkout (`.venv`, Python 3.14, NumPy, torch, pandas,
polars, pyarrow, xarray, JAX, pydantic installed). The others are leads from reading the code. I did not
run the test suite or the linters, and I did not change any source file.

Priority: **P1** wrong result or crash on realistic input, **P2** inconsistent or fragile
behaviour, **P3** cleanup.

---

## 1. Correctness

### 1.1 `objects_are_equal` can return a non-`bool` **[verified]** (P1)

`ObjectEqualHandler` returns `actual == expected` unchanged. For a dataclass that holds a NumPy
array, `@dataclass` builds `__eq__` from a tuple comparison and the result is an array:

```python
@dataclass
class D:
    a: np.ndarray


objects_are_equal(D(np.ones(3)), D(np.ones(3)))  # array([ True,  True,  True])
```

The function is typed `-> bool`, and `if objects_are_equal(...)` then raises "truth value of an array
is ambiguous". The same happens for any type whose `__eq__` returns an array-like or a lazy object
(pandas `Index`, dask arrays, SciPy sparse matrices).

- Coerce in `ObjectEqualHandler`: `try: result = bool(actual == expected)`, and decide what
  `ValueError` means (fall back to `False` and report, or re-raise with a message that names the type
  and says to register a tester).
- Add a property test: for every pair from the existing strategies, `type(objects_are_equal(a, b)) is bool`.

### 1.2 Tolerance and `equal_nan` are silently ignored for several common types **[verified]** (P1)

Only `int` and `float` (and their subclasses) go through `ScalarEqualityTester`. Everything else falls to
`DefaultEqualityTester` (`==`), which ignores `atol`, `rtol` and `equal_nan`:

| Input                                                                              | Result  |
| ---------------------------------------------------------------------------------- | ------- |
| `objects_are_allclose(np.float32(1.0), np.float32(1.0000001), atol=1e-3)`          | `False` |
| `objects_are_equal(np.float32('nan'), np.float32('nan'), equal_nan=True)`          | `False` |
| `objects_are_allclose(1+1j, 1+1.0000001j, atol=1e-3)`                              | `False` |
| `objects_are_equal(complex('nan'), complex('nan'), equal_nan=True)`                | `False` |
| `objects_are_allclose(F(1.0), F(1.0000001), atol=1e-3)` (a dataclass with a float) | `False` |
| `objects_are_allclose({1.0}, {1.0000001}, atol=1e-3)`                              | `False` |

`np.float64` works only because it subclasses `float`; `np.float32`, `np.int64` and `np.bool_` do not. This
is the most surprising gap for a library whose purpose is tolerant comparison of nested objects.

- Register `np.floating`/`np.integer`/`np.bool_` (guarded by `is_numpy_available`) with the scalar
  tester, converting with `float()`/`int()` inside the handler. Add `complex` with a
  `cmath.isclose`-based handler.
- Add a `DataclassEqualityTester` (same type, then compare the fields through the registry, like
  `MappingEqualityTester`). It fixes 1.1 for dataclasses and gives tolerance inside them.
- Decide whether `set`/`frozenset` should stay exact-only, and document it either way.
- NumPy arrays of dtype `M`/`m` skip the tolerance branch (`is_numeric_array` excludes them) and use
  exact comparison, again without a warning. `PyarrowEqualHandler` does warn when it ignores a
  setting; do the same here.

### 1.3 Polars `Decimal` columns crash the equality check **[verified]** (P1)

```python
objects_are_equal(
    pl.Series([1, 2]).cast(pl.Decimal(10, 2)), pl.Series([1, 2]).cast(pl.Decimal(10, 2))
)
# InvalidOperationError: `is_nan` operation not supported for dtype `decimal[10,2]`
```

`has_nan` (`equality/handler/polars.py`) uses `dtype.is_numeric()`, which is true for `Decimal` and the
integer types, but `is_nan` exists only for floats. Use `dtype.is_float()` (and select
`cs.float()` for frames).

### 1.4 Deeply nested or cyclic data: raw `RecursionError` from `compare()` **[verified]** (P2)

- `a = []; a.append(a); b = []; b.append(b)`: `objects_are_equal(a, b)` raises a `RecursionError` with the
  friendly message, but `compare(a, b)` (and so `assert_objects_equal`) raises the bare
  "maximum recursion depth exceeded". `compare()` calls `registry.objects_are_equal` directly and skips the
  `_compare` wrapper in `interface.py`.
- `max_depth=1000` can never fire with the default Python recursion limit (every level uses several
  frames), so the friendly `MaxDepthExceededError` is unreachable for the default settings.
- Better: detect cycles. Keep a set of `(id(actual), id(expected))` pairs that are currently being
  compared (add on entry, remove on exit, in `check_recursion_depth` or a sibling context manager). A pair
  seen twice is a cycle; treat it as equal (the usual coinductive definition) or raise a clear error.
  This also lets `max_depth` be removed or raised without risk.
- Route `compare()` through the same wrapper as the other entry points.

### 1.5 `compare()` builds two full `repr` strings even when the objects are equal **[verified]** (P2)

`ComparisonResult(True, actual=repr(actual), expected=repr(expected))` runs on every call.
`compare(list(range(10**6)), list(range(10**6))).actual` is a 7.9 MB string, computed on the success path.
`assert_objects_equal` (the common test helper) pays this on every passing assertion.

- Store the objects lazily (`actual: object`, with a `repr`-on-demand property), or truncate with a
  limit (`reprlib`), or only compute them for the failing case.

### 1.6 `recursive_apply` and friends mishandle common containers **[verified]** (P1/P2)

- `recursive_apply(defaultdict(int, {'a': 1}), f)` raises `TypeError: first argument must be callable or
None`: `MappingTransformer` rebuilds with `type(data)(dict)`, which `defaultdict` does not accept. The
  same constructor trick breaks for any mapping or sequence subclass with a different `__init__`.
- `deque` is not a `Sequence` by inheritance (only registered virtually), and `TypeRegistry` resolves by
  `__mro__` only. So `recursive_apply(deque([1, 2]), lambda x: x + 1)` applies `f` to the whole deque
  (`TypeError: can only concatenate deque`), and `dfs_iterate` and `bfs_iterate` yield the deque itself as a
  leaf. The equality registry registers `deque` explicitly; the recursive, iterator, summary and hashing
  registries do not (hashing has it).
- Fixes:
  - Register `deque` (and `OrderedDict` where missing) in every default registry; share one list of
    "container types" instead of five copies (`iterator/bootstrap.py` already does this for two of them).
  - For rebuilding, special-case `defaultdict` (`type(data)(data.default_factory, transformed)`) and fall
    back to `copy.copy(data)` + update for unknown subclasses, or document the constructor contract.
- General point: `Sequence: ...`, `Mapping: ...`, `Collection: ...` entries look like they cover every
  ABC-compatible class, but `TypeRegistry.resolve` never uses `issubclass`, so they only match real
  subclasses. Either say so in the registry docs, or add an opt-in ABC fallback in `_resolve_uncached`
  (after the MRO loop, try registered ABC keys with `issubclass`). The ABC fallback also solves the
  `DataclassHasher`/`HashableHasher` problem in 2.1.

### 1.7 Flat-dict helpers **[verified]** (P2)

- `to_flat_dict({'a': b'xy'})` gives `{'a.0': 120, 'a.1': 121}`: `bytes` and `bytearray` are `Sequence`s and
  are flattened byte by byte. Exclude `(str, bytes, bytearray)` together (and probably `memoryview`).
- `from_flat_dict(to_flat_dict({'a': [1, 2]}))` gives `{'a': {'0': 1, '1': 2}}`. The two functions are not
  inverses for lists. That is fine if documented; if so, name it in both docstrings and add a
  `restore_lists: bool` option, or at least a test that pins the behaviour.
- `to_jsonable` (in `utils/conversion.py`) converts arrays, tensors, paths, enums, dates and models, but only
  recurses for dataclasses. `to_jsonable([np.arange(2)])` returns `[array([0, 1])]`. `convert_to_jsonable`
  does recurse (through `recursive_apply`), so the two are easy to confuse: cross-reference them in the
  docstrings, and handle `bytes`, `Decimal`, `UUID` and `set` (see 2.1).

### 1.8 `polars` `flatten_frame(expand_lists=True)` fails on `Array` columns **[verified]** (P2)

`expand_list_columns` selects `pl.List` and `pl.Array` columns, then calls `.list.len()` and `.list.get()`.
`Array` Series have no `.list` namespace: `InvalidOperationError: expected List data type for list
operation, got: Array(Int64, 2)`. Use `.arr` for `Array` (the width is also known statically from
`dtype.size`, so no `max()` scan is needed).

### 1.9 Files: save-lock robustness (`io/base.py`) (P2)

- **Stale detection can break a live lock.** A lock file older than 300 s is "stale" and removed, but the
  lock's mtime is never refreshed while the holder writes. A legitimate save of a large checkpoint to a
  slow or network disk that takes more than 5 minutes will have its lock broken by a second process, which
  then writes concurrently. Either refresh the mtime from a heartbeat thread, record `pid`/`hostname` in the
  lock file and check liveness, or make `stale_after` a parameter (today the three constants are
  module-private).
- **A broken lock is unlinked by its old owner.** After process B breaks A's stale lock and creates its
  own, A finishes and runs `lock_path.unlink()`, deleting B's lock; process C can then enter while B is
  still writing. Write a unique token into the lock file and delete it only if the token still matches.
- **Temporary file name.** `add_uuid_suffix` produces `data-<hex>.json`, which matches `*.json` globs used by
  readers (dataset loaders, watchers) while a write is in progress. A hidden name (`.data.json.<hex>.tmp`)
  avoids this.
- **`JsonSaver` and `JsonLoader` are asymmetric (verified).** `JsonSaver(encoding="latin-1",
ensure_ascii=False)` writes a file that `JsonLoader().load()` cannot read (`UnicodeDecodeError`; the loader
  opens in binary mode and `json` only auto-detects UTF-8/16/32). Give `JsonLoader` an `encoding` argument.
  `JsonSaver` also has no `default=` hook; `to_jsonable` exists in the package and could be offered as
  `JsonSaver(..., default=to_jsonable)`.
- `fsync=False` by default combined with `replace` can leave an empty file after a power loss. This is
  acceptable, but the docstring of `save_*` should say it in one sentence.

### 1.10 `LazySingleton` deadlocks if a plugin touches the same registry **[verified]** (P2)

`LazySingleton.get` holds a non-reentrant `threading.Lock` while it runs the factory, and the factory runs
the entry-point plugins. A plugin that calls any default-registry function during registration (for
instance `objects_are_equal(...)`, `hash_object(...)`, or an import that does so) re-enters `get()` on the same
thread and hangs forever; I reproduced it with a `register_defaults` function that calls `singleton.get()`.
Use `threading.RLock`, and have the re-entrant call return the registry that is being built (or raise a
clear `RuntimeError`).

### 1.11 `decode_obfuscated_id` wraps instead of rejecting **[verified]** (P3)

`decode_obfuscated_id(encode(5 + 2**64)) == decode_obfuscated_id(encode(5))`: the decoded base-62 value is
masked to 64 bits without a range check, so many strings decode to the same number. Reject values above
`2**64 - 1`. The docstring should also say that this is an affine map modulo 2^64 (two known pairs recover
the key), so it is obfuscation, not protection.

### 1.12 Reducers disagree on NaN **[verified]** (P3)

`NativeReducer().max([1.0, nan, 2.0])` returns `2.0` and `max([nan, 1.0, 2.0])` returns `nan` (built-in
`max` is order dependent), while the NumPy and torch reducers always return `nan`. The reducers are meant to
be interchangeable. Add a NaN check to the native `min`/`max`, or document that NaN input is unsupported.
Also: `auto_reducer()` prefers torch over NumPy; importing torch costs seconds, and NumPy is the better
default for `Sequence[float]`. Consider NumPy first, torch only when the input is a tensor.

### 1.13 xarray equality relies on a private attribute (P2)

`XarrayDataArrayEqualityTester` compares `SameAttributeHandler("_coords")`. `_coords` is a private
attribute of `DataArray` and can change between releases; `coords` (a `Mapping`) is the public way. Also,
`SameDataHandler` compares `Variable.data`, which for dask-backed data is a dask array that falls to `==`
(see 1.1). Check `xr.DataArray` backed by dask and add a test.

---

## 2. Design and API

### 2.1 Hashing covers fewer types than equality (P2) **[verified]**

`hash_object` raises `TypeNotRegisteredError` for `np.float32`, `np.int64`, `uuid.UUID`, `Decimal`, `range`
and any dataclass instance. `np.float64` works only because it subclasses `float`.

- `DataclassHasher` and `HashableHasher` (for objects with a `.hash()` method) exist but cannot be used by
  default: a `type`-keyed registry cannot express "any dataclass" or "any object with this method". Add a
  predicate-based fallback tried after the type lookup fails (`registry.register_fallback(predicate,
hasher)`), and register dataclasses and `SupportsHash` there. The same mechanism would give equality a
  `DataclassEqualityTester` (1.2) and summaries a dataclass summarizer without listing types.
- Register NumPy scalars (`np.integer`, `np.floating`, `np.bool_`) through the built-in hashers (they
  already normalise `float` subclasses), plus `UUID`, `Decimal`, `Fraction`, `range`, `ZoneInfo` and
  `tzinfo`. Make unsupported types fail with a message that names the three remedies (register a hasher,
  `ignore_unhashable=True`, or implement `.hash()`).

### 2.2 Boilerplate that can be generated (P3)

- `utils/imports/*.py` has 13 near-identical modules (four functions each; 14 files with `universal.py`) and
  `utils/fallback/*.py` has 10 more. A table-driven factory (`make_availability_helpers("pandas", install="pandas")` returning the four
  functions) would remove several hundred lines and make adding a dependency a one-line change. Keep the
  per-module names for backward compatibility by assigning the results in each file.
- `utils/imports/__init__.py` re-exports 52 names by hand; `__all__` can be built from the same table.
- Every `*Tester`/`*Summarizer` repeats `_check_dependencies`; a class attribute (`_required_package =
"pandas"`) would do.

### 2.3 Unused or unexported code (P3)

- `identifier.validation.resolve_timestamp_ms` has no caller (ULID and UUIDv7 inline the same logic).
- `display/colorlog.py:configure_colorlog_logging` is not exported from `coola.display` and nothing in the
  repo calls it. Same for the whole `rich` helper set (`is_rich_available`, `check_rich`, ...) and the `rich`
  extra in `pyproject.toml`: no module imports `rich`.
- `hashing.StrHasher` is exported but not part of the default registry (only used in doctests).
- Decide per item: export and document, or delete.

### 2.4 Inconsistent entry-point signatures (P3)

`hash_object(data, registry=None, length=64, ignore_unhashable=False)` takes everything positionally,
while `objects_are_equal`, `summarize` and `recursive_apply` make options keyword-only. Making the options
keyword-only in `hash_object` (after a deprecation cycle) prevents the `hash_object(x, 32)` mistake where
`32` becomes the registry. `RandomManagerRegistry` has `register`, `register_many` and `has_manager` but no
`unregister`, `get` or `find`, unlike the other registries.

### 2.5 Iterators (P3)

- `IteratorRegistry._iterate` copies the ancestor tuple (`(*ancestors, id(data))`) and builds a
  `PathRegistry` for every node, and `id(data) in ancestors` is a linear scan: O(n x depth) time and
  allocations. `PathRegistry.__init__` also skips `super().__init__` and aliases `_state`, which is fragile if
  `BaseTypeDispatchRegistry` gains state. A single mutable set of ancestor ids passed down (add on entry,
  remove on exit) is O(1) per node and removes the subclass.
- The depth-first iterator is built from nested generators, so it hits the recursion limit between depth 300 and 400 (**[verified]**: 300 ok, 400 fails)
  and each yielded leaf pays O(depth) for the `yield from` chain. If deep structures matter, give it an
  explicit stack like the breadth-first one.

### 2.6 `KeyFilterTransformer` double-purposes `func` (P3)

`func` is the predicate for keys, and, through `_value_func`, also the function applied to non-container
values unless `transform_func` is given. With the registry built in `nested/mapping.py` the leaves are
identity transformers, so it works, but with a registry whose leaf is `DefaultTransformer` the predicate
would be applied to every value and replace it with a `bool`. Split into two parameters
(`key_predicate`, `value_func`) and make `value_func` default to identity.

### 2.7 Handler details (P3)

- `if not handler:` / `while current:` in `BaseEqualityHandler` use truthiness; use `is None`/`is not None`
  (a handler subclass with `__len__` or `__bool__` would behave oddly).
- `SameAttributeHandler.__repr__` formats the next handler with `str` (`{self.next_handler}`), the base class
  uses `repr`.
- Pyarrow: `PyarrowEqualityTester` is registered for `pa.Array` and `pa.Table` only. `ChunkedArray` and
  `RecordBatch` go through `==`, which works, but skip the "tolerance is ignored" warnings. Register them.
  The handler emits up to three warnings per call, so a nested structure with many Arrow arrays floods the
  log; warn once per `objects_are_equal` call (state on the config) or once per process.

### 2.8 Summaries (P3)

- `DefaultSummarizer(max_characters=-1)` calls `str(data)` on the whole object and truncates afterwards. For a
  very large custom object the cost is paid before truncation, and the default has no limit. Choose a finite
  default (for example 1000) and truncate through `reprlib` where possible.
- `format_fields` hides everything after `MAX_FIELDS = 10`, but the limit is a module constant that no
  summarizer exposes. Pass `max_fields` through the summarizer constructors.
- `BaseCollectionSummarizer._get_preview`/`_format_items` raise `NotImplementedError`; make them
  `@abstractmethod` so an incomplete subclass fails at instantiation.

### 2.9 Identifiers (P3)

- `SnowflakeIdGenerator._reset_after_fork` replaces the lock but keeps the sequence and last timestamp.
  Parent and child (same `worker_id`) can then produce the same id in the same millisecond. Document that
  `worker_id` must be unique per process, or mix `os.getpid()` into the default worker id after a fork.
- `snowflake` raises `RuntimeError` when the clock moves backwards, which is a legitimate choice. Mention
  it in the guide, with NTP step adjustments as the typical cause.

### 2.10 Smaller points

- `testing/fixtures.py` evaluates `is_cuda_available()` and `is_mps_available()` at import. The latter
  allocates a tensor on the MPS device, so merely importing the fixtures initialises a GPU context. Make the
  markers lazy (`pytest.mark.skipif` accepts a callable condition string, or use `pytest_collection_modifyitems`).
- `LRUCache` is the only stateful container in `utils` that is not thread-safe (the registries and
  singletons are). Say so in the docstring or add a lock.
- `utils.stats.quantile` sorts NaN-containing input in an undefined order; either reject NaN or document it.
- `char_diff_summary` reports `+0.0%` for growth from an empty string (division guarded to `0.0`), which
  hides the change; return `n/a`.
- `utils.fallback.torch` fakes only `cuda`, `nn.utils.rnn.PackedSequence`, `Tensor` and `tensor`; any new
  torch attribute used at import time in a module that is imported without torch will raise `AttributeError`
  instead of the helpful "install torch" message. A `__getattr__` on the fake module that calls
  `raise_torch_missing_error()` makes this robust for every missing attribute.

---

## 3. Tests, tooling and CI

- Add regression tests for every **[verified]** item above. Most are one-liners; the repro snippets are in
  this file.
- Extend the property tests with invariants that would have caught 1.1 and 1.2:
  `type(objects_are_equal(a, b)) is bool`; `objects_are_allclose(x, x)` for every scalar type produced by the
  strategies (include `np.float32`, `np.int64`, `complex`); `objects_are_equal(x, deepcopy(x))` for nested
  structures with `equal_nan=True`.
- Add a concurrency test for the save lock (two processes, a slow `_save_file`, `stale_after` shortened) to pin
  1.9.
- Still open from the earlier review: a one-off mutation-testing run, a coverage `fail_under`, and a second
  type checker in CI. The mutation run would be most useful on `equality/handler` and `registry`.
- `ci-test.yaml` runs the whole unit and integration suite with a 5-minute timeout per matrix cell. That is
  fine today; if the matrix grows, `pytest-xdist` is the cheap way to keep the budget.

---

## 4. Suggested order

1. 1.1 + 1.2 + 1.3 (wrong or crashing equality results; small, local changes with clear tests).
2. 1.6 + 1.7 (container handling in `recursive`, `iterator`, `nested`).
3. 1.4 + 1.5 (cycle detection and the eager `repr`).
4. 1.9 + 1.10 (lock token and `RLock`).
5. 2.1 (predicate fallback in the registry, then dataclass support for equality, hashing and summaries).
6. The rest, as cleanup.
