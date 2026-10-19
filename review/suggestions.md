# Codebase review and suggestions

Scope: all of `src/coola` (~23.8k lines, 18 subpackages), plus `pyproject.toml`, the
pre-commit config, the CI workflow list and the test layout. I read the source with
docstrings stripped, so docstring quality was not reviewed.

State of the repo when I ran it: `ruff check src` and `ty check` pass, and
`tests/unit` runs 5,717 tests in about 12 s with no coverage gaps reported. The code
is consistent and carefully tested. The suggestions below are therefore mostly
correctness edge cases and design consistency, not hygiene.

## Status

All the sections have been handled. Each item has a _Status_ line that says what was done, and
section 7 has a table with one line per item. What is **not** done, and why:

- ~~**Property-based tests**~~ (section 4): implemented, see section 4.
- **Mutation testing** (section 4): a one-off run that has not been done.
- **`fail_under` for the coverage** and a **second type checker** in the CI (section 5): they need a
  decision on the CI jobs.
- **Hashers for arrays and data frames** (2.9): there is no canonical byte form, so the hashing user
  guide shows how to register one.
- **Kept by decision**: the `get_default_*_registry` aliases (2.6), the top-level package without
  re-exports (2.7), the `to_date` `tzinfo` (1.10), the deferred import of the registry equality
  (section 5).

Items marked **[verified]** were reproduced in a Python session on this checkout.
The rest are leads from reading the code. The earlier reviews in this folder
(`src-coola-review.md`, `noqa-review.md`, `python-311-modernization.md`) are not
repeated here.

---

## 1. Correctness issues

### 1.1 Hashing: different values produce the same hash **[verified]** (high)

_(Status: fixed. Hash inputs are namespaced by type: `sequence`, `mapping`, `bytes`, `pydantic`, `Path`, the date class name and the builtin scalar type (`hash_tagged`). Subclasses such as NumPy scalars hash like the builtin value. List and tuple deliberately still hash the same. Not done: `PathHasher` still resolves paths, and the hashers still have no depth guard. All hash values changed.)_

`coola.hashing` hashes scalars with `repr`, strings with their bytes, and containers
by concatenating child hashes. No hasher mixes a type tag into the digest, so
different objects collide:

- `hash_object(1) == hash_object('1')`
- `hash_object([]) == hash_object({}) == hash_object('')`
- `hash_object([1, 2]) == hash_object((1, 2))`

`generate_stable_content_id` and `generate_stable_uuid5` are built on this, so
"stable content id" values can collide across types.

- Prefix each leaf and container hash input with a type tag (e.g. `b"int:"`,
  `b"list:"`). This changes every existing hash, so make it a documented breaking
  change or add a `version` argument.
- If list and tuple are meant to hash alike, say so in the docstring and add a test.
  If not, tag them separately.

Related hashing points:

- `ReprHasher` is registered for `float`, so `np.float64(1.0)` (a `float` subclass)
  hashes `repr()` = `'np.float64(1.0)'` on NumPy 2 and `'1.0'` on NumPy 1.
  **[verified]** for the NumPy 2 output. The hash then differs between NumPy versions
  and from `hash_object(1.0)`. Register NumPy scalars explicitly, or hash via
  `float.__repr__(x)` / `int.__repr__(x)`.
- `PathHasher` calls `path.resolve()`. A relative path then hashes differently
  depending on the current directory, and the result depends on symlinks and the
  filesystem. Hash `path.as_posix()` by default and make resolving opt-in.
- `SequenceHasher` caches child hashes by `id(item)`. That is safe for stored items
  but not for sequences that build items on access (see 1.3).
- `MappingHasher.hash` and `SequenceHasher.hash` recurse without a depth guard, so a
  self-referencing list raises a bare `RecursionError`. `objects_are_equal` has
  `max_depth` and a friendlier message.

### 1.2 `PydanticModelHasher` misses secrets inside tuples and sets **[verified]** (high)

_(Status: fixed. Secrets are found in lists, tuples, sets and frozensets; sets are hashed as sorted lists; the error message names `hash_pydantic_model()`. Not done: the `SecretBytes` / `list[SecretStr]` detection in `display/pydantic.py`.)_

`unwrap_secrets` only walks `dict` and `list`. For a model with
`t: tuple[SecretStr, ...]`, `model_dump()` gives a tuple, which is returned
unchanged. `json.dumps(..., default=str)` then writes `'**********'` for every
secret. With the default `on_secret='error'`:

- no error is raised, and
- two models with different secrets hash identically (reproduced).

That defeats the `error` and `exclude` modes and makes the hash blind to secret
values. Walk `tuple`, `set`, `frozenset` as well, or use `model_dump(mode='json')`
with a custom serializer. Also:

- the error message mentions `hash_model()`, which does not exist (the function is
  `hash_pydantic_model`);
- `display/pydantic.py::secret_field_names` only recognizes `SecretStr`, not
  `SecretBytes`, and not `list[SecretStr]`. `_mask_nested_secret_fields` handles
  nested models and containers but relies on the annotation scan for the top level.
  Align both with the same "is a secret type" predicate.

### 1.3 `id()`-keyed result caches in equality handlers (medium)

_(Status: fixed. The `id()` caches were removed from the mapping and sequence handlers and from `SequenceHasher`.)_

`MappingSameValuesHandler` and `SequenceSameValuesHandler` cache `(id(v1), id(v2))`
per call. This assumes values stay alive for the whole loop. For `Mapping` and
`Sequence` implementations that create their values on access (lazy views, wrappers
over arrays, ORMs), a freed object's id can be reused by a later value. A stale
cached `True`/`False` is then returned for a different pair.

The cache only helps when the same objects repeat, which is rare outside shared
sub-structures. Either remove it, or keep a reference to the objects in the cache
(`cache[key] = (value1, value2, result)`) so ids cannot be recycled while cached.
The same pattern is in `SequenceHasher`.

### 1.4 NumPy string arrays raise with `equal_nan=True` **[verified]** (medium)

_(Status: fixed. `equal_nan` is only forwarded to NumPy for numeric and datetime dtypes.)_

`objects_are_equal(np.array(['a']), np.array(['a']), equal_nan=True)` raises
`TypeError: ufunc 'isnan' not supported`. `handler/numpy.py::array_equal` guards the
tolerance branch with `is_numeric_array` but calls `np.array_equal(..., equal_nan=...)`
unconditionally. Pass `equal_nan` only for float/complex dtypes (`dtype.kind in 'fc'`),
or catch `TypeError` and fall back. Object and datetime arrays deserve a test too.

### 1.5 Scalar tolerance on huge ints **[verified]** (low)

_(Status: fixed. `number_equal` falls back to the exact `int_isclose` when `math.isclose` overflows.)_

`objects_are_allclose(10**400, 10**400 + 1)` raises `OverflowError` from
`math.isclose`. Fall back to exact comparison or an integer-aware difference when
either operand is an `int` that does not fit a float.

### 1.6 `to_flat_dict` / `from_flat_dict` lose data and are not inverses **[verified]** (medium)

_(Status: fixed. Empty containers are kept as leaves, key collisions raise `ValueError`, a `None` leaf no longer hides a nesting conflict, and non-string keys raise `TypeError`. Lists still come back as dicts: the round trip is lossy by design.)_

For `{'a': {}, 'b': [1, 2], 1: 'x', '1': 'y'}`:

- `to_flat_dict` returns `{'b.0': 1, 'b.1': 2, '1': 'y'}`. The empty dict `'a'` is
  dropped, and `1` / `'1'` collide because keys go through `str()` (the later one wins
  silently).
- `from_flat_dict` returns `{'b': {'0': 1, '1': 2}, ...}`. Lists come back as dicts.
- `from_flat_dict` treats an existing `None` leaf as "absent" (`existing is None`),
  so `{'a': None, 'a.b': 1}` silently replaces the `None` instead of reporting a
  key conflict. It also crashes with `AttributeError` for non-string keys.

Suggestions: raise on key collisions after `str()`, emit empty containers as a leaf
(`{}` / `[]`), and document explicitly that the round trip is lossy (or add
`list_keys='index'` to rebuild lists).

### 1.7 `Reducer` backends disagree on the median **[verified]** (medium)

_(Status: fixed. The Torch median averages the two middle values; `TorchReducer` keeps the precision of its input (`float64` for Python lists, the tensor dtype otherwise) and no longer rounds Python floats to `float32` in `max`/`min`/`sort`. Consistency tests compare all reducers. `auto_reducer()` still prefers Torch.)_

For `[1, 2, 3, 4]`: `NativeReducer` and `NumpyReducer` return 2.5, and
`TorchReducer` returns 2.0 (`torch.median` returns the lower middle value). Because
`auto_reducer()` prefers Torch when installed, the result depends on which packages
are installed. Fix by averaging the two middle values in `TorchReducer._median`
(`torch.quantile(x.float(), 0.5)` or sort and average). Also:

- `TorchReducer` computes `mean`, `std`, `quantile` in float32, while the other two
  use float64. Cast to `float64`.
- `torch.quantile` rejects inputs above 16M elements.
- `auto_reducer()` importing Torch just to compute stats is heavy. Prefer NumPy, then
  native, and make Torch opt-in.

### 1.8 `compare()` mutates a global logger (medium)

_(Status: fixed. `EqualityConfig.reporter` and `report_difference()` replace the logger capture: `compare()` takes no lock and changes no logger state. Custom handlers must call `report_difference` to be captured by `compare()`.)_

`equality/result.py::compare` takes a process-wide RLock, then temporarily sets
`coola.equality` logger level to INFO and `propagate = False`. Consequences:

- Concurrent `compare()` calls are serialized.
- Other threads' `coola.equality` records are dropped from normal handlers while a
  comparison is running (`CaptureHandler` filters by thread only for its own
  capture).
- If a handler configured on the `coola.equality` logger itself raises, the state
  restore still runs (good), but the level/propagate swap is visible to any thread.

A cleaner design is to give `EqualityConfig` an optional "reporter" callback that
handlers call, in addition to (or instead of) logging. `show_difference` logging can
then stay as a thin adapter, and `compare` needs no logger tricks. At minimum, document
the thread-safety limits.

### 1.9 `_compare` RecursionError rewrite is hard to read (low)

_(Status: fixed. `check_recursion_depth` raises `MaxDepthExceededError` (a `RecursionError`), and `_compare` no longer matches on the message text.)_

`equality/interface.py::_compare` re-raises when `'Maximum recursion depth' in
str(exc)` and otherwise wraps with a custom message. The string match depends on
CPython's message text, and the branch that wraps is the surprising one. Add a
comment on the intent, or catch `RecursionError` once and always raise the friendlier
message with `from exc`.

### 1.10 Small correctness items

_(Status: fixed, except `to_date`: the `.replace(tzinfo=UTC)` is what satisfies the `DTZ007` lint rule, so it was left. `DuplicateKeyError(KeyError, ValueError)` was added for the `nested` mapping helpers; `str_indent` and `timeblock` now raise `ValueError`.)_

- `validation/*`: `validate_non_negative(float('nan'))` and `validate_positive(nan)`
  pass because `nan < 0` is `False`. `EqualityConfig` adds `validate_finite` after
  the fact; other callers (`min_length`, `context_lines`, `rtol` in
  `objects_are_allclose`, which validates before `EqualityConfig` does) do not. Reject
  NaN inside the comparison validators.
- `utils/stats.py::quantile` silently clamps `q` outside `[0, 1]`. Validate instead.
- `utils/env_vars.py::temp_env_vars` restores with `os.environ.pop(key)`. If the body
  already removed the variable, this raises `KeyError`; use `pop(key, None)`. Also
  move the "set" loop inside the `try` so a failure halfway still restores.
- `utils/format.py::str_indent` raises `RuntimeError` for a negative `num_spaces`
  with a message saying "positive". Use `ValueError` and "non-negative". The same
  applies to `timeblock` (`RuntimeError` for a bad message template) and
  `Handler.chain` style checks.
- `utils/timing.py::timeblock` calls `message.format(time=...)` on user text, so any
  other `{}` in the message raises `KeyError`/`IndexError` after the block ran.
- `utils/date.py::to_date` parses with `strptime`, attaches UTC and then discards it
  with `.date()`. The `replace(tzinfo=UTC)` is a no-op.
- `utils/array.py::to_array` / `utils/tensor.py::to_tensor`: `tensor.numpy()` fails
  for tensors that require grad or live on GPU (needs `.detach().cpu()`);
  `torch.from_numpy` shares memory and fails on negative strides.
- `utils/conversion.py::to_jsonable` recurses through `dict` and `list` only
  (tuples and sets inside dataclasses are not converted) and does not handle NumPy
  scalars, `Path`, `datetime` or `Enum`, which `json.dump` cannot serialize either.
- `nested/mapping.py::merge_mappings(on_duplicate='suffix')` builds `f'{key}_{n}'`
  without checking that the new key does not already exist.
- `nested/mapping.py`: duplicates raise `KeyError` (which quotes its message) while
  all the other validation in the module raises `ValueError`.

---

## 2. API and design consistency

### 2.1 Optional-dependency errors should be `ImportError` (medium)

_(Status: fixed. `MissingPackageError(ModuleNotFoundError, RuntimeError)`, `AlreadyRegisteredError(ValueError, RuntimeError)` and `HandlerCycleError(ValueError, RuntimeError)` were added; the old `RuntimeError` is kept as a base class for compatibility. Not done: `decorator_package_available` still returns `None` silently, because it is public API.)_

`utils/imports/universal.py::raise_package_missing_error` and `check_package` raise
`RuntimeError`. The standard is `ImportError` (or `ModuleNotFoundError`). Callers
writing `try: ... except ImportError:` will not catch these. `RuntimeError` is also
used for registry duplicates, cyclic handler chains, lock timeouts and bad arguments,
which makes `except RuntimeError` meaningless. Proposed mapping:

| Case                                                | Current        | Better                                                           |
| --------------------------------------------------- | -------------- | ---------------------------------------------------------------- |
| Missing optional package                            | `RuntimeError` | `ImportError` (subclass `ModuleNotFoundError`)                   |
| Duplicate registry key (`register(exist_ok=False)`) | `RuntimeError` | `KeyError`/`ValueError`, or a dedicated `AlreadyRegisteredError` |
| Bad argument (`str_indent`, `timeblock`)            | `RuntimeError` | `ValueError`                                                     |
| Handler cycle                                       | `RuntimeError` | `ValueError`                                                     |

Keep a deprecation path: a subclass of both (`class X(ImportError, RuntimeError)`)
preserves old `except RuntimeError` code.

Also `decorator_package_available` returns `None` silently when the package is
missing. A function decorated this way returns `None` instead of failing, which hides
the real error. Prefer `pytest.mark.skipif` semantics only in `coola.testing`.

### 2.2 `is_xxx_available` uses `find_spec` only (low)

_(Status: fixed. The `is_<package>_available` functions now use `module_available`, which imports the module, so a broken installation is reported as unavailable and a warning with the traceback is logged. `package_available` stays a cheap lookup that does not import. `clear_availability_caches()` resets all the cached checks; the functions are registered by the `availability_cache` decorator, so there is no list to maintain.)_

`package_available` returns `True` if the module can be found, not if it imports.
A broken install (ABI mismatch, missing shared library) reports "available", then
crashes at import time with a long trace instead of the intended
"not installed" path. `@cache` / `lru_cache` also makes the answer impossible to
change in-process, which is why tests need to patch the function. Consider
`importlib.util.find_spec` plus a guarded `import` for the packages whose import is
cheap and failure-prone, and expose `cache_clear` in the testing helpers.

### 2.3 Equality tester boilerplate (medium, simplification)

_(Status: fixed. `BaseChainEqualityTester` (`equality/tester/chain.py`) implements `equal`, `objects_are_equal`, the repr and the chain creation; each built-in tester now only defines `_create_handler` (and `_check_dependencies` when it needs an optional package). The 27 `_get_repr_kwargs` methods that returned `{}` were removed in favor of `NoArgsDisplayMixin`, which is now used by the hashers, transformers, summarizers, iterators, finders, random managers, reducers and io classes.)_

About 25 tester classes (`tester/numpy.py`, `pandas.py`, `polars.py`, `torch.py`,
`xarray.py`, `jax.py`, `pyarrow.py`, `scalar.py`, `default.py`, `equal.py`,
`tolerant.py`, `collection.py`) repeat the same body:

```python
def _get_repr_kwargs(self):
    return {}


def __init__(self):
    check_x()
    self._handler = create_chain(...)


def equal(self, other):
    return type(other) is type(self)


def objects_are_equal(self, actual, expected, config):
    return self._handler.handle(...)
```

`HandlerEqualityTester` already exists and does this generically but is unused for
the built-ins (it is only used by user code). Suggestion: build the built-in testers
as `HandlerEqualityTester(create_chain(...))` instances, or a small base class
`ChainEqualityTester` with a class-level `_chain_factory`. That removes several hundred
lines and makes adding a backend a ten-line change. Keep the named classes as thin
subclasses if they are public API.

The same applies to `Hasher`, `Transformer`, `Summarizer`, `ChildFinder` and
`Iterator` implementations: each has `_get_repr_kwargs -> {}` boilerplate that
`NoArgsDisplayMixin` (already defined in `display/mixin.py`) is meant to absorb. It
is currently used nowhere under `src/`. Use it.

### 2.4 `equal()` identity checks are copy-pasted (low)

_(Status: fixed. `is_same_type(value, reference)` in `coola.utils.introspection` is a `TypeGuard`, so the type checker narrows `other` and the redundant `isinstance` clause is gone. It replaces every `type(other) is type(self)` check in `src`.)_

`type(other) is not type(self) or not isinstance(other, type(self))` appears in
`BaseRegistry.equal`, `HandlerEqualityMixin.equal`, `HandlerEqualityTester.equal` and
`BaseCollectionSummarizer.equal`. The second clause is redundant after the first (it
is there to narrow the type for the checker). Use `if type(other) is not type(self):`
plus `cast`, or a shared helper. Other classes only test the first clause, so the
codebase is inconsistent.

### 2.5 Registry duplicate-key semantics (low)

_(Status: fixed, except the atomicity point (not an issue for the built-in registries). `KeyNotRegisteredError(KeyError, LookupError)` is now the base of `TypeNotRegisteredError`: a plain `Registry` raises `KeyNotRegisteredError`, a `TypeRegistry` still raises `TypeNotRegisteredError`. The copy-on-write cache has an explanatory comment. The `issubclass` fallback for virtual subclasses was deliberately **not** added: `str`, `bytes` and `range` are virtual `Sequence`s, so `str` would be dispatched to the `Sequence` handler and iterated character by character. This is documented in the `TypeRegistry` docstring.)_

- `BaseRegistry.register_many` is not atomic with respect to partial failures in
  subclasses overriding `_on_change`, but is fine for the built-in ones.
- `Registry.__getitem__` / `unregister` raise `TypeNotRegisteredError` for non-type
  keys (e.g. `RandomManagerRegistry` string keys). The name is misleading for a
  string-keyed registry; consider `KeyNotRegisteredError` as a base with
  `TypeNotRegisteredError` as a subclass.
- `TypeRegistry.resolve` rebuilds the whole cache dict (`cache.copy()`) on every miss
  while the cache has up to 1,024 entries. Copy-on-write is deliberate (lock-free
  reads), but a burst of distinct types makes this O(n²). Fine in practice; a short
  comment explaining the trade-off would help.
- Resolution walks `dtype.__mro__` only. ABCs registered virtually
  (`collections.abc.Sequence.register(MyClass)`, `range`, `memoryview`) are never
  found even though `Sequence` is registered. Document this; optionally add an
  `issubclass` fallback for ABC keys.

### 2.6 `get_default_*_registry` aliases (low)

_(Status: not changed, by decision. The long names are kept as plain aliases of `get_default_registry`: they are documented in the user guide as "Unambiguous alias" and are useful to avoid import collisions, so they are not deprecated. A deprecation helper was tried and rolled back.)_

Each `interface.py` defines both `get_default_registry` and
`get_default_<x>_registry`. They were kept as public API. A short module-level note
saying which name is canonical, and a `DeprecationWarning` on the long names in the
next minor release, would stop the surface from growing.

### 2.7 Package-level exports (low)

_(Status: not changed, by decision. The top-level package only exposes `__version__` on purpose; the entry points are imported from their subpackage. A lazy re-export was tried and rolled back.)_

`coola/__init__.py` only exports `__version__`. Lazy re-exports through module
`__getattr__` would keep import time low while letting
`coola.objects_are_equal(...)`, `coola.summarize(...)` work. If this is rejected on
purpose (the docstring suggests it is), keep as is.

### 2.8 `KeyFilterTransformer` does not match its siblings (low)

_(Status: fixed. `KeyFilterTransformer` now has the `Transformer` signature `transform(data, func, registry)`; `transform_func` moved to the constructor (`KeyFilterTransformer(transform_func=...)`), and it uses `InlineDisplayMixin`. Passing `transform_func` to `transform` is no longer accepted.)_

`recursive/key.py::KeyFilterTransformer` does not use a display mixin, and its
`transform` takes an extra keyword-only `transform_func` that the `Transformer`
protocol does not have. `nested/mapping.py` builds a private registry around it.
Either document it as an internal helper (leading underscore) or make it satisfy the
protocol (move `transform_func` into the constructor).

### 2.9 Summary coverage vs equality coverage (medium)

_(Status: fixed for `summary`. `BaseMetadataSummarizer` (`summary/metadata.py`) is the shared base, and `NDArraySummarizer` / `TensorSummarizer` now use it. New summarizers, registered in the default registry when their package is installed: `PandasDataFrameSummarizer`, `PandasSeriesSummarizer`, `PolarsDataFrameSummarizer`, `PolarsLazyFrameSummarizer` (does not collect), `PolarsSeriesSummarizer`, `PyarrowArraySummarizer`, `PyarrowTableSummarizer`, `XarrayDataArraySummarizer`, `XarrayDatasetSummarizer`, `XarrayVariableSummarizer`, `JaxArraySummarizer`. All accept `show_data`; wide frames list at most 10 columns. Decisions: `recursive` is unchanged (an array is a leaf passed to the function, which is what callers want). `hashing` is unchanged: a stable array hash needs decisions on dtype, byte order, layout, NaN payloads and object dtype, and pandas/polars have no canonical byte form, so it is left to `register_hashers`.)_

`summary` has summarizers for `list`, `tuple`, `set`, `dict`, NumPy and Torch only.
Equality supports pandas, polars, pyarrow, xarray and JAX. `summarize(df)` falls back
to `str(df)` prefixed by the type, which can be thousands of lines for a DataFrame
even with `DefaultSummarizer`'s default `max_characters=-1`. Add summarizers for
`pandas.DataFrame/Series`, `polars.DataFrame/Series`, `pyarrow.Table`, `xarray.*`
and JAX arrays that print shape, dtype and column names, with the same `show_data`
flag as `NDArraySummarizer`. Make the same decision for `recursive` and `hashing`
(hashing has no NumPy/Torch/pandas hasher at all, so `hash_object(np.array(...))`
raises or falls back to `object` handling).

### 2.10 BFS/DFS iterators (low)

_(Status: fixed. The default registries register `Collection` instead of `Iterable`, so iterators and generators are leaves and are never consumed. `BaseChildFinder.is_leaf` (default `False`, `True` for `DefaultChildFinder`) replaces the `isinstance(..., DefaultChildFinder)` check. Both `bfs_iterate` and `dfs_iterate` raise `ValueError("Cycle detected ...")` for a container that contains itself; a shared sub-container is not a cycle. In DFS the iterators receive a registry view that knows the path, so custom iterators that call `registry.iterate` are protected too.)_

- Both register `Iterable` as a "container". A generator, file object or infinite
  iterator passed in is consumed (or never ends). Restrict to `Collection` or
  document the behavior in the docstring.
- `ChildFinderRegistry.iterate` detects a leaf with
  `isinstance(child_finder, DefaultChildFinder)`. That couples the traversal to one
  concrete class: a user finder that subclasses it, or a custom leaf finder that does
  not, changes behavior. Add an `is_leaf` property to `BaseChildFinder` instead.
- There is no cycle protection. A self-referencing list loops forever in BFS (the
  queue grows without bound) and recurses to `RecursionError` in DFS.

### 2.11 `random` package (low)

_(Status: fixed. `BaseRandomManager.seed(seed)` is now the single "save, seed, restore" context manager; `random_seed`, `numpy_seed` and `torch_seed` delegate to it. The NumPy seed folding (`seed % 2**32`) was kept and is now documented, with tests, rather than rejected: rejecting would break `random_seed(seed)` for the other libraries with seeds above 2\**32. The docstrings and the user guide state that only the global generators are managed (not `random.Random()`, `numpy.random.default_rng()` or `torch.Generator()`). `RandomManagerRegistry.set_rng_state` now also warns about registered keys that are missing from the state.)_

- `random_seed`, `numpy_seed` and `torch_seed` each reimplement "save state, seed,
  yield, restore". Put one `seed(self, seed)` context manager on `BaseRandomManager`
  and have the three functions delegate.
- `NumpyRandomManager.manual_seed` does `seed % 2**32`, so seeds `1` and `2**32 + 1`
  give identical streams and negative seeds wrap silently. Reject out-of-range seeds
  or document the folding.
- Only legacy global RNGs are handled (`np.random.seed`, `random.seed`,
  `torch.manual_seed`). Say so in the docs, since modern NumPy code uses
  `np.random.default_rng()` which this does not touch.
- `RandomManagerRegistry.set_rng_state` ignores unknown keys with a log warning;
  keys present in the registry but missing from `state` are skipped silently.

### 2.12 `io` package (medium)

_(Status: fixed. (1) The `BaseLoader` / `BaseSaver` equality testers are now registered by the default equality registry (`_get_io_equality_testers`, with a deferred import of `coola.io.base`), not as an import side effect; the test workaround that re-registered them in `tests/unit/io/conftest.py` was removed. (2) `BaseFileSaver` falls back to a check + rename when `os.link` is not supported. (3) The write access needed on the directory (temporary file and lock file) is documented; a `lock_dir` option was not added, since the temporary file needs the directory to be writable anyway. (4) `save(..., fsync=True)` (also `save_json`, `save_pickle`, `save_text` and `save_torch`) flushes the file before the commit and the directory after; the default is unchanged. (5) The pickle warnings already existed on the loaders and `load_pickle`; a security note was added to `load_torch` (`weights_only=True` default) and to the user guide.)_

- `io/base.py` registers `EqualNanEqualityTester` for `BaseLoader`/`BaseSaver` in the
  global equality registry at import time. Importing `coola.io` therefore mutates a
  singleton in another package. Register them in
  `_get_native_equality_testers()` (they already are in the same family as
  `BaseHandler`/`BaseEqualityTester`) or via the plugin entry point, not as an import
  side effect.
- `BaseFileSaver.save(exist_ok=False)` publishes with `os.link`. Hard links are not
  supported on some filesystems (FAT/exFAT, some network and container volumes);
  the failure is an `OSError` that looks unrelated. Fall back to `os.open(O_EXCL)` +
  rename, or catch `OSError` and raise `FileExistsError` only when it really exists.
- The `.lock` file is created next to the target, so saving into a read-only
  directory with `exist_ok=True` fails at lock time. Document that, or allow a
  `lock_dir`.
- No `fsync` before the atomic rename, so a power cut can leave an empty target.
  Fine for caches, worth a sentence in the docs, or an `fsync=True` option.
- `PickleLoader.load` unpickles arbitrary data. The docstring should carry a
  prominent security note (as `SECURITY.md` probably does for the factory). `TorchLoader`
  defaults to `weights_only=True`, which is good; mention that default next to the
  pickle warning.

### 2.13 Factory security note (medium)

_(Status: fixed. `allowed_prefixes` is available in `resolve_object`, `resolve_loader`, `resolve_saver`, `is_object_config`, `is_loader_config` and `is_saver_config`; the caller's list replaces the `_allowed_prefixes_` key of the configuration so a configuration cannot widen it. `is_object_config` returns `False` when the return type hint cannot be resolved (`NameError`) and uses `issubclass`, so virtual subclasses match. The docstrings, `SECURITY.md` (including that the `is_*_config` functions import the target) and the factory user guide describe the risk and the mitigation. The default is unchanged: no allow-list.)_

`factory.resolve_object` / `factory.factory` import and call whatever `_target_`
names. `allowed_prefixes` exists but is opt-in and not reachable from
`resolve_object` (only from `factory(**cfg)` via `_allowed_prefixes_`). Configs that
come from YAML/JSON files or users therefore execute arbitrary callables. Suggestions:

- Add `allowed_prefixes=` to `resolve_object` and `resolve_loader` / `resolve_saver`.
- State in the docstring and `SECURITY.md` that configs must be trusted.
- `is_object_config` calls `get_type_hints`, which can raise `NameError` for
  forward references; wrap it and return `False`.
- `is_object_config` uses `cls in t.__mro__` where `issubclass(t, cls)` also covers
  ABC virtual subclasses.

### 2.14 Identifiers (low)

_(Status: fixed. (1) `ObjectIdGenerator` regenerates its process value and counter, and recreates its lock, in the child of a fork (`coola.identifier.fork.track_fork`, based on `os.register_at_fork`); `SnowflakeIdGenerator` recreates its lock and its docstring and the user guide warn that forked children must use a distinct `worker_id`. (2) `generate_ulid` and `generate_uuid7` are now monotonic within a process when the current time is used (`MonotonicRandom`: the random part is incremented within the same millisecond or if the clock moves backward, and borrows the next millisecond on overflow); an explicit `timestamp_ms` keeps a random part. (3) `generate_obfuscated_id` and `decode_obfuscated_id` have a \"not encryption\" warning, and so does the user guide. (4) The `snowflake` clock regression point needed no change.)_

- `ObjectIdGenerator` and `SnowflakeIdGenerator` are process-wide `@cache`
  singletons. After `os.fork()` the child inherits the same random process value and
  counter, so two forked workers can emit identical ObjectIds. Re-create the
  generator in `os.register_at_fork(after_in_child=...)`.
- `generate_ulid` and `generate_uuid7` are not monotonic within the same millisecond.
  The ULID spec recommends monotonic ordering; for UUIDv7 it is optional but
  commonly expected for DB indexes. Either implement a monotonic counter or state
  "not monotonic within one millisecond" in the docs.
- `obfuscated.py` is an affine permutation (`n * a + b mod 2**64`). It hides
  sequence numbers from casual inspection but is trivially invertible from two samples
  (and `decode_obfuscated_id` is provided). Add a one-line "not encryption" warning.
- `snowflake.py` handles clock regression by raising `RuntimeError` (good), and the
  wait-for-next-ms loop re-checks the clock every iteration (so it cannot spin
  forever). That resolves item 3.5 of the earlier review.

### 2.15 `testing.fixtures` repetition (low)

_(Status: fixed. `coola.testing.fixtures` builds each pair of marks with the public `requires(is_available, name)`: supporting a new dependency is one line plus the two names in `__all__`. The 28 marks are unchanged (same condition and same skip reason). The `pytest` dependency is left as is, by decision: it is stated in the module docstring, and the fixtures are only useful with `pytest`.)_

`testing/fixtures.py` defines 28 near-identical `pytest.mark.skipif` lines. A tiny
helper (`_requires(name, is_available)` returning the pair) makes adding a backend a
one-line change. `testing` also imports `pytest` at module import time, so
`pytest` should be listed as an optional extra (`coola[testing]`) or the import should
be guarded.

### 2.16 Naming (low)

_(Status: fixed. (1) The `torch_numpy` module docstring, the three `check_torch_numpy` / `is_torch_numpy_available` / `torch_numpy_available` docstrings, the `coola.testing.fixtures` module docstring and the testing user guide now say that these check a pair of packages (or a feature of PyTorch, for `torch_cuda_*` and `torch_mps_*`), not one package. (2) `utils/imports/torch_numpy.py` no longer imports `torch` at module scope: it is imported inside `is_torch_numpy_available`, after the cheap checks. `import coola.utils.imports` imports neither `torch` nor `numpy` anymore (it imported both before); tests run a fresh interpreter to check it.)_

- `torch_numpy_*` fixtures and `is_torch_numpy_available` are named after the pair,
  while all the others name one package. Fine, but add a note in the docstring.
- `utils/imports/torch_numpy.py` imports `torch` at module scope (via the fallback
  alias) to probe `.numpy()`, so importing `coola.utils.imports` imports Torch if it
  is installed. That undermines the lazy-import intent. Probe inside the function.

---

## 3. Performance

- **Summary at the depth limit stringifies the whole collection.**
  `BaseCollectionSummarizer.summarize` runs `text = str(data)` before checking
  whether the collection is longer than the preview limit, and then discards `text`
  for large collections. For a 2M-element list this costs real time and memory.
  Compute `str(data)` only when `len(data) <= preview_limit`. (My quick timing of a 2M
  list at `max_depth=0` was ~0.06 s, so this matters mainly for huge or deeply nested
  data, but the fix is two lines.)
  _(Status: fixed in #1999.)_
- **`SequenceSameValuesHandler` / `MappingSameValuesHandler`** allocate a `cache` dict
  per call, even for short lists of scalars where nothing can repeat. Skip the cache
  for `len < 2`, or drop it (see 1.3). For lists of floats/ints, a fast path
  (`actual == expected` on the whole list when both contain only built-in scalars and
  no tolerance is set) would avoid per-element dispatch.
  _(Status: partly fixed in #2000. The per-call `cache` no longer exists. `SequenceSameValuesHandler` now has a fast path for lists/tuples of exact scalars (`int`, `str`, `bytes`, `bool`, `None`); floats are excluded because of NaN and tolerance. `MappingSameValuesHandler` is unchanged.)_
- **Per-element dispatch**: `registry.objects_are_equal` calls `type(actual)`, resolves
  through `TypeRegistry.resolve`, and runs a 3–4 handler chain per scalar. The
  `tests/benchmarks` folder exists; add a benchmark for `list[float]` of 100k items so
  a fast path can be measured.
  _(Status: benchmark added in #2001; the dispatch itself is unchanged.)_
- **Pandas/Polars handlers** run `isna().any().any()` and `is_nan()` scans before the
  actual comparison. For large frames that is an extra full pass. When `equal_nan` is
  `False`, `assert_frame_equal` already reports NaN differences if the other frame
  has no NaN, but not when both contain NaN at the same place, so the pre-scan is
  needed. Consider doing it once on `df1` only when `df1 is not df2` and skipping
  when the shapes already differ (cheap checks first).
  _(Status: fixed in #2002. Shapes are checked before the NaN scan; the scan already ran on the first operand only.)_
- **`LazyFrame` handler** calls `collect()` on both inputs without schema check.
  Compare `collect_schema()` first to fail fast.
  _(Status: fixed in #2003.)_
- **`TypeRegistry.resolve`** is already cached; no change needed.

---

## 4. Tests

The suite is large (7.5k tests, ~18 s, plus the property-based tests) and covers every line. Status of the improvements:

- ~~**Order/config coupling.**~~ _(Fixed. The tests of `tests/unit/utils/test_timing.py` set the
  level with `caplog.at_level(logging.INFO, logger=...)` in the test. The suite passes with
  `-o log_level=WARNING`, `ERROR` and `DEBUG`. Note that `caplog.set_level` in a fixture does not
  work, because the capture handler of the test is not the one of its fixtures. The guideline is in
  `tests/TESTING_GUIDE.md`.)_
- ~~**Global `log_level = "DEBUG"`**~~ _(Fixed. It was removed from `[tool.pytest.ini_options]`.)_
- ~~**Property-based tests.**~~ _(Implemented for `hash_object` (#2020) and for `objects_are_equal` /
  `objects_are_allclose` (#2022), with Hypothesis (`hypothesis` is in the development dependencies,
  and `tests/properties/conftest.py` skips the tests if it is not installed). They are in
  `tests/properties/` (`strategies.py`, `test_hashing.py` and `test_equality.py`, about 40
  properties each) and run in their own CI workflow, `ci-test-properties.yaml`, called from `ci.yaml`
  for all the supported Python versions.
  Equality: reflexive, symmetric and transitive; strict type checks (`int`, `float`, `bool`,
  `str`/`bytes`, subclasses, `dict`/`OrderedDict`); lengths, keys and differences found at any depth;
  order of sequences, `dict`, `OrderedDict` and sets; `NaN` and `-0.0`; time zones; tolerance
  (`allclose` with zero tolerance behaves like `equal`, follows `math.isclose`, applies inside
  containers, does not relax the type check); unregistered types; the inputs are not modified;
  self-referencing list; `show_difference` does not change the result; equal objects have the same
  hash.
  Hashing: fixed-width lowercase hex, deterministic (also across `PYTHONHASHSEED`), consistent with
  equality, sensitive to the type, the structure and the order, composable from the hashes of the
  children, no silent false equality, scalar subclasses hash like the builtin, and the input is not
  modified.
  **They found a real bug**: the fast path of `SequenceSameValuesHandler` compared the items with
  `==`, so `[False]` and `[0]`, `[1, 2]` and `[True, 2]`, or `[1, "a"]` and `[1.0, "a"]` were equal
  although the types of the items are different. The fast path now also checks the types of the
  items, with unit tests.
  The remaining properties were added afterwards, in the same directory:
  `test_compare.py` (`compare` agrees with `objects_are_equal` and `objects_are_allclose`, an equal
  result has no path and no reason, the path leads to a difference at any depth, the message names
  the path, `assert_objects_equal` / `assert_objects_allclose` raise iff the objects differ with the
  message of `compare`, `compare` does not modify its inputs and does not log),
  `test_nested.py` (`from_flat_dict(to_flat_dict(x)) == x` with any separator, one entry for each
  leaf, a scalar that is also a prefix is a conflict, key collisions; `merge_mappings` is idempotent,
  keeps every key, `first` / `last` keep the first / last value, `raise` raises iff a key has two
  values, `suffix` loses no value, associative for the keys; `flatten_mapping` with `always_prefix`,
  idempotent on a flat mapping, `prefix` loses no value, `raise` raises iff an inner key has two
  values), `test_reducer.py` (the NumPy and PyTorch reducers agree with the native one for floats and
  integers; the mean, the median and the quantiles are bounded and monotonic; the quantiles `0`,
  `1` and `0.5` are the min, the max and the median; independent of the order; the inputs are not
  modified and the results are Python numbers; constant sequences), and `test_registry.py`
  (`TypeRegistry.resolve` returns the first registered class of the MRO, including on a diamond
  hierarchy, and never uses a virtual subclass; stateful tests compare `Registry` and `TypeRegistry`
  with a dictionary after any sequence of `register`, `register_many`, `unregister`, `clear`,
  `get` and `resolve`, which checks that the cache of `resolve` is invalidated). The registry
  properties that the previous review reported as done were not in the repository.)_
- ~~**Differential tests** for the three reducers~~ _(Done: `tests/unit/reducer/test_consistency.py`
  compares them on fixed datasets: even and odd lengths, unsorted input, duplicates, tiny and huge
  magnitudes, and `tests/properties/test_reducer.py` compares them on generated inputs.)_
- ~~**Fuzz `compare()` thread-safety**~~ _(Covered by `tests/integration/equality/test_result.py`:
  concurrent comparisons do not mix their results, and a record logged by another thread during a
  comparison is not dropped.)_
- ~~**Minimal-install CI job**~~ _(Already exists: the `min` job of `ci-test.yaml` installs
  `--no-optional-deps` and runs the unit and integration tests. I also ran the whole suite locally in
  a virtual environment with only the required dependencies: 8,087 passed, 2,851 skipped, and one
  failure that needs `ruff`, a development tool, to be installed.)_
- **Mutation testing** (e.g. `mutmut` on `equality/handler`): not done. It is a one-off run that is
  worth doing before a release: 100% line coverage does not show that the assertions are strong.
- ~~`tests/package_checks.py` and `tests/TESTING_GUIDE.md`~~ _(`CONTRIBUTING.md` now links to the
  testing guide.)_

---

## 5. Tooling and configuration

`pyproject.toml`:

- ~~`[tool.ruff.lint.fixable]` / `unfixable` / `exclude`~~ _(Fixed: removed, about 70 lines. They
  were the defaults, and the set of linted files is unchanged.)_
- ~~`[tool.ruff.lint.mccabe] max-complexity = 10`~~ _(Fixed: `C90` is now in `select`, so the limit
  is enforced. `flatten_mapping` and `merge_mappings` were too complex (12); small helpers were
  extracted.)_
- ~~`known-first-party = ["src"]`~~ _(Fixed: `["coola"]`.)_
- ~~`pythonpath = ["src"]`~~ _(Fixed: removed, the tests run against the installed package. The
  `min` and `all` CI jobs install the project.)_
- **`fail_under` in `[tool.coverage.report]`**: not changed. A global value would fail the `min` CI
  job (no optional dependency installed, most tests skipped) and a unit-only run. Enforce the
  threshold with `--cov-fail-under` on the one job that runs the full suite (`all`), or rely on
  Codecov.
- **Second type checker** (`pyright` / `mypy`): not done. It needs a decision on a new, non-blocking
  CI job, and its output on the optional-dependency code has not been looked at.
- **import-linter exception `coola.registry.base -> coola.equality.interface`**: kept, by decision.
  `BaseRegistry.equal` has to compare the registered values with the full equality machinery (nested
  values, `equal_nan`, objects with an `equal` method), and a local replacement would duplicate it.
  The import is deferred and documented in the contract.
- ~~`[project.optional-dependencies]` has no `all` extra~~ _(It exists: the `autoextras` hook
  generates it, and the README and the get started page document `coola[all]`.)_
- **`Programming Language :: Python :: 3.15` classifier**: not changed. The Python versions tested by
  the CI come from the `workflow-config-action` and not from a file of the repository, so they cannot
  be checked here. (`.github/GUIDELINES.md` also mentions a `dev/config/test_matrix.json` that does
  not exist.)

Repository layout:

- ~~`debug/`~~ _(Fixed: `debug/equality_examples.py` is now `examples/equality.py`, and
  `tests/integration/test_examples.py` runs every script of `examples/` in a fresh interpreter, so
  they cannot rot. `dev/` is kept: it is the version-pinning script and its config.)_
- `.gitignore` already ignores `coverage.xml` and `htmlcov`.
- ~~Catalogue of the workflows~~ _(Fixed: `.github/GUIDELINES.md` has a table with the trigger and the
  purpose of each of the 34 workflows.)_

---

## 6. Documentation

- ~~Add a "how to add a new backend" page (previous review 4.3, still valid). The checklist is:
  availability probe in `utils/imports`, fallback in `utils/fallback`, handler, tester,
  registration in `tester/interface.py`, then `summary`, `hashing`, `recursive`, `iterator` as
  applicable. Right now it is easy to add equality support and forget the other four registries (see
  2.9).~~ _(Fixed: `docs/docs/dev/add_backend.md`, in the Developer Guide. It lists the files for the
  dependency, the equality support and the tests, and a table with the decision to make for the
  summary, hashing, recursive, iterator, random and reducer packages.)_
- ~~Document the semantic contract of `equal_nan`, `atol`, `rtol` per backend in one table.~~
  _(Fixed: "Behavior by Type: NaN and Tolerance" in the equality user guide. The table was checked
  against the code. The notes were corrected: pandas does not ignore `equal_nan=False`, it counts any
  missing value (`NaN`, `None`, `NaT`, `pd.NA`) as `NaN`, so a data frame with a missing value is not
  equal to its own copy; polars `null` is always equal to `null`; PyArrow ignores the three arguments
  and warns, and Python's default filter shows that warning once per location, so no code change was
  needed; NumPy applies `equal_nan` only to the dtypes that have a `NaN`, and the tolerance only to
  numeric dtypes.)_
- ~~Document that `compare()` and `show_difference=True` are for debugging and not thread-safe
  (1.8).~~ _(Fixed, and the premise changed: since 1.8, `compare()` and the assert functions are
  thread-safe. The new section "Explaining a Difference" of the equality user guide says so, explains
  that `show_difference=True` logs at the INFO level, that a custom handler must call
  `report_difference` to be explained, and that `EqualityConfig` must not be shared between
  threads.)_
- ~~State clearly in the hashing docs which types are supported out of the box and that `set`,
  NumPy, Torch and pandas objects need a registered hasher.~~ _(Fixed: "Supported Types" in the
  hashing user guide, with a NumPy hasher example. It also says that values of different types do not
  collide and that the hash values can change between versions. The outdated `RuntimeError` and the
  `str` in the `SequenceHasher` description were corrected.)_
- ~~`utils/` contains modules unrelated to comparison (`password`, `git`, `date`, `file_size`,
  `text_diff`, `bloom_filter`, `timing`). If they stay, list them in `docs/` under "General helpers"
  so they are not mistaken for core API.~~ _(Fixed: "Core helpers and general helpers" in the utils
  user guide, with a table of the standalone modules.)_

---

## 7. Suggested order of work

Everything is done unless the status says otherwise. Open items are in **bold**.

| #    | Item                                                             | Status                                                                                                                                                 |
| ---- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1.1  | Type-tag hash inputs (collisions across types, NumPy repr, Path) | Fixed (`Path` still resolved)                                                                                                                          |
| 1.2  | Pydantic hasher ignores secrets inside tuples/sets               | Fixed                                                                                                                                                  |
| 1.3  | Remove the `id()` caches                                         | Fixed                                                                                                                                                  |
| 1.4  | NumPy string arrays + `equal_nan` raise                          | Fixed                                                                                                                                                  |
| 1.5  | Scalar tolerance on huge ints                                    | Fixed                                                                                                                                                  |
| 1.6  | Flat dict key collisions and lost empty containers               | Fixed                                                                                                                                                  |
| 1.7  | Torch median disagrees with Native/NumPy; float32 stats          | Fixed                                                                                                                                                  |
| 1.8  | Replace logger mutation in `compare()` with a reporter           | Fixed                                                                                                                                                  |
| 1.9  | `_compare` RecursionError rewrite                                | Fixed                                                                                                                                                  |
| 1.10 | Validators reject NaN; assorted small fixes                      | Fixed (`to_date` kept)                                                                                                                                 |
| 2.1  | `ImportError` for missing optional packages                      | Fixed                                                                                                                                                  |
| 2.2  | Availability checks detect broken installs; cache reset          | Fixed                                                                                                                                                  |
| 2.3  | Collapse tester boilerplate                                      | Fixed                                                                                                                                                  |
| 2.4  | `is_same_type` for the `equal()` checks                          | Fixed                                                                                                                                                  |
| 2.5  | Registry duplicate-key semantics                                 | Fixed (no `issubclass` fallback, by decision)                                                                                                          |
| 2.6  | `get_default_*_registry` aliases                                 | Kept, by decision                                                                                                                                      |
| 2.7  | Package-level exports                                            | Kept, by decision                                                                                                                                      |
| 2.8  | `KeyFilterTransformer` signature                                 | Fixed                                                                                                                                                  |
| 2.9  | Summarizers for pandas/polars/pyarrow/xarray/JAX                 | Fixed (**no hashers**, by decision)                                                                                                                    |
| 2.10 | BFS/DFS iterators: leaves, `is_leaf`, cycles                     | Fixed                                                                                                                                                  |
| 2.11 | `random` package                                                 | Fixed (NumPy seed folding documented)                                                                                                                  |
| 2.12 | `io` package                                                     | Fixed                                                                                                                                                  |
| 2.13 | `allowed_prefixes` on `resolve_object`, security notes           | Fixed                                                                                                                                                  |
| 2.14 | Identifiers: fork safety, monotonic ULID/UUIDv7                  | Fixed                                                                                                                                                  |
| 2.15 | `testing.fixtures` repetition                                    | Fixed                                                                                                                                                  |
| 2.16 | Naming and lazy `torch` import                                   | Fixed                                                                                                                                                  |
| 3    | Performance                                                      | Fixed or benchmarked (#1999 to #2003)                                                                                                                  |
| 4    | Tests                                                            | Fixed; property-based tests implemented (hashing, equality, `compare`, `nested`, reducers, registries; they found a bug), **mutation testing not run** |
| 5    | Tooling and configuration                                        | Fixed, **`fail_under` and a second type checker not done**                                                                                             |
| 6    | Documentation                                                    | Fixed                                                                                                                                                  |
