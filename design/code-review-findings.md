# `coola` Code Review Findings

Scope: `src/coola` (read-only review). Tests in `tests/` were spot-checked for
context on intended behavior, not audited exhaustively.

Overall impression: the codebase is unusually well documented (every public
method has a docstring with a runnable `pycon` example) and follows a
consistent architecture — `TypeRegistry`-backed dispatch registries feeding
chain-of-responsibility handlers/testers for equality, hashing, recursive
transformation, summarization and BFS/DFS iteration. The concerns below are
mostly about hardening edge cases, trimming duplication across the five
parallel registry implementations, and a few real correctness/security
footguns rather than fundamental design problems.

---

## 1. Correctness Risks / Bugs

### High

- **FIXED** — **`SequenceSameValuesHandler` silently accepted sequences of
  different length as equal** — `src/coola/equality/handler/sequence.py:45-63`.
  The handler zipped `actual`/`expected` and only compared up to the shorter
  sequence's length; it never checked `len(actual) == len(expected)` itself,
  even though its docstring said "If the sequences have different length,
  this handler checks only the values of the shortest sequence." This meant
  correctness for sequence types depended entirely on `SequenceSameValuesHandler`
  always being chained after a `SameLengthHandler` in every tester that used
  it (e.g. `SequenceEqualityTester`) — a foot-gun for anyone registering a
  custom tester or reusing this handler standalone. `handle` now checks
  `len(actual) == len(expected)` itself as defense-in-depth (returning
  `False`, with a difference log line when `config.show_difference` is set,
  before ever touching `next_handler`) and the docstring/doctest were updated
  to document the new, correct behavior. Covered by
  `tests/unit/equality/handler/test_sequence.py`: the previous tests that
  asserted `True` for mismatched-length inputs were replaced with tests
  asserting `False`, plus new tests for the difference-logging message, that
  the next handler is not invoked on a length mismatch, and that a length
  mismatch is caught even with no next handler configured.

- **`MappingSameValuesHandler` assumes matching keys** —
  `src/coola/equality/handler/mapping.py:88-100`. Docstring says "This handler
  assumes that all the keys in the first mapping are also in the second
  mapping," but `handle` does `expected[key]` without a `.get`/try-except.
  If this handler is ever used without `MappingSameKeysHandler` first (e.g. a
  future custom tester composes handlers differently, or someone calls it
  directly per its own doctest promise of being usable standalone), a missing
  key raises `KeyError` instead of returning `False`/a sensible equality
  result. Same category of implicit-ordering risk as the sequence handler
  above.

### Medium

- **`TypeRegistry` cache can go stale for ABC virtual subclasses registered
  after resolution** — `src/coola/registry/type.py:117-243`. `resolve()`
  caches by `dtype` and the cache is only invalidated by `_on_change()` on
  `register`/`unregister`/`register_many`/`clear`. If a class is registered
  with `abc.ABCMeta.register()` (virtual subclass, not real inheritance) into
  an ABC that's already an MRO parent used by the registry, `__mro__` walk in
  `_resolve_uncached` won't discover the new relationship anyway (virtual
  subclasses don't appear in `__mro__`), so results are consistent — but it's
  worth an explicit docstring caveat, since users reasonably expect an ABC to
  "count" via `isinstance`, not just MRO.

- **`BaseFileSaver.save`'s `exist_ok=True` path is not atomic against
  concurrent writers** — `src/coola/io/base.py:192-260`. The docstring is
  admirably explicit about this ("This TOCTOU guard does not apply when
  `exist_ok=True`"), so it's a documented limitation rather than a silent bug,
  but two concurrent `save(..., exist_ok=True)` calls to the same path can
  interleave (`tmp_path.replace(path)` from each), silently overwriting one
  writer's data with no way to detect it happened. If any coola consumer
  relies on "save wins deterministically," consider a lock file or leaving
  a note that no writer is guaranteed to win with `exist_ok=True`.

- **FIXED** — **`instantiate_object` with `_init_="__new__"` bypasses
  `__init__` but the isinstance check happens only for the returned object,
  not for whether `__init__` runs at all** —
  `src/coola/factory/instantiation.py:120-165`. For `__new__`,
  `obj = init_fn(cls, *args, **kwargs)` never calls `cls.__init__`, silently
  producing a partially-initialized object if the caller expected
  `__new__`-then-`__init__` semantics (which is what plain
  `cls(*args, **kwargs)` would do). This is intentional (an escape hatch),
  so the fix documents rather than changes the behavior: `instantiate_object`
  and `_instantiate_class_object` docstrings now carry an explicit warning
  that `_init_="__new__"` skips `cls.__init__` entirely, unlike the default
  `__init__` path. Covered by
  `tests/unit/factory/test_instantiation.py::test_instantiate_object_class_init_new_bypasses_init`
  and `::test_instantiate_object_class_init_new_vs_default_init`, which assert
  that the `__new__` path leaves the instance `__dict__` empty while the
  default path populates it as expected.

- **`NativeReducer._std` special-cases length 1 to return `nan` instead of
  raising** — `src/coola/reducer/native.py:57-60`, contrasted with
  `TorchReducer`/`NumpyReducer` (not read in this pass, but referenced in
  `BaseReducer` docstrings as raising `EmptySequenceError` only for *empty*
  input). Confirm the three reducer implementations agree on the length-1
  standard-deviation behavior (`statistics.stdev` raises
  `StatisticsError` for `n < 2`, hence the special-case here) — if
  `NumpyReducer`/`TorchReducer` return `0.0` or `nan` differently for a
  single-element input, that's a cross-backend inconsistency that violates
  the implicit contract that all `BaseReducer` implementations are
  interchangeable.

### Low

- **`BloomFilter.add_and_check`'s double-hashing derives both `h1`/`h2` from
  one SHA-512 digest** — `src/coola/utils/bloom_filter.py:70-93`. This is a
  reasonable, well-documented trade-off (comment explains it), but note
  `_optimal_hash_count` can return an unbounded number of hash rounds for
  very small `n`/large `m` ratios; for `expected_items=1`, `fp_rate` close to
  0, `hash_count` could get large. Not a bug, but worth a sanity cap given
  it's called once per `add_and_check`.

---

## 2. API Design / Consistency Issues

- **Five near-identical registry wrapper classes** — `EqualityTesterRegistry`
  (`src/coola/equality/tester/registry.py`), `TransformerRegistry`
  (`src/coola/recursive/registry.py`), `HasherRegistry`
  (`src/coola/hashing/registry.py`), plus (per the package listing but not
  read in depth this pass) `summary/registry.py`, `iterator/bfs/registry.py`,
  `iterator/dfs/registry.py`. Each hand-implements `register`,
  `register_many`, `has_<x>`, `find_<x>`, and a `_get_repr_kwargs` that
  wraps an internal `TypeRegistry`. The method bodies are one-line
  delegations to `self._state.*` and the docstrings are near-verbatim copies
  with only the noun swapped (compare
  `src/coola/equality/tester/registry.py:81-156` to
  `src/coola/hashing/registry.py:82-154` to
  `src/coola/recursive/registry.py:89-164`). This is a strong candidate for a
  shared generic base (e.g. `TypedDispatchRegistry[V]` in `coola/registry`)
  that these six classes subclass or compose, exposing `register`,
  `register_many`, `has`, `find`, leaving only the type-specific entry point
  (`objects_are_equal`, `transform`, `hash`, `summarize`, iterate) to the
  subclass. This would cut ~500-700 lines of duplicated docstring/boilerplate
  and centralize any future behavior change (e.g. adding an LRU cache
  consistently — see next point).

- **FIXED** — **Inconsistent caching strategy across registries doing the
  same MRO lookup**: `TypeRegistry.resolve()` (`src/coola/registry/type.py`)
  uses a plain dict cache guarded by the registry's own lock, but
  `EqualityTesterRegistry.find_equality_tester`'s docstring claimed "Results
  are cached using an LRU cache (256 entries)" — yet the implementation just
  calls `self._state.resolve(data_type)`, which is the *unbounded* dict
  cache in `TypeRegistry`, not an LRU. The stale docstring was rewritten to
  describe the actual (unbounded, per-instance) internal cache instead of
  claiming a bounded LRU that doesn't exist, in both
  `src/coola/equality/tester/registry.py` (class docstring) and the
  `TypeRegistry` module itself. (Actually adding a bounded/LRU cache is left
  as a separate follow-up — see the performance section below — since it
  would change eviction behavior, not just documentation.)

- **FIXED** — **Mixed `_getitem_not_registered_msg` design led to
  inconsistent `KeyError` wording between `__getitem__`, `.resolve()` and
  `.unregister()`** — `TypeRegistry` previously produced three different
  strings for the same "lookup miss" condition: `"Type {key} is not
  registered"` (no quotes, from `_not_registered_msg`, used by
  `unregister`), `"Type '{key}' is not registered"` (quoted, from
  `_getitem_not_registered_msg`, used by `__getitem__`), and `"Could not
  find a registered type for {dtype}"` (a third, hardcoded message in
  `_resolve_uncached`). `TypeRegistry` now defines a single
  `_not_registered_msg` (`"Type '{key}' is not registered"`) and both
  `_resolve_uncached` and `unregister`/`__getitem__` (via the inherited
  `BaseRegistry._getitem_not_registered_msg`, which delegates to
  `_not_registered_msg`) raise `KeyError` with that same wording; the
  now-redundant `_getitem_not_registered_msg` override was removed. Covered
  by a new test,
  `test_type_registry_not_registered_message_is_consistent` in
  `tests/unit/registry/test_type.py`, which asserts `resolve`, `__getitem__`
  and `unregister` all raise `KeyError` with identical text for the same
  missing type; the existing tests for each of the three call sites were
  updated to match the unified message.

- **`register_many`'s "atomic" claim is per-registry, not cross-registry** —
  the docstrings (e.g. `src/coola/registry/base.py:273-326`) call the
  operation atomic when `exist_ok=False`, which is true for the underlying
  dict mutation, but `_on_change()` is invoked exactly once after the bulk
  `dict.update`, same as `register`. That's fine and consistent — just flag
  that "atomic" here specifically means "no error occurs after partial
  mutation," not thread-isolation across the whole call (a concurrent reader
  could still observe a state where some but not all new keys are visible
  mid-`update`, though CPython's GIL makes `dict.update` itself atomic in
  practice). Consider clarifying the docstring's atomicity claim to be
  precise about which guarantee is meant.

- **`EqualityConfig` documents itself as "not thread-safe"** and correctly
  recommends one instance per comparison (`src/coola/equality/config.py:29-36`),
  but `objects_are_equal`/`objects_are_allclose`
  (`src/coola/equality/interface.py:16-129`) accept a shared, possibly
  externally-constructed `registry: EqualityTesterRegistry` while
  constructing a fresh `EqualityConfig` per call — good — but nothing
  prevents a caller from passing the *same* `EqualityConfig` instance into
  two concurrent top-level calls (the API doesn't accept a pre-built config
  at all today, which actually protects against this). Just flag as a
  design constraint worth keeping in mind if a future PR adds a `config=`
  parameter to the public functions.

- **`resolve_object`'s "any `dict` subclass is treated as config" caveat**
  (`src/coola/factory/resolve.py:71-77`) is a sharp edge silently baked into
  behavior: `resolve_object(Counter(...), cls=Counter)` would try to treat the
  `Counter` instance as a factory config dict and fail looking for
  `_target_`. The docstring calls this out, which is good, but the function
  doesn't raise a more specific/actionable error in that exact case — the
  resulting `TypeError` ("missing the `_target_` key") could be confusing
  when the *actual* problem is "you don't need to resolve this, it's already
  an instance." Consider checking `isinstance(obj, cls) and isinstance(obj,
  dict) and OBJECT_TARGET not in obj` and giving a more specific error hint.

---

## 3. Code Duplication / Reuse Opportunities

- **Registry boilerplate** — see section 2's first bullet; this is the
  single largest duplication opportunity in the package (six structurally
  identical wrapper classes around `TypeRegistry`).

- **`SupportsAllCloseNan` (`src/coola/equality/handler/allclose.py:20-43`)
  and `SupportsTolerantEqual` (`src/coola/equality/handler/tolerant.py:21-59`)
  Protocols duplicate the `allclose` method signature verbatim** (both
  declare the same `allclose(self, other, rtol=1e-5, atol=1e-8,
  equal_nan=False) -> bool`). `SupportsTolerantEqual` could simply extend
  `SupportsAllCloseNan` and add the `equal` method, avoiding the copy-pasted
  signature and docstring.

- **`hasattr(x, "method") and callable(x.method)` guard pattern repeated** —
  `src/coola/equality/handler/allclose.py:91`,
  `src/coola/equality/handler/tolerant.py:122-127`. A tiny shared helper
  (`_supports(obj, *method_names)`) in `coola.equality.handler.utils` would
  remove the duplicated guard and make it easy to extend if a third handler
  needs the same check.

- **`_get_repr_kwargs` returning `{}`** appears in many leaf classes purely
  to satisfy the `BaseDisplayMixin` abstract method contract (e.g.
  `PickleLoader._get_repr_kwargs` in `src/coola/io/pickle.py:37-38`,
  `DefaultTransformer._get_repr_kwargs` in
  `src/coola/recursive/default.py:55-56`, `SequenceHasher._get_repr_kwargs`
  in `src/coola/hashing/sequence.py:42-43`). Since this is the common case
  (stateless handler/hasher/transformer with no constructor args), consider
  giving `BaseDisplayMixin` (or a new `NoArgsDisplayMixin`) a default
  `_get_repr_kwargs` returning `{}`, and only requiring the override when a
  class actually has state — this removes a large number of trivial,
  content-free method bodies across the codebase.

- **Type-lookup docstring/example blocks are copy-pasted nearly verbatim**
  across `find_equality_tester`, `find_transformer`, `find_hasher` (see
  section 2). Beyond the code itself, this means any correction to the
  behavioral description (e.g. the stale "LRU cache (256 entries)" claim
  flagged above) has to be hunted down and fixed in multiple places — which
  is presumably how the inconsistency was introduced.

---

## 4. Performance Concerns

- **`TypeRegistry.resolve()`'s MRO walk is `O(len(mro))` per cache miss and
  the cache is unbounded** (`src/coola/registry/type.py:209-242`). For the
  package's steady-state use (a fixed, small set of registered types), this
  is fine. It becomes a concern only if a consumer calls `objects_are_equal`/
  `hash`/`transform` on many distinct dynamically-generated classes (e.g.
  pydantic models created per-request, or `NamedTuple`s created in a loop) —
  each distinct type grows the cache forever with no eviction, which is a
  slow, unbounded memory leak in long-running processes. Given the
  docstring in `EqualityTesterRegistry.find_equality_tester` (section 2)
  already claims an "LRU cache (256 entries)" was intended, adding a real
  bounded cache (or exposing a way to clear/bound it) would fix both the
  documentation mismatch and this leak risk.

- **`SequenceHasher.hash` and `SequenceSameValuesHandler`/
  `MappingSameValuesHandler` recurse through `registry.hash`/
  `config.registry.objects_are_equal` per element with no short-circuit
  reuse of already-computed hashes for repeated/interned values** — for
  workloads with highly repetitive nested structures (e.g. many identical
  sub-trees), there's no memoization keyed by `id()`/structural hash, so
  identical sub-structures are re-hashed/re-compared repeatedly. This is a
  reasonable trade-off for a general-purpose library (memoization requires
  either accepting `id()`-based caching pitfalls with mutable objects, or
  extra bookkeeping) but worth noting as a potential opportunity if hashing
  large nested configs becomes a hot path.

- **`BaseRegistry.items()/keys()/values()` all take a full `dict.copy()`
  under the lock on every call** (`src/coola/registry/base.py:367-419`).
  This is the right trade-off for thread-safety (avoids returning a live
  view that could be mutated concurrently while iterated), but for
  registries queried in hot loops (e.g. inside a comparison of many objects)
  this is an O(n) allocation per call rather than O(1). If any of these
  registries end up on a hot path outside of setup/registration time, this
  is worth revisiting — as-is, it looks like registries are populated once
  at import/config time and read via `resolve`/`get`, which don't copy, so
  it's likely fine in practice.

- **`BloomFilter._hashes` computes a fresh SHA-512 digest per call** —
  `src/coola/utils/bloom_filter.py:70-93` — appropriate for its stated use
  case (approximate duplicate detection over documents), not a concern at
  the intended data volumes, but SHA-512 is heavier than necessary purely
  for a non-cryptographic bloom filter; a faster non-cryptographic hash
  (e.g. xxhash/murmur, if an optional dependency is acceptable) would reduce
  CPU cost for very large corpora. Low priority given no dependency is
  currently required for this pure-stdlib implementation.

---

## 5. Type Hints / Documentation Gaps

- **`AllCloseNanHandler.handle`'s signature types `actual` as
  `SupportsAllCloseNan` but the method itself checks `hasattr(actual,
  "allclose")` before trusting that** (`src/coola/equality/handler/allclose.py:90-99`).
  This is good defensive coding, but it means the type hint is aspirational/
  not load-bearing — a static type checker will happily accept a
  `SupportsAllCloseNan`-typed argument reaching this handler, but the
  runtime code path exists specifically to guard against the *opposite*
  case (an arbitrary object without `allclose`). Consider documenting in the
  class docstring that the type hint documents the "happy path" contract but
  the implementation is deliberately defensive against arbitrary inputs
  reaching it via the dispatch registry (which resolves by `type(actual)`,
  not by protocol conformance).

- **`BaseEqualityHandler.handle`'s docstring explicitly documents why every
  handler keeps the full `(actual, expected, config)` signature even when
  unused** (`src/coola/equality/handler/base.py:83-90`) — this is exemplary;
  called out here as a model for how the "noqa: ARG002 everywhere" pattern
  elsewhere in the file (e.g. `FalseHandler.handle`,
  `src/coola/equality/handler/native.py:53-59`) should be justified. No
  action needed, just noting the consistency is good and should be
  preserved as new handlers are added.

- **Public functions accept `object` for `actual`/`expected` almost
  everywhere** (e.g. `objects_are_equal(actual: object, expected: object,
  ...)` in `src/coola/equality/interface.py:83-91`), which is correct but
  means callers get no type-level help distinguishing "compatible" from
  "incompatible" types — inherent to the domain (comparing arbitrary
  objects), not a gap to fix, just noting that documentation (not typing) is
  doing all the work of communicating behavior here, which raises the bar
  for keeping docstrings accurate (see the stale LRU-cache claim above as a
  concrete instance of docs drifting from code).

- **`coola/__init__.py` exposes only `__version__`** (`src/coola/__init__.py:8`)
  — the top-level package docstring says "Use this package to compare nested
  objects, summarize complex structures..." but none of `objects_are_equal`,
  `objects_are_allclose`, or similarly central entry points are re-exported
  at the top level; users must know to import from `coola.equality`,
  `coola.hashing`, `coola.recursive`, etc. This is a legitimate design
  choice (avoids import-time cost / circular-import risk given the
  lazy-registry pattern used throughout), but the top-level docstring's
  framing ("Use this package to compare nested objects...") reads as if
  `import coola; coola.objects_are_equal(...)` should work, which it
  doesn't. Either adjust the docstring to point explicitly at the
  submodules, or add deliberate lazy re-exports (e.g. via `__getattr__` in
  `__init__.py`) if ergonomics matter more than import-time cost.

- **`EqualityConfig.__post_init__` validates `atol`/`rtol`/`max_depth` but
  not `equal_nan`/`show_difference`/`registry` types** — reasonable, since
  those are simple/duck-typed, but note the *type* of `registry` isn't
  checked at all — passing a non-`EqualityTesterRegistry` object with a
  compatible-looking `objects_are_equal` method would work by duck typing
  (arguably a feature, not a bug), but passing something entirely wrong
  produces a late, possibly confusing `AttributeError` deep in a handler
  rather than an immediate, clear error at `EqualityConfig` construction.

---

## 6. Error Handling

- **`import_object`/`factory`/`instantiate_object`/`resolve_object`
  (`src/coola/factory/instantiation.py`, `src/coola/factory/resolve.py`) form
  a dynamic-import-and-call pipeline that will import and execute arbitrary
  module code and instantiate arbitrary classes from a string path.** This
  is a deliberate, Hydra-style design (used by `coola.io`'s
  `is_loader_config`/`resolve_loader` to build loaders/savers from `dict`
  configs — `src/coola/io/base.py:273-373`). There is no allowlist or
  restriction on which modules/classes can be targeted. If any of these
  entry points (`factory()`, `resolve_object()`, `resolve_loader()`,
  `resolve_saver()`) can ever be reached with a `_target_` string derived
  from untrusted input (e.g. a config file uploaded by a third party, or a
  network payload), this is a remote-code-execution vector. Worth an
  explicit **Security** note in the public docstrings of `factory`,
  `resolve_object`, `resolve_loader`, and `resolve_saver` warning that
  `_target_` must come from a trusted source, mirroring what many similar
  libraries (Hydra, OmegaConf) document prominently.

- **`PickleLoader.load` uses `pickle.load` on arbitrary file paths**
  (`src/coola/io/pickle.py:43-45`), suppressed with `# noqa: S301` (the
  bandit/ruff rule for exactly this risk) but with no docstring-level
  warning to *callers* of `PickleLoader`/`load_pickle` that loading a pickle
  file from an untrusted source can execute arbitrary code during
  unpickling. The `# noqa` silences the linter but doesn't communicate the
  risk to library users reading the API docs. Recommend adding an explicit
  "Warning: only load pickle files from trusted sources" note to
  `PickleLoader`'s and `load_pickle`'s public docstrings
  (`src/coola/io/pickle.py:18-35`, `86-110`), not just the loader itself —
  the loader is one of the friendliest public entry points in `coola.io` and
  most likely to be reached by end users skimming examples rather than
  internals.

- **`AllCloseNanHandler.handle` and `TolerantEqualHandler.handle` swallow
  any exception `actual.allclose(...)`/`actual.equal(...)` might raise** —
  they don't; on closer look there's no try/except around those calls
  (`src/coola/equality/handler/allclose.py:90-99`,
  `src/coola/equality/handler/tolerant.py:119-139`), so a buggy user-defined
  `allclose`/`equal` implementation that raises will propagate up through
  `objects_are_equal` uncaught. That's arguably correct behavior (fail
  loudly rather than silently returning `False`), but it's worth confirming
  this is the intended contract and documenting it — currently neither
  handler's docstring says what happens if the delegated method raises.

- **`BaseFileSaver.save`'s cleanup path can itself raise and mask the
  original exception** — `src/coola/io/base.py:245-260`: on failure inside
  the `try` block, the `except BaseException: tmp_path.unlink(missing_ok=True);
  raise` re-raises the original exception, which is correct, but the
  `os.link`/`finally: tmp_path.unlink(...)` block above it (lines 253-257)
  means if `os.link` raises `FileExistsError` (the expected TOCTOU case) the
  `finally` unlinks `tmp_path` and the `FileExistsError` propagates — good.
  But if `_save_file` itself raises, the outer `except BaseException` at
  line 258 unlinks `tmp_path` *again* (already handled by the same path) —
  actually on inspection this is fine since `_save_file`'s failure happens
  before `os.link`/`tmp_path.replace` is attempted, and the `unlink` call
  uses `missing_ok=True`, so double-unlink attempts are harmless. No actual
  bug — flagged only because the nested try/finally-inside-try/except is
  dense enough to warrant a comment mapping each failure mode to its
  handler, for future maintainers.

- **`TypeRegistry.resolve()` raises `KeyError` with a message built from
  `dtype` — `f"Could not find a registered type for {dtype}"` (`type.py:241`)
  — but callers like `HasherRegistry.hash` catch it narrowly
  (`except KeyError:` at `src/coola/hashing/registry.py:254-259`) to
  implement `ignore_unhashable`.** If `TypeRegistry.resolve()`'s internals
  (or a future refactor) ever raise `KeyError` for an unrelated reason (e.g.
  a bug indexing into `self._state`), `HasherRegistry.hash` would silently
  swallow it into the `ignore_unhashable` placeholder path instead of
  surfacing the real bug. Consider a dedicated exception type (e.g.
  `TypeNotRegisteredError(KeyError)`) so call sites can catch precisely the
  "no registration found" case without also catching unrelated `KeyError`s
  from implementation bugs.

---

## 7. Test Coverage Gaps (spot check)

File-count parity between `src/coola/<pkg>` and `tests/unit/<pkg>` is good
overall (see table below), which suggests decent breadth, but file-count
parity doesn't guarantee the *interesting* branches are covered:

| package | src files | test files |
|---|---|---|
| equality/handler | 21 | 21 |
| equality/tester | 17 | 16 |
| hashing | 14 | 13 |
| io | 7 | 8 |
| reducer | 6 | 5 |
| recursive | 11 | 10 |
| registry | 4 | 4 |
| identifier | 13 | 14 |
| iterator | 17 | 17 |
| factory | 5 | 4 |
| summary | 11 | 10 |
| random | 7 | 6 |

Specific gaps worth checking directly (not confirmed absent, but not seen in
this pass and worth a targeted look given the findings above):

- **Concurrent-mutation tests for `BaseRegistry`** — the class is explicitly
  documented as thread-safe with `RLock`, but a review of `tests/unit/registry`
  for actual multithreaded stress tests (e.g. many threads calling
  `register`/`unregister`/`resolve` concurrently) would validate the
  concurrency claims rather than only the single-threaded API surface.
- **FIXED** — **`SequenceSameValuesHandler` used standalone with mismatched
  lengths and no preceding `SameLengthHandler`** — the handler now checks the
  length itself (see §1), so this is covered directly rather than only via
  `SequenceEqualityTester`'s chain; regression tests in
  `tests/unit/equality/handler/test_sequence.py` assert `False` for
  mismatched-length inputs used standalone.
- **`TypeRegistry.resolve()` cache invalidation across `register_many` with
  partial overlap and `exist_ok=True`** — confirm a test exercises
  re-resolving a type after its registration is overwritten via
  `register_many(..., exist_ok=True)`, not just via `register`.
  `_on_change()` is called in both paths per the code, but this is a natural
  seam for a regression to slip through unnoticed.
- **`BaseFileSaver.save` concurrent-writer race with `exist_ok=True`** — the
  docstring documents the exact race condition; a test simulating it (two
  savers targeting the same path) would at least document the accepted
  behavior in an executable form, even if the "fix" is just "last writer
  wins."
- **`instantiate_object`/`factory` with malicious-looking `_target_` (e.g.
  targeting `os.system`, `eval`, dunder attribute traversal like
  `"builtins.eval"`)** — given the RCE-adjacent design flagged in section 6,
  tests that document current behavior (it *will* successfully resolve and
  call `os.system` if given the chance) help future maintainers understand
  this is accepted, known behavior rather than an oversight, and make any
  future decision to add an allowlist a deliberate, tested change.
- **`get_password`'s `confirm=True` mismatch path** and **non-interactive
  terminal path** in `src/coola/utils/password.py` are inherently hard to
  unit test (docstring says as much — "no doctest example because it
  requires interactive terminal input"); confirm `tests/unit/utils` mocks
  `sys.stdin.isatty`/`getpass.getpass` to cover the `RuntimeError` and
  mismatched-confirmation `ValueError` branches, since these are exactly the
  kind of branches that are easy to skip when a doctest can't cover them.

---

## 8. Architectural Observations

- **Consistent "TypeRegistry + chain-of-responsibility" pattern across
  equality, hashing, recursive transformation, and (per the file listing)
  summary/iterator modules** is the strongest architectural asset here: new
  types are supported by registering a tester/hasher/transformer without
  touching dispatch logic, and MRO-based fallback means partial coverage
  (e.g. registering `Sequence` instead of every concrete sequence type)
  works out of the box. This pattern is applied uniformly enough that
  extracting the shared registry logic (section 2/3) is low-risk and would
  pay for itself quickly.

- **Deliberate, well-commented avoidance of circular imports** — e.g.
  `EqualityConfig`'s `create_default_registry()` factory function doing a
  local import (`src/coola/equality/config.py:14-26`), and
  `EqualityTesterRegistry.__init__`'s local import of `TypeRegistry`
  (`src/coola/equality/tester/registry.py:70-76`) — both are explained
  in comments. This is good practice, but the number of local,
  comment-justified imports (`# noqa: PLC0415`) scattered through the
  registry/config/tester modules suggests the module dependency graph is
  close to a genuine cycle (`coola.equality` ↔ `coola.registry`). If this
  pattern keeps growing as more registries are added, it may be worth a
  dedicated `coola._internal` or `coola.registry.type` boundary layer that
  every registry depends on one-way, with the type-specific packages never
  imported from `coola.registry` (which appears to already be mostly true —
  worth confirming no exceptions have crept in).

- **`LazySingleton` (`src/coola/utils/singleton.py`) is a clean, purpose-built
  primitive** introduced specifically to support the "each package exposes a
  lazily-built, cached `get_default_registry()`" convention seen in
  `coola.equality.tester.interface`, `coola.hashing` (implied by
  `get_default_registry` used in `src/coola/hashing/registry.py`
  docstrings), and `coola.recursive`. Good example of recognizing a repeated
  pattern and extracting a shared primitive — this is exactly the kind of
  extraction that sections 2/3 recommend doing for the registry wrapper
  classes too.

- **`BaseRegistry`'s `equal()` method takes a consistent-ordering dual lock
  via `id()` comparison to avoid deadlock** (`src/coola/registry/base.py:159-165`)
  — a subtle, correctly-handled concurrency detail that's easy to get wrong;
  called out here as a positive architectural detail worth preserving and
  perhaps documenting as a pattern to reuse if other cross-instance-locking
  methods are added in the future (e.g. a hypothetical `TypeRegistry.merge()`).

- **`coola.io`'s `BaseLoader`/`BaseSaver` registering themselves into the
  *equality* default registry as a side effect of module import**
  (`src/coola/io/base.py:375-377`: `get_default_registry().register_many({BaseLoader:
  EqualNanEqualityTester(), BaseSaver: EqualNanEqualityTester()}, exist_ok=True)`
  at module scope) is a cross-cutting, import-time side effect: merely
  importing `coola.io` mutates global state in `coola.equality`. This works
  today (the `exist_ok=True` avoids errors on re-import/re-registration) but
  it's an implicit coupling that's easy to miss when reasoning about either
  module in isolation, and it means the behavior of `objects_are_equal` for
  `BaseLoader`/`BaseSaver` subclasses silently depends on whether
  `coola.io` happened to be imported yet. Consider documenting this
  cross-module registration explicitly in both modules' docstrings (or
  moving it to an explicit `coola.io.register_equality_testers()` call sites
  can opt into) so it's discoverable rather than a side effect of import
  order.

---

## Summary of Highest-Priority Actions

1. Harden or explicitly test the "must be chained after a length/keys check"
   assumptions in `SequenceSameValuesHandler` (**FIXED** — see §1) and
   `MappingSameValuesHandler` (§1).
2. Extract a shared base for the six `TypeRegistry`-backed dispatch
   registries to eliminate ~500+ lines of duplicated boilerplate/docstrings
   and fix the discovered doc/behavior drift in one place (§2, §3).
3. Reconcile the "LRU cache (256 entries)" docstring claim in
   `EqualityTesterRegistry.find_equality_tester` with the actual unbounded
   `TypeRegistry` cache, and consider bounding the cache to avoid unbounded
   growth with dynamically generated types (§2, §4).
4. Add explicit security warnings to the public docstrings of
   `factory`/`resolve_object`/`resolve_loader`/`resolve_saver` and
   `PickleLoader`/`load_pickle` about executing/deserializing untrusted
   input (§6).
5. Introduce a dedicated "type not registered" exception distinct from bare
   `KeyError` so callers like `HasherRegistry.hash`'s `ignore_unhashable`
   path can't accidentally swallow unrelated bugs (§6).
