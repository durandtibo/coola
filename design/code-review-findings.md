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

- **PARTIALLY ADDRESSED (documented, not fixed)** — **`MappingSameValuesHandler`
  assumes matching keys** — `src/coola/equality/handler/mapping.py`. `handle`
  still does `expected[key]` without a `.get`/try-except, so a missing key
  still raises `KeyError` instead of returning `False` when this handler is
  used standalone/without `MappingSameKeysHandler` preceding it — unlike the
  sibling `SequenceSameValuesHandler` finding above, the underlying behavior
  was **not** changed. What was added is an explicit **Warning** block in the
  docstring spelling out the assumption and the `KeyError` consequence, plus
  identity-based memoization of repeated value-pair comparisons (see §4). The
  defense-in-depth fix applied to `SequenceSameValuesHandler` was not mirrored
  here — worth doing for consistency, or explicitly deciding the documented
  contract is sufficient.

### Medium

- **FIXED** — **`TypeRegistry` cache can go stale for ABC virtual subclasses
  registered after resolution** — `src/coola/registry/type.py:117-243`.
  `resolve()` caches by `dtype` and the cache is only invalidated by
  `_on_change()` on `register`/`unregister`/`register_many`/`clear`. If a
  class is registered with `abc.ABCMeta.register()` (virtual subclass, not
  real inheritance) into an ABC that's already an MRO parent used by the
  registry, `__mro__` walk in `_resolve_uncached` won't discover the new
  relationship anyway (virtual subclasses don't appear in `__mro__`), so
  results are consistent — but it was worth an explicit docstring caveat,
  since users reasonably expect an ABC to "count" via `isinstance`, not just
  MRO. Both `TypeRegistry`'s class docstring and `resolve()`'s docstring now
  carry an explicit **Note** explaining that resolution follows
  `dtype.__mro__` (real, static inheritance) only, that
  `abc.ABCMeta.register()` virtual subclasses never appear there — so
  registering a value for an ABC does not make `resolve()` match its virtual
  subclasses even though `isinstance` would — and that callers must
  register the subclass itself (or a real ancestor) to make it resolve.
  Covered by two new regression tests in `tests/unit/registry/test_type.py`:
  `test_type_registry_resolve_does_not_match_abc_virtual_subclass` and
  `test_type_registry_resolve_abc_virtual_subclass_registered_after_lookup`,
  which assert `resolve()` raises `KeyError` for a virtual subclass both
  before and after the ABC itself is registered (cached or not), and that
  registering the concrete subclass directly fixes resolution.

- **FIXED** — **`BaseFileSaver.save`'s `exist_ok=True` path is not atomic
  against concurrent writers** — `src/coola/io/base.py:192-260`. The
  previous per-path lock (`_get_save_lock`) only serialized `save` calls
  *within this process* via a `threading.Lock`; the docstring documented
  that `exist_ok=True` was not protected against other processes racing
  `tmp_path.replace(path)`. `save` now acquires a cross-process lock
  (`_save_lock`, in `src/coola/io/base.py`) that combines the existing
  in-process `threading.Lock` with a lock file (`<path>.lock`) created via
  `os.open(..., O_CREAT | O_EXCL)`, which atomically fails with
  `FileExistsError` if another process (or thread) already holds it; the
  caller polls until it can create the file, then removes it when done. This
  serializes the full write/commit sequence — including the `exist_ok=True`
  `tmp_path.replace(path)` commit — across processes, not just threads, so
  concurrent writers can no longer interleave their commits (whichever call
  wins the lock last still wins, but no ordering beyond "no interleaving" is
  promised, matching the intentional "last writer wins" semantics of
  `exist_ok=True`). The docstrings were updated to describe the new
  cross-process guarantee. The pre-existing
  `tests/unit/io/test_base.py::test_base_file_saver_save_concurrent_exist_ok_true_serializes_writers`
  and `::test_base_file_saver_save_concurrent_exist_ok_true_does_not_interleave`
  (thread-based, so they exercise the in-process half of the lock) still
  pass. New tests cover the lock-file half directly, by manually creating
  the `<path>.lock` file to stand in for a holder in another process:
  `test_acquire_file_lock_creates_and_removes_lock_file`,
  `test_acquire_file_lock_blocks_while_another_holder_owns_the_lock_file`,
  `test_acquire_file_lock_times_out_when_never_released`,
  `test_base_file_saver_save_blocks_while_lock_file_is_held`,
  `test_base_file_saver_save_removes_lock_file_after_success`, and
  `test_base_file_saver_save_removes_lock_file_after_failure`. Also manually
  verified end-to-end with real OS processes (`multiprocessing.Process`, not
  threads) racing `save(..., exist_ok=True)` on the same path: the result is
  always exactly one writer's full content and no lock file is left behind.

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

- **FIXED** — **`NativeReducer._std` special-cases length 1 to return `nan`
  instead of raising** — `src/coola/reducer/native.py:57-60`. The concern was
  whether `NumpyReducer`/`TorchReducer` agree with `NativeReducer` on the
  length-1 standard-deviation behavior. New
  `tests/unit/reducer/test_consistency.py` parametrizes the same behavioral
  assertions across all three `BaseReducer` implementations (max/min/mean/
  median/std/sum, including the length-1 and empty-sequence edge cases), and
  they pass, confirming `NativeReducer`'s `nan`-for-length-1 behavior matches
  `NumpyReducer`/`TorchReducer` rather than being a cross-backend
  inconsistency.

### Low

- **FIXED** — **`BloomFilter.add_and_check`'s double-hashing derives both `h1`/`h2` from
  one SHA-512 digest** — `src/coola/utils/bloom_filter.py:70-93`. This is a
  reasonable, well-documented trade-off (comment explains it), but note
  `_optimal_hash_count` can return an unbounded number of hash rounds for
  very small `n`/large `m` ratios; for `expected_items=1`, `fp_rate` close to
  0, `hash_count` could get large. Not a bug, but worth a sanity cap given
  it's called once per `add_and_check`. Fix: `_optimal_hash_count`
  now clamps its result to `_MAX_HASH_COUNT` (32), covered by
  `test_bloom_filter_hash_count_is_capped_for_extreme_parameters` and
  `test_bloom_filter_add_and_check_works_with_capped_hash_count` in
  `tests/unit/utils/test_bloom_filter.py`.

---

## 2. API Design / Consistency Issues

- **FIXED** — **Five near-identical registry wrapper classes** —
  `EqualityTesterRegistry` (`src/coola/equality/tester/registry.py`),
  `TransformerRegistry` (`src/coola/recursive/registry.py`), `HasherRegistry`
  (`src/coola/hashing/registry.py`), `SummarizerRegistry`
  (`src/coola/summary/registry.py`), `ChildFinderRegistry`
  (`src/coola/iterator/bfs/registry.py`), and `IteratorRegistry`
  (`src/coola/iterator/dfs/registry.py`) each hand-implemented `register`,
  `register_many`, `has_<x>`, `find_<x>`, and a `_get_repr_kwargs` that
  wrapped an internal `TypeRegistry`, with one-line delegations to
  `self._state.*` and near-verbatim docstrings. Added
  `BaseTypeDispatchRegistry[V]` in `src/coola/registry/dispatch.py` (exported
  from `coola.registry`), a shared generic base providing `__init__`,
  `_get_repr_kwargs`, `register`, `register_many`, `has`, and `find`. All six
  classes now subclass it and keep only their type-specific, richly
  docstringed `has_<x>`/`find_<x>` wrappers (thin one-liners delegating to
  `self.has`/`self.find`) plus their domain entry point (`objects_are_equal`,
  `hash`, `transform`, `summarize`, `find_children`/`iterate`). This removed
  the duplicated `__init__`/`_get_repr_kwargs`/`register`/`register_many`
  bodies and docstrings from all six files. As a side effect, fixing this
  also required breaking a latent import cycle: `BaseRegistry.equal()`
  (`src/coola/registry/base.py`) imported `coola.equality.interface` at
  module scope, which transitively imports `EqualityTesterRegistry` — that
  import is now local to `equal()`, which also let `EqualityTesterRegistry`
  drop its own local `TypeRegistry` import workaround. Covered by
  `tests/unit/registry/test_dispatch.py` (new): behavior tests for
  `BaseTypeDispatchRegistry` itself (init/copy-on-init, register with/without
  `exist_ok`, `register_many`, `has` vs. MRO-resolving `find`, missing-key
  `KeyError`, `repr`), plus parametrized tests asserting all six registries
  are `BaseTypeDispatchRegistry` subclasses and that `register_many` on each
  concrete registry actually goes through the shared implementation. The
  full existing test suite (5877 tests) and all doctests in the touched
  modules still pass unchanged.

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

- **FIXED** — **`register_many`'s "atomic" claim is per-registry, not
  cross-registry** — the docstring (`src/coola/registry/base.py:273-326`)
  called the operation atomic when `exist_ok=False` without qualifying what
  "atomic" meant: that guarantee is "either every key-value pair in the call
  ends up registered in *this* registry, or none do," not cross-registry
  isolation — if the same mapping is registered into several registries in
  sequence, a failure on a later registry does not roll back an earlier,
  already-successful one. The docstring now states this explicitly. Covered
  by two new tests in `tests/unit/registry/test_base.py`:
  `test_base_registry_register_many_is_atomic_per_registry_on_duplicate`
  (a failed call leaves the registry unchanged) and
  `test_base_registry_register_many_is_not_atomic_across_registries` (a
  successful call on one registry is not rolled back when a later call on a
  different registry fails).

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

- **FIXED** — **`resolve_object`'s "any `dict` subclass is treated as
  config" caveat** (`src/coola/factory/resolve.py:63-128`).
  `resolve_object(Counter(...), cls=Counter)` used to always treat a `dict`
  (including `dict` subclass instances like `Counter`/`OrderedDict`) as a
  factory configuration, even when it was already a valid `cls` instance,
  raising a confusing `TypeError` ("missing the `_target_` key") instead of
  just returning the object. `resolve_object` now only takes the
  factory-configuration branch for a `dict` when `cls` is *not* a `dict`
  subclass that `obj` already satisfies — i.e. when `cls` is itself a
  `dict` subclass (e.g. `Counter`, `OrderedDict`) and `obj` is already a
  valid instance of it, `obj` is returned as-is like any other pass-through
  case; a plain `dict` describing how to build such an instance, or a
  `dict` subclass instance that is *not* a valid `cls` instance (e.g. a
  `Counter` when `cls=OrderedDict`), is still treated as configuration as
  before. The docstring's **Note** was rewritten to describe the new,
  narrower rule. Covered by new tests in
  `tests/unit/factory/test_resolve.py`:
  `test_resolve_object_dict_subclass_instance_matching_cls_is_passed_through`,
  `test_resolve_object_counter_instance_matching_cls_is_passed_through`,
  `test_resolve_object_plain_dict_with_dict_subclass_cls_is_treated_as_config`,
  and `test_resolve_object_dict_subclass_instance_not_matching_cls_is_treated_as_config`.

---

## 3. Code Duplication / Reuse Opportunities

- **Registry boilerplate** — see section 2's first bullet; this is the
  single largest duplication opportunity in the package (six structurally
  identical wrapper classes around `TypeRegistry`).

- **FIXED** — **`SupportsAllCloseNan` (`src/coola/equality/handler/allclose.py:20-43`)
  and `SupportsTolerantEqual` (`src/coola/equality/handler/tolerant.py:21-59`)
  Protocols duplicate the `allclose` method signature verbatim** (both
  declared the same `allclose(self, other, rtol=1e-5, atol=1e-8,
  equal_nan=False) -> bool`). `SupportsTolerantEqual` now extends
  `SupportsAllCloseNan` (`class SupportsTolerantEqual(SupportsAllCloseNan,
  Protocol)`) and only adds the `equal` method, removing the copy-pasted
  `allclose` signature and docstring. Covered by two new tests in
  `tests/unit/equality/handler/test_tolerant.py`:
  `test_supports_tolerant_equal_extends_supports_allclose_nan` (asserts
  `SupportsAllCloseNan` is in `SupportsTolerantEqual`'s MRO) and
  `test_supports_tolerant_equal_does_not_redefine_allclose` (asserts
  `allclose` is inherited, not redeclared, on `SupportsTolerantEqual`); the
  existing `test_tolerant.py`/`test_allclose.py` suites (76 tests) still pass
  unchanged.

- **FIXED** — **`hasattr(x, "method") and callable(x.method)` guard pattern
  repeated** — `src/coola/equality/handler/allclose.py:91`,
  `src/coola/equality/handler/tolerant.py:122-127`. Added
  `supports_methods(obj, *method_names)` to
  `src/coola/equality/handler/utils.py` (exported from
  `coola.equality.handler`), which checks that `obj` has every named method
  and that each is callable. `AllCloseNanHandler.handle` and
  `TolerantEqualHandler.handle` now call `supports_methods(actual, ...)`
  instead of the inlined `hasattr`/`callable` checks. Covered by new tests
  in `tests/unit/equality/handler/test_utils.py`
  (`test_supports_methods_*`: no method names, single/multiple methods
  present, a missing method among several, a non-callable attribute, a
  plain `object()`, and a builtin `int` method), plus the existing
  `test_allclose.py`/`test_tolerant.py` suites, which still pass unchanged
  since the observable behavior of both handlers is identical.

- **FIXED** — **`_get_repr_kwargs` returning `{}`** appears in many leaf
  classes purely to satisfy the `BaseDisplayMixin` abstract method contract
  (e.g. `PickleLoader._get_repr_kwargs` in `src/coola/io/pickle.py:37-38`,
  `DefaultTransformer._get_repr_kwargs` in
  `src/coola/recursive/default.py:55-56`, `SequenceHasher._get_repr_kwargs`
  in `src/coola/hashing/sequence.py:42-43`). Added `NoArgsDisplayMixin`
  (`src/coola/display/mixin.py`), a mixin providing a default
  `_get_repr_kwargs` returning `{}`; a stateless class can now mix it in
  alongside `InlineDisplayMixin`/`MultilineDisplayMixin` (e.g. `class Foo(
  NoArgsDisplayMixin, InlineDisplayMixin)`) instead of writing a trivial,
  content-free override, while a class with actual constructor arguments
  keeps overriding `_get_repr_kwargs` as before. Exported from
  `coola.display`. Covered by new tests in
  `tests/unit/display/test_mixin.py` (empty-dict default, composition with
  both `Inline`/`MultilineDisplayMixin`, that a subclass override still
  takes precedence, and that it is a `BaseDisplayMixin` subclass).
  Migrating the existing leaf classes listed above to use it is left as a
  follow-up.

- **FIXED** — **Type-lookup docstring/example blocks are copy-pasted nearly
  verbatim** across `find_equality_tester`, `find_transformer`, `find_hasher`,
  `find_summarizer`, `find_child_finder`, `find_iterator` (and their
  `has_<x>` counterparts; see section 2). Beyond the code itself, this meant
  any correction to the behavioral description (e.g. the stale "LRU cache
  (256 entries)" claim flagged above) had to be hunted down and fixed in
  multiple places — which is presumably how the inconsistency was
  introduced; the six `find_<x>` docstrings also disagreed on caching
  details (some said nothing, one said "unbounded, per-instance", two just
  said "caches the result for performance"). The full MRO/caching/`KeyError`
  behavioral description now lives once, on
  `BaseTypeDispatchRegistry.has`/`.find` (`src/coola/registry/dispatch.py`),
  including a new **Note** on `find` documenting the internal cache. Each
  concrete registry's `has_<x>`/`find_<x>` wrapper docstring was trimmed
  down to its own Args/Returns/Example plus a `See also` pointer back to the
  base method, instead of re-describing the shared behavior. Covered by new
  tests in `tests/unit/registry/test_dispatch.py`:
  `test_registry_wrapper_docstrings_reference_base_class` (each wrapper's
  docstring points at `BaseTypeDispatchRegistry.has`/`.find`),
  `test_registry_wrapper_docstrings_do_not_duplicate_mro_prose` (the
  "Method Resolution Order" behavioral description no longer appears
  copy-pasted in any `find_<x>` docstring), and
  `test_base_type_dispatch_registry_find_docstring_documents_caching`
  (the caching behavior is documented on the base class). All existing
  tests and doctests in the touched modules still pass.

---

## 4. Performance Concerns

- **FIXED** — **`TypeRegistry.resolve()`'s MRO walk is `O(len(mro))` per
  cache miss and the cache was unbounded** (`src/coola/registry/type.py`).
  For the package's steady-state use (a fixed, small set of registered
  types), this was fine, but a consumer calling `objects_are_equal`/`hash`/
  `transform` on many distinct dynamically-generated classes (e.g. pydantic
  models created per-request, or `NamedTuple`s created in a loop) grew the
  cache forever with no eviction — a slow, unbounded memory leak in
  long-running processes. `TypeRegistry._cache` is now an `OrderedDict`
  used as a bounded LRU cache capped at 1024 entries (`_MAX_CACHE_SIZE`):
  `resolve()` moves a hit to the most-recently-used end and, after
  inserting a new entry, evicts the least-recently-used entry once the
  cache exceeds the cap; `_on_change()` still clears the whole cache on
  `register`/`unregister`/`register_many`/`clear` as before. The stale
  "LRU cache (256 entries)" docstring in
  `EqualityTesterRegistry.find_equality_tester`
  (`src/coola/equality/tester/registry.py`) was updated to say 1024,
  matching the real cap it delegates to. Covered by new tests in
  `tests/unit/registry/test_type.py`:
  `test_type_registry_resolve_cache_is_bounded_lru` (cache never exceeds
  1024 entries and evicts the least-recently-used type first),
  `test_type_registry_resolve_cache_lru_order_updated_on_access` (re-resolving
  a cached type protects it from eviction), and
  `test_type_registry_resolve_cache_cleared_does_not_exceed_max_size`
  (the cache stays bounded across a `_on_change()` clear and repopulation).

- **FIXED** — **`SequenceHasher.hash` and `SequenceSameValuesHandler`/
  `MappingSameValuesHandler` recurse through `registry.hash`/
  `config.registry.objects_are_equal` per element with no short-circuit
  reuse of already-computed hashes for repeated/interned values** — for
  workloads with highly repetitive nested structures (e.g. many identical
  sub-trees), there was no memoization keyed by `id()`/structural hash, so
  identical sub-structures were re-hashed/re-compared repeatedly. All three
  now keep a cache local to a single `hash`/`handle` call: `SequenceHasher.hash`
  (`src/coola/hashing/sequence.py`) caches `id(item) -> hash string`, so an
  item appearing at several positions in the sequence is passed to
  `registry.hash` only once; `SequenceSameValuesHandler.handle`
  (`src/coola/equality/handler/sequence.py`) and
  `MappingSameValuesHandler.handle`
  (`src/coola/equality/handler/mapping.py`) cache
  `(id(value1), id(value2)) -> bool`, so a repeated pair of values (e.g. the
  same shared sub-object referenced under several indices/keys) is passed to
  `config.registry.objects_are_equal` only once. The cache is per-call
  (a fresh dict each time), so it introduces no cross-call staleness for
  mutable objects — it only avoids redundant work for objects that are
  still the same object *within* one comparison/hash. Docstrings for all
  three were updated to describe the new identity-based memoization.
  Covered by new tests asserting the underlying `registry.hash`/
  `objects_are_equal` mock is called once for a value repeated several
  times but still once per distinct value otherwise:
  `test_sequence_hasher_hash_reuses_cached_result_for_repeated_object` and
  `test_sequence_hasher_hash_does_not_reuse_cache_across_different_objects`
  in `tests/unit/hashing/test_sequence.py`;
  `test_sequence_same_values_handler_handle_reuses_cached_result_for_repeated_object`
  and
  `test_sequence_same_values_handler_handle_does_not_reuse_cache_across_different_objects`
  in `tests/unit/equality/handler/test_sequence.py`; and
  `test_mapping_same_values_handler_handle_reuses_cached_result_for_repeated_object`
  and
  `test_mapping_same_values_handler_handle_does_not_reuse_cache_across_different_objects`
  in `tests/unit/equality/handler/test_mapping.py`.

- **FIXED** — **`BaseRegistry.items()/keys()/values()` all took a full
  `dict.copy()` under the lock on every call** —
  `src/coola/registry/base.py:367-419`. Thread-safety still requires never
  returning a live view of `_state` that could be mutated concurrently
  while iterated, but repeated read-only calls between mutations paid for a
  fresh O(n) allocation every time. `BaseRegistry` now caches the snapshot
  copy on `self._snapshot` (lazily built by a new `_get_snapshot()` helper)
  and reuses it across calls; a new `_invalidate()` helper — called by
  `register`, `register_many`, `unregister` and `clear` instead of calling
  `self._on_change()` directly — clears `self._snapshot` back to `None`
  before delegating to `_on_change()`, so subclasses that override
  `_on_change` (e.g. `TypeRegistry`, to clear its own resolution cache)
  keep working unchanged. `items`/`keys`/`values` now call
  `self._get_snapshot()` instead of `self._state.copy()` directly. Covered
  by new tests in `tests/unit/registry/test_base.py`: repeated calls reuse
  the same cached dict (`is` identity), the cache starts `None` before the
  first read, each mutating method (`register`, `register_many`,
  `unregister`, `clear`) invalidates it while a failed `register` (duplicate
  key, `exist_ok=False`) does not, and a previously returned view stays
  detached (unaffected) after a later mutation.

- **FIXED** — **`BloomFilter._hashes` computes a fresh SHA-512 digest per
  call** — `src/coola/utils/bloom_filter.py:79-102` — appropriate for its
  stated use case (approximate duplicate detection over documents), not a
  concern at the intended data volumes, but SHA-512 is heavier than
  necessary purely for a non-cryptographic bloom filter. Switched to
  `hashlib.blake2b(item, digest_size=32)`: still pure stdlib (no new
  dependency), faster than SHA-512 in CPython, and requests exactly the 32
  bytes needed (two 16-byte halves for `h1`/`h2`) instead of truncating a
  64-byte SHA-512 digest. Covered by new tests in
  `tests/unit/utils/test_bloom_filter.py`:
  `test_bloom_filter_hashes_uses_blake2b` (pins the digest algorithm and
  the exact index derivation), plus
  `test_bloom_filter_hashes_yields_hash_count_indices`,
  `test_bloom_filter_hashes_is_deterministic`, and
  `test_bloom_filter_hashes_differ_for_different_items`.

---

## 5. Type Hints / Documentation Gaps

- **FIXED (documented)** — **`AllCloseNanHandler.handle`'s signature types
  `actual` as `SupportsAllCloseNan` but the method itself checks
  `hasattr(actual, "allclose")` before trusting that**
  (`src/coola/equality/handler/allclose.py`). The class docstring now
  explicitly documents that `SupportsAllCloseNan` describes the intended
  "happy path" contract, while the implementation is deliberately defensive
  against arbitrary inputs reaching it via the dispatch registry (which
  resolves by `type(actual)`, not by protocol conformance).

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

- **FIXED (docstring reworded, no re-exports added)** — **`coola/__init__.py`
  exposes only `__version__`** — the module docstring was rewritten and now
  explicitly says it "only exposes `__version__`" and does not re-export
  entry points, pointing readers at `coola.equality`/`coola.summary`
  explicitly (e.g. `from coola.equality import objects_are_equal`), instead
  of the old, misleading "Use this package to compare nested objects..."
  framing. The alternative option (adding lazy re-exports via
  `__getattr__`) was not taken — this was the "adjust the docstring" branch
  of the original suggestion, a deliberate choice, not an oversight.

- **STILL OPEN** — **`EqualityConfig.__post_init__` validates
  `atol`/`rtol`/`max_depth` but not `equal_nan`/`show_difference`/`registry`
  types** — confirmed still true (`src/coola/equality/config.py`). Reasonable, since
  those are simple/duck-typed, but note the *type* of `registry` isn't
  checked at all — passing a non-`EqualityTesterRegistry` object with a
  compatible-looking `objects_are_equal` method would work by duck typing
  (arguably a feature, not a bug), but passing something entirely wrong
  produces a late, possibly confusing `AttributeError` deep in a handler
  rather than an immediate, clear error at `EqualityConfig` construction.

---

## 6. Error Handling

- **FIXED (documented)** — **`import_object`/`factory`/`instantiate_object`/
  `resolve_object` (`src/coola/factory/instantiation.py`,
  `src/coola/factory/resolve.py`) form a dynamic-import-and-call pipeline
  that will import and execute arbitrary module code and instantiate
  arbitrary classes from a string path.** This is a deliberate, Hydra-style
  design (used by `coola.io`'s `is_loader_config`/`resolve_loader` to build
  loaders/savers from `dict` configs — `src/coola/io/base.py`). The RCE
  vector itself is unchanged (by design), but explicit **Security** blocks
  now exist on `factory`, `instantiate_object`/`_instantiate_class_object`
  (`src/coola/factory/instantiation.py`), `resolve_object`
  (`src/coola/factory/resolve.py`), and `resolve_loader`/`resolve_saver`
  (`src/coola/io/base.py`), all warning that `_target_`/`object_path` must
  come from a trusted source. No tests to add here (documentation-only
  fix); see the still-open test-coverage suggestion in §7 for exercising
  the "malicious `_target_`" behavior explicitly.

- **FIXED (documented)** — **`PickleLoader.load` uses `pickle.load` on
  arbitrary file paths** (`src/coola/io/pickle.py`), suppressed with
  `# noqa: S301`. Both `PickleLoader`'s class docstring and `load_pickle`'s
  docstring now carry an explicit **Warning** block stating that
  unpickling can execute arbitrary code and that only trusted pickle files
  should be loaded.

- **FIXED (documented)** — **`AllCloseNanHandler.handle` and
  `TolerantEqualHandler.handle`'s contract when the delegated method
  raises** (`src/coola/equality/handler/allclose.py`,
  `src/coola/equality/handler/tolerant.py`) — confirmed there is indeed no
  try/except around `actual.allclose(...)`/`actual.equal(...)`, so a
  raising user-defined method still propagates uncaught through
  `objects_are_equal` (unchanged, and the right behavior). Both
  docstrings now have a **Note** stating this explicitly: "If
  `actual.allclose`/`actual.equal` is present but raises an exception, that
  exception propagates unchanged out of `handle`."

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

- **STILL OPEN** — **`TypeRegistry.resolve()` raises bare `KeyError`** (the
  message text was unified as part of §2, but the exception *type* was not
  changed) — callers like `HasherRegistry.hash` still catch it narrowly
  (`except KeyError:` in `src/coola/hashing/registry.py`) to implement
  `ignore_unhashable`. No `TypeNotRegisteredError` (or similar `KeyError`
  subclass) exists anywhere in `src/coola/` as of this review, so the risk
  described originally is unchanged: a future bug that raises `KeyError`
  from inside `resolve()` for an unrelated reason would still be silently
  swallowed by `ignore_unhashable`-style call sites. Not addressed.

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

- **STILL OPEN** — **Concurrent-mutation tests for `BaseRegistry`** —
  confirmed: no `Thread`/multithreaded stress test exists in
  `tests/unit/registry/` as of this review. The class is documented as
  thread-safe with `RLock`, but that claim is only validated at the
  single-threaded API-surface level.
- **FIXED** — **`SequenceSameValuesHandler` used standalone with mismatched
  lengths and no preceding `SameLengthHandler`** — the handler now checks the
  length itself (see §1), so this is covered directly rather than only via
  `SequenceEqualityTester`'s chain; regression tests in
  `tests/unit/equality/handler/test_sequence.py` assert `False` for
  mismatched-length inputs used standalone.
- **PARTIALLY COVERED** — **`TypeRegistry.resolve()` cache invalidation
  across `register_many` with partial overlap and `exist_ok=True`** —
  `tests/unit/registry/test_type.py` has a `register_many` +
  `exist_ok=True` overwrite test, but it doesn't specifically re-resolve a
  type via `.resolve()` (to populate the cache) before overwriting via
  `register_many(..., exist_ok=True)` and then assert the cache reflects
  the new value — the exact seam originally flagged is not explicitly
  exercised.
- **FIXED** — **`BaseFileSaver.save` concurrent-writer race with
  `exist_ok=True`** — now covered directly; see the tests listed under the
  §1 `BaseFileSaver.save` fix (cross-process lock file tests plus the
  pre-existing thread-based serialization/non-interleaving tests).
- **STILL OPEN** — **`instantiate_object`/`factory` with malicious-looking
  `_target_` (e.g. targeting `os.system`, `eval`, dunder attribute
  traversal like `"builtins.eval"`)** — confirmed: no such test exists in
  `tests/unit/factory/`. Security warnings were added to the docstrings
  (§6), but no test documents the accepted "it will call `os.system` if
  given the chance" behavior in executable form.
- **FIXED (already covered)** — **`get_password`'s `confirm=True` mismatch
  path** and **non-interactive terminal path** in
  `src/coola/utils/password.py` — `tests/unit/utils/test_password.py`
  already mocks `sys.stdin.isatty`/`getpass.getpass` and covers both the
  non-interactive `RuntimeError` path and the mismatched-confirmation case.

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

1. **PARTIALLY FIXED** — `SequenceSameValuesHandler` now hardens the
   "must be chained after a length check" assumption itself (**FIXED** —
   see §1). `MappingSameValuesHandler`'s equivalent "must be chained after
   `MappingSameKeysHandler`" assumption is only **documented** (an explicit
   Warning block), not hardened the same way — it still raises `KeyError`
   standalone on a missing key (§1). Mirroring the sequence-handler fix
   here is the main remaining item from this bullet.
2. **FIXED** — Extracted a shared base (`BaseTypeDispatchRegistry`) for the
   six `TypeRegistry`-backed dispatch registries, eliminating the duplicated
   boilerplate/docstrings and fixing the discovered doc/behavior drift in one
   place (§2, §3).
3. **FIXED** — Reconciled the "LRU cache" docstring claim in
   `EqualityTesterRegistry.find_equality_tester` with the `TypeRegistry`
   cache by making the cache a real bounded LRU (1024 entries) instead of
   the previous unbounded dict, fixing both the documentation mismatch and
   the unbounded growth risk with dynamically generated types (§2, §4).
4. **FIXED** — Added explicit security warnings to the public docstrings of
   `factory`/`resolve_object`/`resolve_loader`/`resolve_saver` and
   `PickleLoader`/`load_pickle` about executing/deserializing untrusted
   input (§6). A test documenting the accepted malicious-`_target_`
   behavior is still not present (§7).
5. **STILL OPEN** — No dedicated "type not registered" exception distinct
   from bare `KeyError` exists; `TypeRegistry.resolve()` still raises plain
   `KeyError`, so callers like `HasherRegistry.hash`'s `ignore_unhashable`
   path can still accidentally swallow unrelated bugs (§6). This is the
   most significant unresolved item from the original review.

### Other confirmed-still-open items (not in the original top 5)

- `EqualityConfig.__post_init__` doesn't validate the `registry` argument's
  type (§5).
- No multithreaded stress tests for `BaseRegistry` concurrency claims (§7).
- No test for `TypeRegistry` cache invalidation specifically through
  `register_many(..., exist_ok=True)` after a prior `.resolve()` populated
  the cache (§7).
- No test exercising `instantiate_object`/`factory` with a malicious-looking
  `_target_` (§7).
