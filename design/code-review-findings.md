# coola Package Review — Findings

Full-package manual review of `src/coola` (192 files), split across 4 subagents
covering: (1) equality/hashing/registry/recursive/reducer, (2) utils/identifier/random,
(3) io/nested/iterator/factory/summary, (4) display/testing/top-level.

Date: 2026-09-09

## Bugs (correctness)

1. **FIXED** — **`nested/__init__.py:13`** — `__all__` lists `remove_keys`, but no such function
   exists (`nested/mapping.py` only defines `remove_keys_if`, `remove_keys_containing`,
   `remove_keys_starting_with`). `from coola.nested import remove_keys` raises
   `ImportError`; `from coola.nested import *` raises `AttributeError`.
   **Fix:** add the missing import/alias, or remove it from `__all__`.

2. **FIXED** — **`summary/mapping.py:56-57`, `summary/sequence.py:60-61`, `summary/set.py:57-58`** —
   at the `max_depth` boundary, these summarizers call
   `registry.summarize(str(data), depth=depth + 1, max_depth=max_depth)`, which
   stringifies the *entire* container via default `repr`/`str` before any `max_items`
   truncation applies. A huge collection at the depth cutoff dumps unbounded output
   instead of the documented compact form.
   **Fix:** truncate (e.g. via `max_items`/`max_characters`) before stringifying.

3. **FIXED** — **`identifier/snowflake.py:96-208` (`generate`)** — `__init__` validates
   `last_timestamp_ms - _EPOCH_MS` fits in 41 bits, but `generate()` never validates
   `timestamp_ms - _EPOCH_MS` before packing it with `<< _TIMESTAMP_SHIFT`. A timestamp
   beyond `_EPOCH_MS + 2**41 - 1` silently overflows into worker/sequence bits instead
   of raising.
   **Fix:** call `validate_bit_range(timestamp_ms - _EPOCH_MS, _TIMESTAMP_BITS, ...)`
   inside `generate()`.

4. **FIXED** — **`utils/conversion.py:69-70` (`to_jsonable`)** — `dataclasses.asdict(data)` deep-copies
   non-dataclass field values; it does not recurse into `numpy.ndarray`, `torch.Tensor`,
   or `pydantic.BaseModel` fields nested inside a dataclass. The docstring's claim that
   dataclasses are converted "recursively" is only true for nested dataclasses — nested
   ndarray/Tensor/BaseModel fields pass through unconverted (not JSON-serializable).
   **Fix:** recursively walk the dict/list produced by `asdict` and apply `to_jsonable`
   to each leaf.

5. **FIXED** — **`utils/format.py:463-470` (`find_best_byte_unit`)** — off-by-one at unit boundaries:
   the loop picks a unit when `(size / multiplier) > 1`, so `size == 1024` returns
   `"B"` instead of `"KB"` (`"1024.00 B"` instead of `"1.00 KB"`).
   **Fix:** use `>=` instead of `>`.

6. **FIXED** — **`equality/handler/numpy.py:86-90` (`array_equal`)** — only `array1`'s dtype is
   checked via `is_numeric_array` before calling `np.allclose`; if `array2` is
   non-numeric (mismatched dtypes reaching this handler, e.g. via a custom registry),
   `np.allclose` can raise instead of returning `False`.
   **Fix:** also require `is_numeric_array(array2)` before calling `np.allclose`.

7. **FIXED** — **`random/registry.py:171-173` (`set_rng_state`)** —
   `for key, value in state.items(): self._state[key].set_rng_state(value)` raises a
   raw `KeyError` if `state` contains a key not currently registered, unlike
   `manual_seed`/`get_rng_state` which only touch what's registered.
   **Fix:** skip unknown keys (or raise a clear, documented error).

8. **FIXED** — **`recursive/key.py:54-80` (`KeyFilterTransformer`)** — `func` is overloaded for
   two unrelated roles: the key-drop predicate (`func(key)`) and, unchanged, the
   recursive value-transform function (`registry.transform(value, func)`). The class's
   own doctest shows the surprising result (`{"keep": 1, "secret": 2}` →
   `{'keep': False}` — the survivor's value gets run through the drop predicate too).
   **Fix:** split into separate `predicate` and `transform_func` parameters.

9. **NOT A BUG** — **`reducer/base.py` (`BaseBasicReducer`)** — `max/mean/median/min/quantile/std` are
   routed through `_is_empty()` → `EmptySequenceError`, but `sort` is a plain
   `abstractmethod` with no empty-check wrapper, so `NativeReducer.sort([])`,
   `NumpyReducer.sort([])`, `TorchReducer.sort([])` silently return `[]` instead of
   raising like the rest of the API.
   **Resolution:** unlike max/mean/median/min/quantile/std, sorting *is* well-defined
   for an empty sequence (`sorted([]) == []`), so `sort` should not raise. An earlier
   pass routed `sort` through `_is_empty`/`EmptySequenceError` for consistency with the
   other methods; that was reverted — `sort([])` now returns `[]` again. Rather than
   keep a pass-through `sort`/`_sort` pair in `BaseBasicReducer`, `sort` was dropped
   from it entirely (with a comment explaining why) and each subclass now implements
   the public `sort` (inherited from `BaseReducer`) directly instead of a private
   `_sort`, and the affected tests/docstrings were updated accordingly.

10. **FIXED** — **`hashing/sequence.py:29-38` (`SequenceHasher`)** — doctest calls
    `hasher.hash([1, 2, 3], registry=registry)` with no expected output line; will fail
    under `--doctest-modules`.
    **Fix:** add the expected hex-digest output.

## API inconsistency / design issues

11. **FIXED** — **`display/testing/fixtures.py:104-118`** — `torch_cuda_available`,
    `torch_numpy_available`, `torch_mps_available` have no `_not_available`
    counterparts, unlike every other backend fixture.
    **Fix:** add the missing `_not_available` marks.

12. **FIXED** — **`display/pydantic.py:65-77` (`_format_pydantic_model`)** — secret-field
    exclusion only operates at the top level of `model.model_dump()`; a `SecretStr`
    nested inside a child `BaseModel` field is not masked.
    **Fix:** recurse into nested `BaseModel` values, or use `model_dump(mode="json")`.

13. **FIXED** — **`display/mixin.py:21-35`** — `BaseDisplayMixin` has no leading underscore but is
    not exported from `coola.display`, forcing consumers to import from the private
    submodule path if they want to type-hint against it.
    **Fix:** export it, or rename with a leading underscore to signal it's private.

14. **FIXED** — **`display/colorlog.py:50-76`** — colored formatter is attached unconditionally
    when `colorlog` is available, even when output isn't a TTY (e.g. redirected to a
    file/CI log), producing raw ANSI codes in non-interactive output.
    **Fix:** check `sys.stderr.isatty()` before attaching the color handler.

15. **NOT A BUG** — **`hashing/str.py` vs `hashing/string.py`** — `StrHasher` (calls `str(data)` first)
    and `StringHasher` (assumes input is already `str`) have confusingly similar names,
    risking accidental misuse.
    **Fix:** rename one to make the semantic difference clear from the name.

16. **FIXED** — **`equality/handler/native.py:127-192` (`SameAttributeHandler`)** — hand-rolls
    the same `equal()` pattern that `HandlerEqualityMixin` exists specifically to avoid
    duplicating.
    **Fix:** `HandlerEqualityMixin` now supports an optional `_equality_attrs()` hook
    (default `()`) naming extra instance attributes to compare alongside type and
    `next_handler`. `SameAttributeHandler` uses the mixin with
    `_equality_attrs() -> ("name",)` instead of reimplementing `equal()`.

17. **FIXED** — **`registry/vanilla.py` (`Registry`) vs `registry/type.py` (`TypeRegistry`)** — ~90%
    duplicated: identical thread-safe implementations of `__contains__`,
    `__getitem__`, `__setitem__`, `__iter__`, `__len__`, `__repr__`, `__str__`,
    `clear`, `equal` (including the same lock-ordering trick), `get`, `has`,
    `register`, `register_many`, `unregister`, `items`, `keys`, `values`.
    `TypeRegistry` only adds MRO `resolve()`/caching on top.
    **Fix:** have `TypeRegistry` compose/extend a shared base with `Registry`.

18. **`get_default_registry()` singleton pattern (non-thread-safe, repeated 6x)** —
    `equality/tester/interface.py`, `hashing/interface.py`, `recursive/interface.py`,
    `random/interface.py`, `iterator/bfs/interface.py`, `iterator/dfs/interface.py`,
    `summary/interface.py` all use
    `if not hasattr(fn, "_registry"): fn._registry = ...` with no lock — two threads
    racing on first use can each build a different registry, silently discarding any
    prior seed/registration state on the loser.
    **Fix:** guard with `threading.Lock` (or initialize eagerly at import time), and
    factor the duplicated logic into one shared helper.

19. **FIXED** — **`summary/mapping.py`, `summary/sequence.py`, `summary/set.py`** — `.summarize()`
    is near-identical (~25 lines each): same empty/zero-`max_items`/depth-limit/
    truncation logic, differing only in iteration/formatting.
    **Fix:** factor the shared skeleton into `BaseCollectionSummarizer` as a template
    method.

20. **FIXED** — **`factory/constants.py:22`, `factory/resolve.py:18`, `factory/instantiation.py:70`** —
    `OBJECT_INIT = "_init_"` is defined and exported "to be robust to naming change"
    but `factory()`/`instantiate_object()` hardcode the literal `"_init_"` instead of
    referencing the constant, defeating its purpose.
    **Fix:** `factory()` no longer declares an explicit `_init_` parameter; it now
    pops the `OBJECT_INIT` key out of `**kwargs` (falling back to `"__init__"`), so a
    config dict built with the `OBJECT_INIT` constant is honored the same way as one
    using the literal `"_init_"` string. Covered by
    `test_factory_object_init_constant_matches_literal_key`.

21. **FIXED** — **`nested/mapping.py`** — `merge_mappings`'s `"suffix"` strategy leaves the
    *first* occurrence under the plain key and suffixes only later ones, while
    `flatten_mapping`'s `"prefix"` strategy renames *both* the first and later
    occurrences once a conflict is detected. Two similar dedup APIs behave differently
    for the "keep everything" case.
    **Fix:** documented the asymmetry prominently in both functions' docstrings, with
    each cross-referencing the other's behavior. Covered by
    `test_merge_mappings_flatten_mapping_first_occurrence_asymmetry`.

22. **`iterator/bfs`/`iterator/dfs`** — `_register_default_child_finders` /
    `_register_default_iterators` bootstrap code is near copy-pasted between the two
    packages (plus the same singleton idiom as #18).
    **Fix:** share a common bootstrap/singleton helper.

23. **FIXED** — **`identifier/objectid.py` / `identifier/snowflake.py`** — both maintain a private
    module-level default generator instance with near-identical wrapper functions.
    **Fix:** each module now exposes a public `get_default_generator()` function that
    lazily builds and caches the shared instance on itself (the same
    `hasattr(fn, "_x")`-on-the-function singleton pattern already used by
    `get_default_registry()` in e.g. `equality/tester/interface.py`), replacing the
    previously eagerly-built, private module-level instance. Covered by
    `test_generate_object_id_uses_shared_default_generator` and
    `test_generate_snowflake_id_uses_shared_default_generator`.

## Minor / hardening notes

24. **FIXED** — **`identifier/nanoid.py:65-95`** — `os.urandom(...)` sized off `length` in a
    loop with no upper bound; a very large `length` allocates a correspondingly large
    buffer per iteration. Low risk since bounded by caller input, but worth capping.
    **Fix:** `generate_nano_id` now raises `ValueError` when `length` exceeds a
    `_MAX_LENGTH` cap (1024). Covered by `test_generate_nano_id_max_length_is_valid`,
    `test_generate_nano_id_length_above_max_raises`, and
    `test_generate_nano_id_very_large_length_raises`.

25. **FIXED** — **`utils/env_vars.py:19-67` (`check_env_vars`)** — logs use emoji prefixes
    (`✅`/`❌`) baked into the message, inconsistent with the package's plain-text log
    style and a potential issue for non-UTF-8 log sinks.
    **Fix:** Removed the `✅`/`❌` emoji from the info/warning log messages in
    `check_env_vars`. Covered by `test_check_env_vars_logs_success_message` and
    `test_check_env_vars_logs_warning_message`.

26. **FIXED** — **`display/pydantic.py:71-73`** — `exclude_fields` names that don't
    exist on the model are silently ignored, which can mask a caller's typo (e.g.
    `exclude_fields=["nmae"]`). Consider warning/raising on unknown field names.
    **Fix:** `_format_pydantic_model` now emits a `RuntimeWarning` listing any
    `exclude_fields` names not present on the model, while still applying the
    exclusion for the names that do match. Covered by
    `test_str_pydantic_model_exclude_fields_missing_field`,
    `test_str_pydantic_model_exclude_fields_missing_field_and_valid_field`,
    `test_str_pydantic_model_exclude_fields_no_warning_for_valid_field`, and
    `test_repr_pydantic_model_exclude_fields_missing_field`.

## Second pass (2026-09-10) — additional findings

27. **`summary/collection.py:249-254` (`BaseCollectionSummarizer.summarize`)** — when
    `max_items` is explicitly negative (documented as "show all items, no
    truncation") and the depth limit is hit, the code falls through to
    `text = str(data)` unconditionally, since the truncation branch only fires
    `if self._max_items >= 0 and len(data) > self._max_items`. A large collection
    summarized with `max_items=-1` and nesting deeper than `max_depth` dumps the
    entire untruncated `str(data)` at that depth, e.g.
    `SequenceSummarizer(max_items=-1).summarize([0]*100000, registry, depth=1, max_depth=1)`
    produces a 100000-element string. Same class of unbounded-stringify bug as #2,
    reintroduced for the `max_items < 0` case that fix didn't cover.
    **Fix:** cap the depth-limit preview length regardless of the sign of
    `max_items`, rather than falling back to plain `str(data)`.

28. **`hashing/mapping.py:80` (`MappingHasher.hash`)** — `for key in sorted(data.keys())`
    requires all keys to be mutually comparable. A dict with heterogeneous key
    types (e.g. `{1: "a", "b": 2}`) raises `TypeError: '<' not supported between
    instances of 'str' and 'int'` instead of hashing, even though `MappingHasher`
    documents only "no hasher registered for a key/value type" as a raise
    condition.
    **Fix:** sort by each key's own hash string (already computed per-item) rather
    than the raw key.

29. **FIXED** — **`equality/handler/format.py:53-58` (`format_mapping_difference`)** — same
    pattern as #28: `sorted(missing_keys)` / `sorted(additional_keys)` assumes
    mutually comparable keys. Comparing two dicts with mixed-type keys (e.g.
    `{1: "x", "y": "z"}` vs `{}`) raises `TypeError` while building the "mappings
    have different keys" diagnostic, turning a legitimate assertion failure into
    an unrelated crash in the message-formatting code path.
    **Fix:** sort by `repr`/`str` of each key instead of the raw key.

30. **`iterator/bfs/registry.py:296-306` (`ChildFinderRegistry.iterate`)** — decides
    whether a value is a "container" via a structural
    `isinstance(current, (Mapping, Iterable))` check, but the child finder actually
    used to expand it comes from `TypeRegistry.resolve`, which walks the concrete
    `dtype.__mro__` only. For an object whose class implements `__iter__` without
    inheriting from `collections.abc.Iterable` (never appearing in its MRO),
    `resolve()` falls back to `DefaultChildFinder`, which yields nothing — so
    `is_container` is `True` but `children` is empty, and `queue.extend(children)`
    silently drops the value with no expansion and no leaf output. Repro:
    ```python
    class Custom:
        def __iter__(self):
            yield 1
            yield 2


    list(bfs_iterate([Custom()]))  # -> [] instead of [1, 2] or [Custom()]
    ```
    The DFS counterpart (`iterator/dfs/registry.py`) has no such split-brain check
    since it dispatches purely through the registry, so BFS and DFS disagree on
    the same input, and BFS additionally loses data silently.
    **Fix:** decide containership solely through the registry (e.g. have
    `find_child_finder` report whether it resolved to a real container finder vs.
    the default) instead of a separate `isinstance` test that can disagree with
    MRO-based resolution.

31. **`io/base.py:239-297` (`BaseFileSaver.save`) — hardening note** — the
    `exist_ok=False` path uses `os.link` + `unlink` specifically to guard against
    a concurrent creation of `path` between the initial check and the commit,
    but the `exist_ok=True` path commits via a plain `tmp_path.replace(path)` with
    no such TOCTOU guard: a file created concurrently by another process between
    the `path.is_file()` check and the replace is silently overwritten, and a
    concurrently-removed parent directory can surface a less specific `OSError`
    than the `IsADirectoryError`/`FileExistsError` raised on the other branch.
    **Fix:** document explicitly that the TOCTOU guard applies only when
    `exist_ok=False`.

## Note on scope

An earlier automated `/code-review` pass was also run but only inspected the latest
git diff (a docstring-only fix), not the package as a whole — hence this separate
manual, module-by-module review to get full coverage.
