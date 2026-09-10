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

15. **`hashing/str.py` vs `hashing/string.py`** — `StrHasher` (calls `str(data)` first)
    and `StringHasher` (assumes input is already `str`) have confusingly similar names,
    risking accidental misuse.
    **Fix:** rename one to make the semantic difference clear from the name.

16. **`equality/handler/native.py:127-192` (`SameAttributeHandler`)** — hand-rolls the
    same `equal()` pattern that `HandlerEqualityMixin` exists specifically to avoid
    duplicating.
    **Fix:** extend the mixin to support the extra `name` field instead of
    reimplementing it.

17. **`registry/vanilla.py` (`Registry`) vs `registry/type.py` (`TypeRegistry`)** — ~90%
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

19. **`summary/mapping.py`, `summary/sequence.py`, `summary/set.py`** — `.summarize()`
    is near-identical (~25 lines each): same empty/zero-`max_items`/depth-limit/
    truncation logic, differing only in iteration/formatting.
    **Fix:** factor the shared skeleton into `BaseCollectionSummarizer` as a template
    method.

20. **`factory/constants.py:22`, `factory/resolve.py:18`, `factory/instantiation.py:70`** —
    `OBJECT_INIT = "_init_"` is defined and exported "to be robust to naming change"
    but `factory()`/`instantiate_object()` hardcode the literal `"_init_"` instead of
    referencing the constant, defeating its purpose.
    **Fix:** wire `_init_` handling through the constant, or drop the constant and its
    docstring claim.

21. **`nested/mapping.py`** — `merge_mappings`'s `"suffix"` strategy leaves the *first*
    occurrence under the plain key and suffixes only later ones, while
    `flatten_mapping`'s `"prefix"` strategy renames *both* the first and later
    occurrences once a conflict is detected. Two similar dedup APIs behave differently
    for the "keep everything" case.
    **Fix:** document the asymmetry prominently, or align first-occurrence handling.

22. **`iterator/bfs`/`iterator/dfs`** — `_register_default_child_finders` /
    `_register_default_iterators` bootstrap code is near copy-pasted between the two
    packages (plus the same singleton idiom as #18).
    **Fix:** share a common bootstrap/singleton helper.

23. **`identifier/objectid.py` / `identifier/snowflake.py`** — both maintain a private
    module-level default generator instance with near-identical wrapper functions.
    **Fix (minor):** a small `_singleton(cls)` helper would remove the repetition.

## Minor / hardening notes

24. **`identifier/nanoid.py:65-95`** — `os.urandom(...)` sized off `length` in a loop
    with no upper bound; a very large `length` allocates a correspondingly large
    buffer per iteration. Low risk since bounded by caller input, but worth capping.

25. **`utils/env_vars.py:19-67` (`check_env_vars`)** — logs use emoji prefixes
    (`✅`/`❌`) baked into the message, inconsistent with the package's plain-text log
    style and a potential issue for non-UTF-8 log sinks.

26. **`display/pydantic.py:71-73`** — `exclude_fields` names that don't exist on the
    model are silently ignored, which can mask a caller's typo (e.g.
    `exclude_fields=["nmae"]`). Consider warning/raising on unknown field names.

## Note on scope

An earlier automated `/code-review` pass was also run but only inspected the latest
git diff (a docstring-only fix), not the package as a whole — hence this separate
manual, module-by-module review to get full coverage.
