# Review of `src/coola`

Scope: ~23.7k lines across 18 subpackages. I read the registry core, `utils`, `io`,
`random`, `identifier`, and the `*/interface.py` entry points in full or in part. I
skimmed the rest. I did not run the tests or linters. Treat the findings below as
leads to verify, not confirmed defects.

Overall the code is in good shape: consistent docstrings with doctests, thread-safe
registries, clear module boundaries (import-linter is in place), and a typed package
(`py.typed`). The suggestions below are mostly about consistency and scope.

## 1. Scope / cohesion

1. **Unrelated modules in the library.** _(Decision: keep `utils/bloom_filter.py` and
   the `identifier/` package as standalone utilities. The `BloomFilter` docstring was
   made generic, and it is now documented in the `utils` reference and user guide. The
   `identifier/` package already has its own guide and reference page.)_
   `utils/bloom_filter.py` is not imported anywhere in `src/`, and `identifier/` is far
   from coola's stated purpose (compare, summarize and transform nested objects), which
   is why they are documented as standalone.
2. **Stale `__pycache__` files.** `src/coola/__pycache__` holds `equal.*.pyc`,
   `reduction.*.pyc`, `summarization.*.pyc`, `types.*.pyc` and `testing.*.pyc` from
   modules that no longer exist. They are untracked, but a stale `.pyc` can shadow
   or confuse imports in some setups. Delete them locally.

## 2. Duplication across registries

The `equality`, `hashing`, `recursive`, `summary`, `iterator/dfs` and `iterator/bfs`
packages each repeat the same scaffolding:

- `register_x(mapping, exist_ok)`, `get_default_registry()`, `_register_default_x(registry)`,
  and a `_default_registry` singleton wrapper.
- A package-specific alias (e.g. `get_default_hasher_registry = get_default_registry`
  in `hashing/interface.py`).

Suggestions:

- Factor the "lazy singleton + plugin loading + default population" into one helper
  (a small generic `DefaultRegistry[T]` in `coola.utils.singleton`). Each interface
  module would then reduce to a few lines.
- Decide whether the `get_default_*_registry` aliases are public API. If they are
  legacy, deprecate them with a warning. If not, remove them.
- `random/registry.py` defines `RandomManagerRegistry` on top of a plain dict and does
  not share `BaseRegistry`. That is reasonable because its keys are strings, but it
  repeats register/has logic that `Registry` (string-keyed) already provides. Consider
  composing `Registry[str, BaseRandomManager]`.

## 3. Correctness / robustness leads

1. **Cross-process file lock in `io/base.py`.** _(Fixed: lock files older than
   300 s are now treated as stale and broken, see `_break_stale_lock`; the
   PID/`filelock` options below were not adopted.)_ `_acquire_file_lock` uses an
   `O_EXCL` lock file with a timeout. If a process is killed while holding the lock,
   the `.lock` file stays and every later save times out. Consider storing the PID
   and timestamp in the lock file and breaking locks older than some threshold, or
   using `fcntl`/`msvcrt` (or the `filelock` package) so the OS releases the lock
   on process death.
2. **Plugin loading swallows all exceptions** (`utils/singleton.py`, `_load_plugin`,
   `except Exception` plus a `RuntimeWarning`). This is a defensible choice, but a
   broken plugin is easy to miss. Consider logging with `exc_info` or a strict-mode
   env var that re-raises (useful in CI).
3. **Lazy import inside `registry/base.py`** (`from coola.equality.interface import
objects_are_equal`, line ~177) makes the low-level registry depend on the
   `equality` package. This is a layering inversion and is probably why a `noqa:
PLC0415` is needed. Check whether it can be done through a callable injected by
   `equality`, or by a small comparison function in `utils`.
4. **Repeated `dict.copy()` in `BaseRegistry.__repr__`/`__str__`/`__iter__`.**
   _(Fixed: all three now use `_get_snapshot()`.)_ A
   snapshot cache (`_snapshot`) already exists, and `__iter__`, `__repr__` and
   `__str__` do not use it, so they pay O(n) per call. Use the snapshot there too.
5. **Snowflake / ObjectId generators.** These depend on wall-clock time. Please
   check that the clock going backwards (NTP step) is handled. A spin loop with
   `time.sleep` appears at `snowflake.py:193`. Confirm there is an upper bound or
   clock-regression error so it cannot spin forever.

## 4. API surface and ergonomics

1. The top-level `coola/__init__.py` only exports `__version__`. This is explicit in
   its docstring and is fine, but users must know the submodule for each entry point.
   Consider re-exporting the three or four main functions (`objects_are_equal`,
   `objects_are_allclose`, `summary`) lazily (module `__getattr__`) so import time
   stays low.
2. Many `# noqa: ARG002` markers exist because handlers must keep a uniform
   signature (`registry`, `ignore_unhashable`, `depth`, `max_depth`...). Configure
   per-file ignores for those directories in `pyproject.toml` instead of annotating
   each method. This removes about 30 noise comments.
3. Optional-dependency handling is split between `utils/imports` and
   `utils/fallback`. A short doc page on how a new optional backend (numpy, torch,
   polars, xarray, pandas, pydantic) is added end to end would reduce the chance of
   registering it in only some of the registries.

## 5. Testing and tooling suggestions

- ~~Add a property-based test (hypothesis) for `TypeRegistry.resolve` MRO behavior
  and cache invalidation under concurrent `register` calls.~~ (Done:
  `tests/integration/properties/test_registry.py` for MRO and cache invalidation, and
  concurrent register/unregister tests in `tests/integration/registry/test_type.py`.)
- ~~Add a concurrency test for `io/base.py` save locking that kills a child process
  mid-write (see 3.1).~~ (Done: `tests/integration/io/test_base.py`.)
- Add a test that every package listed in `__all__` is actually importable without
  optional dependencies installed (a "minimal install" CI job).
- Add a CI check that `__pycache__`-style stale artifacts and unused modules
  (e.g. `vulture`) are not present.

## Priority

| #   | Item                                                 | Effort | Value  |
| --- | ---------------------------------------------------- | ------ | ------ |
| 3.1 | ~~Stale lock-file recovery~~ (fixed)                 | M      | High   |
| 1.1 | ~~Decide on bloom filter / identifier scope~~ (kept) | S      | High   |
| 2   | Shared default-registry helper                       | M      | Medium |
| 3.3 | Remove registry -> equality import                   | M      | Medium |
| 3.4 | ~~Use snapshot in repr/iter~~ (fixed)                | S      | Low    |
| 4.2 | Per-file ruff ignores                                | S      | Low    |
