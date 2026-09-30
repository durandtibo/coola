# `coola` Improvement Suggestions

Scope: `src/coola` (all subpackages). This complements
`code-review-findings.md` (bug/hardening focus) with **design, API and
maintainability** suggestions. Nothing here is a bug; items are ordered by
estimated value / effort. Line counts are from the current tree.

## 1. Deduplicate the five parallel registry/interface stacks (high value)

**Fixed** (`BaseTypeDispatchRegistry` in `coola.registry` and
`make_default_registry_singleton` in `coola.utils.singleton`; per-package
`register_X` / `get_default_registry` kept as thin public wrappers).

`equality`, `hashing`, `recursive`, `summary`, `random`, `iterator/{bfs,dfs}`
each ship the same skeleton:

- `registry.py` (145–225 lines each, ~765 total): a `TypeRegistry` wrapper with
  `has_X`, `find_X`, and one dispatch method (`hash`, `summarize`, ...).
- `interface.py`: `register_X`, `get_default_registry`, `_register_default_X`,
  `_build_default_registry`, a `LazySingleton`, and one user-facing function.

The pieces that differ are the dispatch method signature and the default
entries. Suggestions:

1. Introduce a generic `BaseTypeDispatchRegistry[V]` in `coola.registry` that
   owns `register`, `register_many`, `has`, `find`, `__repr__`, and
   lazy-default handling. Subclasses would only add their one dispatch method
   and rename aliases (`find_hasher = find`) if backwards compatibility needs it.
2. Add a small factory, e.g. `make_default_registry_accessor(build, register)`,
   returning `(get_default_registry, register_fn)`. This removes ~6 near-identical
   functions per package and guarantees consistent thread-safety and
   `exist_ok` semantics.
3. Have the docstring examples reference the shared base once, instead of being
   copy-pasted with different nouns (they now drift; e.g. differing wording on
   cache invalidation).

Risk: public class names and methods must stay; do it as an internal refactor
with deprecation aliases if any names change.

## 2. A dedicated "not registered" exception (medium)

**Fixed** `TypeRegistry.find/resolve` and `BaseRegistry.__getitem__` raise bare `KeyError`.
Callers cannot distinguish "no handler for this type" from an ordinary dict
miss inside user code. Add `class TypeNotRegisteredError(KeyError, LookupError)`
(subclassing `KeyError` keeps existing `except KeyError` code working) and include
the type's MRO plus the nearest registered types in the message. This makes the
most common user error (forgetting to register a custom type) self-diagnosing.
Already noted as open in `code-review-findings.md`; recommended as a first,
low-risk step.

## 3. Public API surface and discoverability (medium)

**Partially fixed** (distinct `get_default_*_registry` aliases and `tests/unit/test_public_api.py`
added; lazy top-level re-exports not done).

- `coola/__init__.py` exports only `__version__`. That is a deliberate,
  documented choice, but it forces users to learn six import paths. Consider
  lazily re-exporting the four headline functions (`objects_are_equal`,
  `objects_are_allclose`, `summarize`, `hash_object`) via module `__getattr__`
  (PEP 562). Lazy loading preserves the "no optional-dependency import at
  startup" property and keeps `import coola` cheap.
- **Fixed** Every subpackage defines its own `get_default_registry`. Same name, different
  return types, so `from coola.x import *` collisions and confusing IDE
  auto-imports. Consider distinct public aliases (`get_default_hasher_registry`, ...) while keeping
  the old names.
- **Fixed** Add an `__all__` consistency test (all names in `__all__` importable; every
  public module has `__all__`) to prevent API drift.

## 4. Optional-dependency handling (medium)

**Partially fixed** (`LazyModule` / `lazy_import` exist in `coola.utils.imports`, but the
conditional `if is_X_available(): import X` blocks remain and there is no uniform
backend registration hook / entry-point group yet).

Currently three mechanisms coexist: `coola.utils.imports` (`is_X_available`,
decorators), `coola.utils.fallback.*` (stub modules), and `TYPE_CHECKING`
guards plus `if is_X_available(): import X` blocks repeated at the top of many
files (e.g. `equality/tester/interface.py` has seven).

- Centralise via a lazy-module proxy (one `LazyModule("numpy")` object per
  backend). This removes the repeated conditional imports and the
  `# pragma: no cover` markers scattered around them, and `fallback/` can be
  reduced to the proxy raising an actionable message (`pip install coola[numpy]`).
- Give every backend a uniform registration hook (entry point group
  `coola.backends`, or a module-level `register()` function) so
  `_register_default_X` in each interface stops importing/if-checking each
  backend by name. Third-party packages (e.g. for cupy, dask, arrow-like
  types) could then plug in without touching coola, which is the stated purpose
  of `register_*` today but currently only possible imperatively.

## 5. Equality: results and diagnostics (medium-high)

- **Fixed** `objects_are_equal` returns `bool`; the reason for a mismatch is available
  only through logging (`show_difference`). Add a structured variant such as
  `compare(actual, expected, ...) -> ComparisonResult` (`equal: bool`, `path`
  to first difference, `reason`, `actual`/`expected` reprs, all differences when
  `fail_fast=False`). `bool(result)` keeps drop-in semantics. This is the most
  requested capability for a testing helper (pytest assertion messages, custom
  reporting) and the handler chain already computes the information.
- **Partially fixed** (`assert_objects_equal` / `assert_objects_allclose` added in
  `coola.equality`; no pytest plugin yet) Provide a pytest plugin / `assert_objects_equal` that
  raises `AssertionError`
  with a path-annotated diff (`data["a"][2]: 1 != 3`). `coola.testing` is
  currently only skip-markers; this would be a natural home.
- **Fixed** (eager `__post_init__` validation, finite check, unknown options rejected; kept
  non-frozen because of the depth counter) `EqualityConfig` validates lazily. Convert to a frozen
  dataclass with `__post_init__` validation of `atol`/`rtol` (non-negative,
  finite) and reject unknown options.
- **Fixed** (matrix in `docs/uguide/equality.md`) Document/decide NaN and `-0.0` semantics table per
  backend (numpy/torch/pandas/polars/jax/pyarrow); tests exist, but a single matrix in
  the docs would prevent per-backend divergence.

## 6. Typing and static checks (medium)

- **Partially fixed** (`BaseEqualityTester[T]` is generic; strict mypy in CI not verified) `py.typed` is present; make sure generics survive: `BaseRegistry[K, V]` is
  generic but the concrete registries expose `BaseHasher[Any]`. Parameterise
  handlers/testers on the data type (`BaseEqualityTester[T]`) and run
  `mypy --strict` (or pyright strict) on `src/coola` in CI if not already.
- **Fixed** (`Hasher`, `Summarizer`, `Transformer` protocols; registries accept them) Use `typing.Protocol` for the small strategy interfaces (`Hasher`,
  `Summarizer`, `Transformer`) so users can register plain classes/functions
  without subclassing the ABCs. Keep the ABCs as convenience bases.
- Replace `TypeVar` K/V with PEP 695 syntax when the minimum Python moves to
  3.12 (currently `>=3.10`).

## 7. Performance (medium, measure first)

**Partially fixed** (lock-free `TypeRegistry.resolve` cache hits via a copy-on-write snapshot;
registry benchmarks in `tests/benchmarks/test_registry_benchmark.py`; `nested/*` reviewed: `to_flat_dict` now flattens
into one shared dict instead of merging per-level dicts; `from_flat_dict`, `flatten_mapping` and
`merge_mappings` were already single-pass; recursion limit documented and interpreter `RecursionError` wrapped with an actionable message; explicit-stack equality deliberately not done because handler chains are recursive by design).

- Equality/summary/iteration dispatch resolves the type via `TypeRegistry`
  (LRU-cached, good). For hot loops on large nested structures, the per-node cost
  is a lock acquisition plus cache lookup. Options: a lock-free read path (copy
  on write of an immutable snapshot swapped atomically), which makes reads
  contention-free under threads. The `_snapshot` mechanism in `BaseRegistry`
  is already halfway there.
- Add a benchmark suite (`asv` or `pytest-benchmark`) with three canonical
  workloads (deep dict/list, large numpy arrays, DataFrame-heavy) and run it
  in CI on demand. Refactors in section 1 should be gated on it.
- `objects_are_equal` on very deep structures uses recursion; document the
  recursion limit and consider an explicit stack (the `iterator/bfs|dfs`
  modules already model that) to avoid `RecursionError` on pathological input.
- `nested/*` (411-line `mapping.py`, 315-line `flat.py`): confirm
  flatten/unflatten are single-pass and avoid intermediate dict copies.

## 8. Module organisation (low-medium)

- `coola.utils` is a grab bag (`git.py`, `password.py`, `secret.py`,
  `bloom_filter.py`, `stats.py`, `timing.py`, `file_size.py`,
  `text_diff.py`, ...). Several are unrelated to the package's mission (compare,
  summarise, transform). Consider grouping (`utils.text`, `utils.data_structures`,
  `utils.system`) and marking rarely-used ones as private (`_`) or moving them to a
  sibling distribution, so the public API commitment stays small.
- `identifier/` (12 modules: ulid, snowflake, nanoid, uuid4/5/7, checksummed,
  obfuscated, prefixed, ...) is conceptually separate from the rest. It
  could be a subpackage with its own docs section, or an extra (`coola[identifier]`), to make the
  core easier to understand.
- Same for `io/`, `display/`, `reducer/`, `random/`, `factory/`, `validation/`:
  add a short `docs/architecture.md` with a diagram showing which packages
  depend on `registry` (core) versus which are standalone helpers, and enforce
  it with an import-linter contract (`registry` and `utils` must not import
  from feature packages; feature packages must not import each other's
  internals). The lazy singleton comment about circular imports between
  `coola.equality` and `coola.registry` suggests the layering is already
  blurry.

## 9. Security-sensitive surfaces (medium)

**Fixed** ("trusted source only" notices in `io/pickle.py`, `io/torch.py` and
`factory/`; `TorchLoader` defaults to `weights_only=True`; `import_object` and
`factory` accept an allow-list of module prefixes, `allowed_prefixes=` /
`_allowed_prefixes_=`; hashing and `mask_secret` docs state their non-security
scope).

- `io/pickle.py`, `io/torch.py` (`torch.load`) and `factory/instantiation.py`
  (`instantiate_object` from a dotted path) execute arbitrary code by design.
  Add a shared, clearly documented "trusted input only" notice in the
  docstrings and README, default `torch.load(weights_only=True)` where
  possible, and offer an optional allow-list of module prefixes for the factory
  (`allowed_prefixes=`). Cross-reference the open finding in
  `code-review-findings.md`.
- `hashing` names: make clear in docs that these are content hashes for
  identity/caching, not for security; `identifier/obfuscated.py` and
  `utils/secret.py`/`password.py` should state their threat model explicitly (obfuscation is not
  encryption).

## 10. Documentation and testing (low effort)

**Partially fixed** (`htmlcov/` and `coverage.xml` are git-ignored; `xdoctest` is a dev
dependency). Still open: extension tutorial, Hypothesis property tests, registry
concurrent-mutation stress test.

- Docstring examples are excellent and doctested. Add the doctest run to CI if
  not already, and add a "How to add support for a new type" tutorial that
  walks through equality + summary + hash + iterator registration for one
  custom class; this is the package's main extension story and is currently
  scattered over five interface modules.
- Add property-based tests (Hypothesis) for the invariants: `objects_are_equal`
  is reflexive/symmetric, `hash_object` agrees with equality,
  `recursive_apply(identity)` returns an equal object, `flatten` then
  `unflatten` round-trips.
- Add the concurrent-mutation stress test for `BaseRegistry` noted as open.
- Track coverage of the `# pragma: no cover` optional-dependency branches by
  running the matrix job with each backend installed and absent (there are
  `coverage.xml`/`htmlcov` at repo root; make sure they are git-ignored).

## 11. Suggested order

1. **Fixed** `TypeNotRegisteredError` (section 2) — small, immediately useful.
2. **Fixed** Structured comparison result + `assert_objects_equal` (section 5).
3. **Fixed** Registry/interface deduplication (section 1) with benchmarks (section 7) as a safety net.
4. **Partially fixed** Lazy backend proxy + entry-point registration (section 4).
5. **Open** Import-linter layering and `utils` regrouping (section 8).
6. **Partially fixed** Lazy top-level re-exports (still open) and API aliases (done) (section 3).
