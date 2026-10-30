# Architecture and Design

This document is the **blueprint** of `coola`: the guidelines that every sub-package follows, and a
checklist to implement a new one. The design of each sub-package is on its own page (see
[Design of the packages](#design-of-the-packages)). The rules that users rely on are in the
[user guides](../get_started.md), as a **Properties** table per package.

## Principles

1. **Properties first.** A sub-package starts from a table of properties (what is guaranteed) and a
   list of what is _not_ guaranteed. Every design choice exists to make one property true **by
   construction** (in one place), not by the discipline of each implementation.
2. **One engine, one extension point.** The engine owns the invariants (identity, type, depth, cycles,
   order, laziness, errors). A plug-in (tester, hasher, transformer, summarizer, child finder, rounder,
   manager) only **describes one type**. A plug-in cannot forget a protection because it never recurses
   and never receives the registry.
3. **Strict and loud.** Different types are different. An unsupported type raises an error that names
   the type and how to register it. A default is never silently substituted for a missing object.
4. **coola owns the semantics, the libraries own the structure.** A number is compared, hashed or
   rounded by one kernel; a library adapter only says what the structure is (columns, dtype, shape).
5. **Small and easy to maintain.** Delete code that exists only to work around the design instead of
   adding code next to it. A helper that is one `isinstance` or one comparison is not a function.
6. **Breaking changes are acceptable in a major release,** compatibility code is not kept: no alias, no
   deprecation period. The migration guide lists the changes.

## Anatomy of a sub-package

```
coola/<package>/
    __init__.py    package docstring with the Properties; the public names; lazy_exports for the
                   names that need an optional library
    interface.py   the public functions: validate the arguments, build the config, run the engine
    base.py        the abstract plug-in class (BaseXxx) and its contract, documented in the docstring
    config.py      a frozen dataclass for the options, if there are several (validated in __post_init__)
    engine.py      the loop that owns the invariants
    registry.py    the registry subclass, if the package dispatches on the type
    <kind>.py      the built-in plug-ins by theme (native, sequence, mapping, ...)
    <library>.py   the plug-ins of one optional library (numpy, torch, pandas, ...)
```

The files that a package does not need are not created (`reducer` has no engine, `random` has no
dispatch). The layout of the tests mirrors it (`tests/unit/<package>/test_<module>.py`).

## Package layering

Packages fall into two groups:

- **Core**: `registry` (type-based dispatch), `utils` (generic helpers), `validation`, `formatting`
  (turning values into text) and `display`. A core package imports no feature package.
- **Features**: `equality`, `hashing`, `summary`, `recursive`, `iterator`, `rounding`, `random`,
  `nested`, `jsonable`, `reducer`, `io`, `factory`, `identifier` and `testing`. `equality`, `hashing`, `summary`,
  `recursive`, `iterator` and `rounding` are built on `registry`; `reducer` and `random` are standalone
  helpers. `identifier` depends on `hashing` (the stable identifiers are built on `hash_object`), and
  `hashing` must not import `identifier`. `nested` is built on the engines of `recursive` and
  `iterator`, and `jsonable` on the engine of `recursive`.

The allowed imports are:

| Package      | May import                                                         |
| ------------ | ------------------------------------------------------------------ |
| `validation` | the standard library                                               |
| `utils`      | the standard library and `validation`                              |
| `formatting` | `utils` and `validation`                                           |
| `display`    | `utils`, `validation` and `formatting`                             |
| `registry`   | core packages (`utils`, `validation`, `formatting`)                |
| a feature    | core packages and the features it is built on; never the `testing` |

The contract is enforced with [import-linter](https://import-linter.readthedocs.io/) (configured in
`pyproject.toml`), without exception. Run it with `lint-imports`. A new package adds its place in the
contracts.

## The engine and the plug-in

Every package that walks a structure (`equality`, `hashing`, `summary`, `recursive`, `iterator`) has
the same shape, so a contributor who knows one knows the others.

```
public function (validates, builds the config, picks the registry)
        |
        v
engine: explicit stack of frames; per-call type -> plug-in cache; objects on the current path
        |   plug-in = registry.find_xxx(type(item))        MRO, fallbacks
        |   result  = plug-in.describe(item)               pure: no registry, no recursion
        |        leaf     -> the engine finishes it (compare, hash, call func, print)
        |        children -> push them; combine the results when they are done
        v
the result
```

**The engine owns:**

- the **identity and type rules** (strict type, the same object is equal);
- the **lookup** of the plug-in: by MRO, cached for the call in a plain dict (a registration during a
  call is seen by the next call);
- the **depth**: an explicit stack or queue, so there is no `RecursionError`, whatever the nesting or
  the caller's stack. The depth is bounded by memory only (`max_depth` is a policy, not a safety net);
- the **cycles**: the objects (not their ids) on the current path are kept, and a container that
  contains itself is an error (`ValueError`), or "equal" for `equality`. A container reachable by
  several paths is not a cycle;
- the **order**: depth first, left to right; a mapping gives its values in insertion order, a
  dataclass its fields in definition order, a set its iteration order (not specified);
- the **laziness**: generators or children consumed on demand, no work after the first difference;
- the **errors**: an exception of a plug-in propagates unchanged (except where totality is a property,
  as in `summary`).

**A plug-in contract** has these rules, documented in the base class and checked by one parametrised
unit test over the default registry:

1. It is **pure and cheap**: it does not call the registry, does not recurse, does not modify the
   data, and does not depend on `id()`, on `hash()` or on the iteration order of a set.
2. It returns a **description** (a verdict or a leaf, or the children and how to combine them), not a
   result computed by recursion.
3. It is **stateless** or holds its configuration only, so it can be shared by threads and registries,
   and is equal to another instance of its class.
4. The built-in plug-ins are listed in a table (type, plug-in, default registration) on the package
   page.

All the traversal packages enter the same containers (`list`, `tuple`, `deque`, `Sequence`, `set`,
`frozenset`, `dict`, `Mapping`, `MappingProxyType`, dataclasses) and treat text, bytes, iterators,
arrays and unknown types as leaves, so the same data gives the same leaves everywhere. See
[Nested data and leaves](../uguide/nested_data.md).

## Registries and dispatch

A package that dispatches on the type uses a subclass of `TypeRegistry` (`coola.registry`):

- **Precedence**: a registered type (most specific along the MRO, `object` excluded) > a fallback
  predicate (registration order) > `object` > `TypeNotRegisteredError`. The dataclass rule is a
  fallback (`dataclasses.is_dataclass`), not a marker type. The MRO lookup does not call
  `issubclass`; a virtual subclass is a fallback.
- **State is replaced, never mutated**: a lookup of a known type takes no lock; writers are
  serialized. Views (`items()`, `keys()`) are snapshots. `copy()` gives an independent registry that
  shares the handlers.
- **No silent overwrite**: registering an existing key raises `AlreadyRegisteredError` unless
  `exist_ok=True` (keyword-only). `register_many` is atomic.
- **Fork-safe and picklable**: a child created by `os.fork` never blocks on a lock held by a thread
  that does not exist there (`track_fork`).
- **Lazy groups**: the handlers of an optional library are registered with `register_lazy`, and
  loaded when a type from its package is looked up or when the whole content is read.
- **Default registry**: built once with `make_default_registry_singleton` (`coola.registry.default`),
  which also loads the **plugin group** `coola.<package>.<plural noun>` (for example
  `coola.hashing.hashers`). A broken plugin is skipped with a warning (`COOLA_STRICT_PLUGINS=1` raises).
  Tests isolate the defaults with `temporary_default_registries`.
- **Domain registry**: `class HasherRegistry(TypeRegistry[Hasher])` adds `has_hasher`, `find_hasher`
  (one-line delegations to `has` and `resolve`) and the entry point of the domain. A package also
  exposes `get_default_registry()` (and an alias with the domain name) and `register_<plural>(mapping,
*, exist_ok=False)`.
- A **process-wide lazily built object** that is not a registry uses `LazySingleton`, not
  `functools.cache`.

A package whose key is a name and not a type (`random`) subclasses `Registry` instead.

## Optional dependencies

- **torch**: PyTorch tensor support
- **numpy**: NumPy array support
- **pandas**: DataFrame support
- **polars**: Polars DataFrame support
- **xarray**: xarray support
- **jax**: JAX array support
- **pyarrow**: PyArrow table support
- **pydantic**: pydantic model support, and the formatting of models in `coola.display`
- **rich**: terminal output of `coola.display.rich`
- **colorlog**: colored logging of `coola.display.colorlog`

Each optional dependency is only imported when used (lazy loading): `import coola` and the
import of its subpackages do not import any of them. The `__init__.py` files expose the names that
need a library with `lazy_exports`, and the default registries load the entries of a library the
first time they look for one of its types (`register_lazy`). See
[Lazy imports](add_backend.md#lazy-imports).

The rules for a package that uses an optional library:

- **Import**: never at module level of `src`. `import coola.<package>` imports no optional library.
  Use `lazy_exports` in `__init__.py` for the public names that need one.
- **Availability**: `is_<pkg>_available()` (`coola.utils.imports`) is the single, cached definition of
  "available". `check_<pkg>()` raises `MissingPackageError` with one message and the `pip install`
  hint. `package_available(name)` asks without importing.
- **Fallback modules**: the module of a library guards its import with
  `if TYPE_CHECKING or is_<pkg>_available(): import ... else: from coola.utils.fallback.<pkg> import ...`.
  The fakes raise `MissingPackageError` at the first use, before anything is changed.
- **A missing library is an error at the call**, not at import. A function never runs half without its
  library; where "no feature" is a correct result (colour of a log), the function says so.
- **Adding a library**: two functions in `imports`, one fake module in `fallback/`, one row in
  `testing/marks.py`, the entries in `pyproject.toml`, and the lazy group of each registry. See
  [How to add support for a new library](add_backend.md).

## API conventions

- **Options are keyword-only** (`*`), after the data and the first required arguments.
- **Arguments are validated with `coola.validation`** before any work and before any state changes:
  a wrong type is a `TypeError`, a wrong value a `ValueError`, both naming the argument. Do not write
  `raise ValueError` for a type, range or membership check, and do not add a validator that is one
  `isinstance` or one comparison.
- **Several options are one frozen dataclass** (`EqualityConfig`, `SummaryConfig`) validated in
  `__post_init__`; it is shareable, hashable and thread-safe.
- **The input is never modified.** A function that rebuilds containers returns new ones; leaves are
  shared, never copied.
- **Errors name the cause**: the type, the argument, and what to do (register a handler, install
  `coola[numpy]`). Exceptions are the builtin ones plus a small named hierarchy (`MissingPackageError`,
  `TypeNotRegisteredError`, `UnsupportedTypeError`); no error code.
- **A shared helper is a public function of `coola.utils`** with its own tests, not a private function
  imported across modules.
- **Honest types**: a generic parameter reflects what is checked.
- **The package docstring** carries the Properties table; each public function has a Google-style
  docstring (`Args`, `Returns`, `Raises`) and a `pycon` example that runs as a doctest.

## Concurrency, forks and state

- No global mutable state except the documented ones: the availability caches, the `track_fork`
  registry, the `LazySingleton` instances and the default registries.
- The state of one call (stack, cache, ancestors) is local to the call, so threads and re-entrant calls
  are independent. No thread-local state, no lock on the read path.
- Process-global resources that coola manages (the random generators) are not made thread-safe; the
  limit is documented.
- After a `fork`, locks are rebuilt in the child; a resource whose uniqueness depends on the process
  (a `worker_id`) raises in the child instead of duplicating values.

## Performance

- Performance is a requirement: both the passing and the failing path are measured. A simpler design
  that is slower is rejected unless a benchmark gate is met (`tests/benchmarks/`).
- **Early exit**: check the cheap properties first (type, shape, dtype), then the values; stop at the
  first difference; format a message only when asked.
- **Bounded work** where the output is bounded (`summary` never iterates beyond `max_items`).
- **Cache per call**, not per process, for the type lookup.
- Arrays and columns go through vectorised kernels; there is no Python loop per element.

## Documentation of a sub-package

A new package ships these pages:

| Page                     | Content                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| `uguide/<package>.md`    | Overview, **Properties**, **Not guaranteed**, examples, extension, non-goals |
| `refs/<package>.md`      | `::: coola.<package>` directives                                             |
| `dev/<package>.md`       | Constraints, architecture, contracts, public API, dependents, decisions      |
| `migration/<version>.md` | Breaking changes with a before/after table                                   |
| `mkdocs.yml` navigation  | The three pages above                                                        |

Write down **what the package does not do** and the decisions with the alternatives that were
rejected; a later reader needs the reason, not only the choice.

## Testing blueprint

- **Unit tests** mirror the source tree, one behaviour per test, literal expected values, a regression
  test for every bug (checked to fail without the fix), every function (private included) tested, 100%
  coverage without `# pragma: no cover` on testable lines.
- **Contract test** over the default registry: one sample per registered type, so every plug-in
  honours the rules of its contract (pure, reflexive, does not modify the sample).
- **Property tests** (`tests/properties/`, seeded): the invariants of the Properties table, and the
  cross-package invariants (equal objects have the same hash; the same data gives the same leaves).
- **Integration tests** (`tests/integration/`): real libraries without patching, threads
  (`run_threads`), `os.fork`, subprocesses for the lazy imports. Every patched behaviour of a unit test
  has a real counterpart here.
- **Golden vectors** for any format that users persist (hashes, identifiers).
- **Benchmarks** for the packages with a performance gate.
- Optional libraries use the `*_available` marks of `coola.testing.marks`, and tests that register
  something use `isolated_default_registries`.

## Checklist: a new sub-package

1. Write the Properties table and the "not guaranteed" list first.
2. Decide the place in the layering and add the import-linter contract.
3. Define the plug-in contract and the engine; the interface validates and delegates.
4. Add the registry (a `TypeRegistry` subclass), the default registry, the plugin group, and the lazy
   groups of the optional libraries.
5. Keep optional libraries lazy: `lazy_exports`, fallback modules, `check_<pkg>()`.
6. Make the options keyword-only and validate them with `coola.validation`.
7. Write the unit, contract, property and integration tests; add a benchmark if performance matters.
8. Write the user guide, the reference, the architecture page and the migration entry.
9. Run `ruff`, `ty`, `lint-imports` and the tests, including the doctests of the guides.

## Design of the Packages

Each sub-package of `coola` has its own page with the invariants, the contracts between the parts,
the public API, the dependents and the decisions behind the design.

| Package            | Page                                                               |
| ------------------ | ------------------------------------------------------------------ |
| `coola.display`    | [Display mixins, pydantic formatting, terminal output](display.md) |
| `coola.equality`   | [Equality engine, testers, numeric kernel, tables](equality.md)    |
| `coola.factory`    | [Objects from configurations, allow-list](factory.md)              |
| `coola.formatting` | [Value-to-text helpers](formatting.md)                             |
| `coola.hashing`    | [Hash engine, `encode` contract, encoding](hashing.md)             |
| `coola.identifier` | [Stable and time-ordered identifiers](identifier.md)               |
| `coola.io`         | [Loaders, savers, atomic write](io.md)                             |
| `coola.iterator`   | [`dfs_iterate` and `bfs_iterate`, child finders](iterator.md)      |
| `coola.jsonable`   | [JSON data, one registry, one rule per type](jsonable.md)          |
| `coola.nested`     | [Flat dicts, keys, records](nested.md)                             |
| `coola.random`     | [Seed managers and RNG state](random.md)                           |
| `coola.recursive`  | [`recursive_apply` engine, transformers](recursive.md)             |
| `coola.reducer`    | [Statistics backends](reducer.md)                                  |
| `coola.registry`   | [`Registry` and `TypeRegistry`](registry.md)                       |
| `coola.rounding`   | [One rule, kernels, adapters](rounding.md)                         |
| `coola.summary`    | [Summary engine, summarizers](summary.md)                          |
| `coola.testing`    | [pytest marks and fixtures](testing.md)                            |
| `coola.utils`      | [Foundation helpers](utils.md)                                     |
| `coola.validation` | [Argument validators](validation.md)                               |

## Module organization

```
coola/
├── equality/, hashing/, summary/, recursive/, iterator/, rounding/   # traversal and dispatch
├── nested/, jsonable/, reducer/, random/, io/, factory/, identifier/ # helpers built on the core
├── registry/             # Generic registries and type dispatch
├── formatting/           # Turning values into text
├── display/              # Display mixins, pydantic formatting, terminal output
├── validation/           # Argument validators
├── testing/              # pytest marks and fixtures
└── utils/                # The helpers that coola is built on
```

## Design decisions that apply everywhere

### Why strict type checking?

**Rationale**: it prevents subtle bugs from implicit type coercion. In scientific computing, knowing
that `1` (int) and `1.0` (float) are treated differently catches numerical issues.
**Trade-off**: less convenient, but explicit and safe.

### Why registry-based dispatch?

**Rationale**: it allows extensibility without modifying core code; users add their own types.
**Trade-off**: slightly more complex than `if/else` chains, much more maintainable.

## References

- [PEP 8](https://www.python.org/dev/peps/pep-0008/): Python style guide
- [Design Patterns](https://refactoring.guru/design-patterns): Gang of Four patterns

## Contributing

To contribute to `coola`'s architecture: understand the existing patterns, follow the conventions
above, document the design decisions on the page of the package, write tests, and update this document
when a guideline changes. See the
[contributing guide](https://github.com/durandtibo/coola/blob/main/CONTRIBUTING.md).
