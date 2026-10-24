# Review of `coola.equality`: target state

Scope: `src/coola/equality` (54 files, 7.9k lines: 50 handler classes, 33 tester classes, one
registry, one config, one result type), checked against the properties table of
`docs/docs/uguide/equality.md` and `docs/docs/dev/architecture.md`. Breaking changes are allowed.

Method: read `__init__`, `config`, `interface`, `result`, `tester/{base,chain,handler,registry,
interface,collection}`, `handler/{base,utils}`; skimmed the other handlers and testers by name and
size; ran probes on `main` (`20087611`, `.venv`). The per-library handlers (numpy, torch, pandas,
polars, pyarrow, xarray, jax) were **not** reviewed line by line. Nothing was changed in `src` or
`tests`. Items marked **[verified]** were reproduced; **[reading]** come from the code only.

`review/suggestions.md` 23.2 (sets) and 23.3 (plain objects) are already done on `main`
(`SetEqualityTester`, `ObjectAttributesEqualityTester`); 23.1 (pydantic `cached_property`) is
not re-checked here and stays open. The table in `suggestions.md` should be updated.

## 1. What is already right (keep)

- The semantics in the properties table (strict type, recursive, NaN, signed zero, time zones,
  tolerance only for numeric-like values, pure, `==` fallback) match the code. Probes:
  `objects_are_equal(L([1]), [1])` and the reverse are both `False`; `compare(1, 1.0)` explains the
  type difference; `objects_are_allclose("a", "b", atol=1)` is `False`.
- Dispatch by MRO over a registry, tester = handler chain, lazy per-library registration,
  cycle detection (`track_comparison`) and `MaxDepthExceededError` are sound and documented.
- `compare` is thread-safe (the `reporter` callback, no logger mutation).

## 2. Problems

| ID  | P   | Finding                                                                                                                             |
| --- | --- | ----------------------------------------------------------------------------------------------------------------------------------- |
| E1  | P1  | Tolerance defaults and parameter order differ between the entry points **[verified]**                                               |
| E2  | P2  | Two reporting mechanisms (logger + `show_difference`, and `reporter`); `show_difference` is not a comparison setting                |
| E3  | P2  | `EqualityConfig` mixes immutable settings and mutable per-comparison state; handlers reach into its privates                        |
| E4  | P2  | Symmetry is not a stated property, and dispatch uses `type(actual)` only **[reading]**                                              |
| E5  | P2  | Public surface is too wide and unclear: handler vs tester duplicate each other, 83 public classes                                   |
| E6  | P2  | Errors: documented `RecursionError`, real `MaxDepthExceededError`, not exported from `coola.equality`                               |
| E7  | P3  | `max_depth` has to be repeated on five functions, and means "raise" here and "truncate" in `summary`                                |
| E8  | P3  | Default registry cannot be scoped or reset (see `suggestions.md` 24.2); `get_default_equality_tester_registry` is a redundant alias |
| E9  | P3  | `result.py` imports the private `_compare`; the `ComparisonResult.path` rebuild relies on report order                              |
| E10 | P3  | No fast path or explicit contract for unhashable/huge inputs; property tests do not cover the new options                           |

### E1. Entry points disagree on tolerance **[verified]**

```
objects_are_allclose(..., *, rtol=1e-5, atol=1e-8, ...)
compare(..., *, equal_nan, atol=0.0, rtol=0.0, ...)
assert_objects_allclose(..., *, rtol=1e-5, atol=1e-8, ...)
```

- `compare` with no tolerance is exact, while `assert_objects_allclose` is the same call with
  `1e-5/1e-8`. `compare(a, b, rtol=1e-5, atol=1e-8)` is the true counterpart of
  `objects_are_allclose(a, b)`; a user switching from the boolean function to `compare` silently
  changes the result.
- `rtol` comes first in two signatures and second in `compare`. Everyone passes them by keyword,
  but `help()` and the docs read inconsistently.
- The defaults (`1e-5`, `1e-8`) are NumPy's. They are a hidden default for _every_ type
  (`Decimal`, `timedelta`, pandas, tensors), and `objects_are_allclose(1.0, 1.0 + 1e-9)` is `True`
  with no argument. This is fine but it is not in the properties table ("Tolerance" row only says
  "uses `atol` and `rtol`").

**Target**

- One tolerance pair, one order (`rtol`, `atol`), one default per function family:
  `objects_are_equal` / `assert_objects_equal` have no tolerance; the `*_allclose` functions use
  `rtol=1e-5, atol=1e-8`.
- `compare` becomes two functions with the same split, or takes `rtol=None, atol=None` meaning
  "exact" and documents that `rtol=1e-5, atol=1e-8` is the allclose preset. Recommended:
  `compare(actual, expected, *, tolerance: Tolerance | None = None, ...)` with
  `Tolerance(rtol=1e-5, atol=1e-8)` (frozen dataclass, validated once). `Tolerance.numpy_default()`
  documents where the numbers come from.
- Add the defaults to the properties table.

### E2. Reporting

Handlers call `report_difference(logger, config, msg, extra={"coola_path": ...})`. If
`config.reporter` is `None` the message goes to a module logger at `INFO`. So:

- `show_difference` is a _logging_ switch, not a comparison setting, yet it lives in
  `EqualityConfig`, in `objects_are_equal` and `objects_are_allclose`, but not in `compare` (which
  hard-codes `True`) nor in the `assert_*` functions.
- A custom handler that logs directly is silently dropped by `compare` (documented in the note of
  `compare`, which is a trap for extension authors).
- The path is a side channel in the `extra` dict of a log record (`coola_path`), reassembled by
  `compare` from reports ordered innermost-first (`reversed(reports)`).

**Target**

- A handler _returns_ a `Difference | None`-style outcome, or calls `config.report(reason, path_part)`;
  no handler knows about `logging`. The tester/registry level decides the sink:
  - `objects_are_equal` / `objects_are_allclose`: no reporting (`reporter=None`, nothing computed).
  - `compare` / `assert_*`: a collecting reporter.
  - `show_difference=True`: kept as sugar that installs a `LoggingReporter(logger)` built on the
    same interface. Document that it logs at `INFO` on the `coola.equality` logger.
- A single interface `Reporter(message: str, path: tuple[object, ...])`, already what `reporter`
  is, but make it the only one and delete the `logger` parameter of `report_difference`.
- Keep reasons lazily formatted (the cost matters only when a reporter is set). Document that
  nothing is formatted when there is no reporter (true today through `show_difference`).

### E3. `EqualityConfig` conflates settings and state **[reading]**

The dataclass holds `registry, equal_nan, atol, rtol, show_difference, max_depth, reporter`
(settings) and `_current_depth, _warned, _in_progress` (state of one run). Consequences:

- It is documented "not thread-safe", and a user who builds one config and reuses it for several
  `registry.objects_are_equal(a, b, config)` calls shares `_warned`/`_in_progress`.
- `handler/pyarrow.py` reads and writes `config._warned`, `handler/utils.py` reads
  `config._in_progress`: private attributes used across modules (the project rule for shared
  helpers is public functions).
- Equal-comparison of two configs ignores `reporter` but also the state; copying with
  `dataclasses.replace` copies `_warned` by default factory but not the depth: surprising.

**Target** (breaking, additive for tester authors)

```python
@dataclass(frozen=True, slots=True)
class EqualityConfig:  # settings only: hashable, shareable, thread-safe
    registry: EqualityTesterRegistry
    equal_nan: bool = False
    rtol: float = 0.0
    atol: float = 0.0
    max_depth: int = 100
    reporter: Reporter | None = None


class ComparisonContext:  # one per comparison, created by the entry point
    config: EqualityConfig
    depth: int
    in_progress: set[tuple[int, int]]
    warned: set[str]

    def descend(self) -> ContextManager[None]: ...  # replaces check_recursion_depth
    def visiting(self, a, b) -> ContextManager[bool]: ...  # replaces track_comparison
    def warn_once(self, name, message) -> None: ...
```

`BaseEqualityTester.objects_are_equal(actual, expected, ctx)` and
`BaseEqualityHandler.handle(actual, expected, ctx)` receive the context. Settings are read as
`ctx.config.equal_nan`; keep read-only convenience properties on the context (`ctx.equal_nan`) so
the migration of handlers is mechanical. The "not thread-safe" note disappears for the config, and
remains (honestly) for the context, which is never shared.

### E4. Symmetry and subclass dispatch **[reading + verified]**

The registry picks the tester from `type(actual)` and each tester starts with `SameTypeHandler`.
That makes strict type checks symmetric for the built-in testers, but:

- `EqualEqualityTester`/`DefaultEqualityTester` call `actual.equal(expected)` / `actual == expected`,
  so a user class with an asymmetric `__eq__` gives `objects_are_equal(a, b) != objects_are_equal(b, a)`.
  (Probe: a class `A` with `__eq__` returning `True` and a subclass `B`: both orders are `False`
  because of the type check, so the strict-type property protects the common case.)
- Symmetry (`f(a, b) == f(b, a)`), reflexivity (`f(a, a)` except documented NaN) and transitivity of
  the exact comparison are not in the properties table, and `tests/properties` only checks some of them.

**Target**

- Add three rows to the properties table: _Reflexive_ (with the NaN exception, already documented in
  the docstrings), _Symmetric_ (for `objects_are_equal`; tolerant comparison is **not** symmetric
  because `rtol` scales with `expected`, as in `numpy.isclose`; say so, or make it symmetric as the
  JAX/torch testers' docs hint), _Not transitive with tolerance_.
- Decide on tolerant symmetry once. Recommended: keep the NumPy definition
  (`|a - e| <= atol + rtol * |e|`), name it in the table, and give every library handler the same
  formula (13.2 in `suggestions.md` already made NumPy/torch/JAX consistent; extend to pandas, polars,
  pyarrow, xarray, `Decimal`, `Fraction`, `timedelta` and test it with one parametrised property test
  over all libraries).
- Add hypothesis properties: `objects_are_equal(a, b) == objects_are_equal(b, a)` over the strategies
  in `tests/properties/strategies.py`.

### E5. Surface area: handlers and testers duplicate each other

There are 50 public handler classes and 33 tester classes. Most testers are a fixed chain
(`BaseChainEqualityTester._create_handler`), so every library has a tester class _and_ a handler
class (`NumpyArrayEqualHandler` + `NumpyArrayEqualityTester`, and so on). The two layers have the
same `equal(other)`/`__repr__`/visualize boilerplate, and two documents explain them (equality
tester guide, equality handler guide).

**Target**

- Public, documented, stable: `objects_are_equal`, `objects_are_allclose`, `compare`,
  `assert_objects_equal`, `assert_objects_allclose`, `ComparisonResult`, `EqualityConfig`,
  `EqualityTesterRegistry`, `BaseEqualityTester`, `register_equality_testers`, the exception types,
  and the _named_ testers users reference (`ObjectAttributesEqualityTester`, `EqualEqualityTester`,
  `DifferenceCloseEqualityTester`, `SequenceEqualityTester`, ...).
- Handlers are an implementation detail of the built-in testers, except for the base class and the
  generic building blocks (`SameObjectHandler`, `SameTypeHandler`, `SameLengthHandler`,
  `SameShapeHandler`, `SameDTypeHandler`, `ObjectEqualHandler`, `create_chain`). Library-specific
  handlers (`Pandas...`, `Polars...`, `Torch...`) move to `coola.equality._handlers.<lib>` (private
  package) and are not re-exported. A user who wants a custom tester writes a class, or composes the
  generic handlers with `HandlerEqualityTester`.
- Keep one idiom for the optional-library testers: a module exposing `get_default_testers()` (already
  the case) and nothing else public.
- Result: about 20 public names instead of 83, and the extending guide shrinks to "write a tester,
  register it".

### E6. Errors

`objects_are_equal` documents `Raises: RecursionError`; `check_recursion_depth` raises
`MaxDepthExceededError(RecursionError)`; `_compare` re-raises it or converts an interpreter
`RecursionError` into another actionable error (`raise_recursion_limit_error`). `MaxDepthExceededError`
lives in `coola.equality.handler.utils`, so a user cannot import it from `coola.equality`.
`objects_are_equal` does not document `ValueError` for `max_depth <= 0`
(**verified**: `max_depth=0` raises `ValueError`); `objects_are_allclose` documents `ValueError` only
for the tolerances.

**Target**

- Export `MaxDepthExceededError` from `coola.equality`, document it in `Raises:` of all five
  functions, and add a `ValueError` line for `max_depth` and `max_repr_chars`.
- Add a table "Failure modes" to the docs with the three cases: cycle (equal by convention, see the
  next point), too deep (`MaxDepthExceededError`), interpreter limit (`RecursionError` with advice).
- The properties table says "A self-referencing container raises a `RecursionError`" but
  `track_comparison` treats a repeated pair as equal. Check which is true on `main` and make the
  table and the code agree (probe before changing; **not verified here**).

### E7. `max_depth` repetition and meaning

The parameter is declared, with the same long docstring paragraph, on `objects_are_equal`,
`objects_are_allclose`, `compare`, `assert_objects_equal`, `assert_objects_allclose` and
`EqualityConfig`. Target: a single documented "Common options" section, referenced from every
docstring, and keep the name but rename the concept in the user guide to _nesting limit_ so it is
not confused with `summarize(max_depth=...)` (`suggestions.md` 28.3 proposes the same for the
other side). Do not rename the keyword unless `summary` is renamed in the same release.

### E8. Registry lifecycle

Already covered by `suggestions.md` 24.1 and 24.2 (fork safety, scoping, reset). For `equality`
specifically, add:

- `with temporary_registry() as r:` / `EqualityTesterRegistry.copy()`, so a test can register a tester
  without leaking it.
- Delete the alias `get_default_equality_tester_registry` (two names for one function, 2 exports).
- `register_equality_testers(mapping, exist_ok=False)` should take `*, exist_ok` (keyword-only, per
  the repository rule).

### E9. Internals of `result.py`

- `from coola.equality.interface import _compare`: a private function imported across modules.
  Make it a public helper (`run_comparison(registry, actual, expected, config)`) or move it to the
  `ComparisonContext` factory of E3 so the three entry points share one code path.
- `compare` builds the path with `for _, location in reversed(reports)` and the reason from
  `reports[0]`. With E2 the reporter can receive structured `(reason, path_part)` and build the path
  as the stack unwinds, which removes the "ordering convention" and makes a handler that reports
  twice for one level harmless.
- `ComparisonResult.format_path` renders `AttributeName` as `.name`: keep, but make `AttributeName`
  public in `coola.equality` since custom handlers need it to produce correct paths.

### E10. Tests and docs

- The docs mention "property-based tests (reflexivity)"; add symmetry and allclose-monotonicity
  (`atol` larger never turns a `True` into `False`) properties.
- One parametrised test that walks `get_default_registry()` and checks that every registered tester
  has a distinct `equal`/`__repr__` and handles `(x, x)` for a sample of its type: this guards the
  "reflexive" row for new testers.
- A short "writing a tester" checklist in the architecture document: same type first, report with a
  path, honour `equal_nan` and the tolerance or document that you do not.

## 3. Target state in one view

```
coola.equality
├── objects_are_equal(actual, expected, *, equal_nan, max_depth, registry)
├── objects_are_allclose(actual, expected, *, rtol, atol, equal_nan, max_depth, registry)
├── compare(actual, expected, *, tolerance=None, equal_nan, max_depth, max_repr_chars, registry)
├── assert_objects_equal / assert_objects_allclose   (same options, + root)
├── ComparisonResult, EqualityConfig (frozen), Tolerance
├── MaxDepthExceededError, AttributeName
├── tester/    BaseEqualityTester, EqualityTesterRegistry, register_equality_testers,
│              named testers (collection, scalar, equal, difference, object, dataclass, ...)
└── _handlers/ building blocks + per-library handlers (private)
```

Data flow: entry point -> `EqualityConfig` (immutable) -> `ComparisonContext` (per call) ->
`registry.find(type(actual))` -> tester -> handlers, which call `ctx.report(...)`; recursion goes
through `ctx.compare_child(a, e, part)` which does the depth check, the cycle check and appends
`part` to the path in one place.

`compare_child` is the one addition worth highlighting: today every container handler must
remember to call `check_recursion_depth`, `track_comparison` and pass a `coola_path`; forgetting one
produces a silent bug (missing path, or missing cycle protection) rather than an error. One method
makes the three behaviours impossible to omit.

## 4. Migration plan (one PR per step, per `CLAUDE.md`)

1. **E1** unify tolerance order/defaults and add the `Tolerance` type; table row; tests that the
   three entry points agree on 20 sample pairs. Behaviour change: `compare` defaults.
2. **E6** export and document the errors; fix the cycle row of the properties table (probe first).
3. **E3 + E9** introduce `ComparisonContext` and `compare_child`; adapt the container handlers, then
   the library handlers; keep a deprecation shim for `handle(actual, expected, config)` for one
   release (the context exposes the config attributes). Breaking for custom handlers.
4. **E2** reporter-only reporting; `show_difference` becomes sugar; remove `logger` parameter of
   `report_difference`. Breaking for custom handlers that call it.
5. **E4** symmetry/tolerance formula for the remaining libraries + hypothesis properties.
6. **E5** privatise library handlers; shrink `__all__` of `coola.equality.handler`; update the
   extending guide. Breaking for imports of `coola.equality.handler.<lib>`.
7. **E8, E7, E10** registry scoping, docs consolidation, test additions.

Steps 1, 2 and 7 are independent; 3 must precede 4 and 6.

## 5. Not covered

- Per-library numerical semantics (NaN, integer overflow in tolerance, masked values, lazy polars).
- Performance (the cost of the context object per recursion, MRO cache hit rate); measure before E3
  with `pytest-benchmark` on a 10k-element nested dict, and keep the regression below 5%.
- GPU, Windows and `fork` behaviour.
