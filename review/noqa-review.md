# Review of `# noqa` usage

Date: 2026-10-03 · Branch: `pre-commit2` · ruff 0.16.10

## Summary

- 165 `# noqa` directives in Python code (`src/` ≈ 52, `tests/` ≈ 113). One more mention is in a docstring (`equality/handler/base.py`), and some are in generated `docs/site/`, which is not source.
- **No directive is unused.** `ruff check --extend-select RUF100 --no-fix .` passes, so nothing can be deleted blindly.
- No file-level or blanket `# noqa` is used. Every directive names a rule code, which is good.
- Roughly **130 of the 165 (~80%) can be removed** with a few config changes and small refactors. The rest are justified.

(Note: `ruff check --select RUF100` alone wrongly reports all 165 as unused, because it disables every other rule. Always use `--extend-select`.)

## Inventory

| Rule                                              | Total           | src   | tests | Verdict                                         |
| ------------------------------------------------- | --------------- | ----- | ----- | ----------------------------------------------- |
| ARG002 (unused method arg)                        | 71 (68 in code) | 36    | 32    | Keep in `src`, or fix via config. Fix in tests. |
| S311 (`random` not crypto-safe)                   | 27              | 0     | 27    | Per-file ignore for tests                       |
| NPY002 (legacy `np.random`)                       | 14              | 0     | 14    | Per-file ignore for tests                       |
| S105 / S107 (hardcoded password)                  | 9 / 2           | 4 / 2 | 5 / 0 | Mixed. See below.                               |
| BLE001 (blind except)                             | 6               | 0     | 6     | Per-file ignore for `tests/integration`         |
| ARG001 / ARG005 (unused function / lambda arg)    | 6 / 4           | 2 / 0 | 4 / 4 | Rename to `_` or ignore in tests                |
| ANN001 / ANN201                                   | 6 / 1           | 0     | 7     | Fix by adding annotations, or ignore in tests   |
| PT012 (`pytest.raises` with multi-statement body) | 5               | 0     | 5     | Fix by restructuring                            |
| PLW1641 (`__eq__` without `__hash__`)             | 3               | 3     | 0     | Investigate, probably a false positive          |
| S603                                              | 2               | 1     | 1     | Keep (subprocess calls)                         |
| PTH108                                            | 2               | 0     | 2     | Keep (deliberate `os.unlink` patch)             |
| PLC0415 (import outside top level)                | 2               | 2     | 0     | Keep (circular-import avoidance)                |
| B024                                              | 2               | 0     | 2     | Keep, or restructure the test                   |
| UP007, TRY004, S301, PERF203                      | 1 each          |       |       | Keep (see below)                                |

## Recommendations

### 1. Tests: per-file ignores in `pyproject.toml` (~60 directives)

`[tool.ruff.lint.per-file-ignores]` already has a `tests/**` block (D, PL, S101). These rules are noise in tests and are suppressed line by line:

- `S311` (27) and `NPY002` (14): tests of the random-seed managers _must_ use `random.*` and `np.random.*` to check seeding. Add both to `tests/**`.
- `S105`, `S106` (5 in `test_password.py`/`test_pydantic.py`): the test strings are fixtures. Add `S105` and `S106` to `tests/**` and delete the narrower `tests/unit/hashing/test_pydantic.py = ["S106"]` entry.
- `BLE001` (6, `tests/integration/**`): catching `BaseException` in worker threads to forward errors to the main thread is intended. Ignore it for `tests/integration/**`.
- `ANN001`, `ANN201` (7): the benchmark `benchmark` fixture arg and one helper. Either add `ANN` to the tests ignore list, or annotate (the pytest-benchmark fixture type is `BenchmarkFixture`, importable under `TYPE_CHECKING`). Ignoring `ANN` in tests is the simplest option.
- `ARG001`, `ARG002`, `ARG005` (≈40 in tests): stubs and fakes with a signature that must match the real one. Ignoring `ARG` for `tests/**` removes all of them. Narrow alternative: keep ARG enabled and rename to `_`-prefixed names where the arg is positional (lambdas, `failing_link(src, dst)`).

Why a global ignore is acceptable: these checks guard production code. Tests are more likely to hide real problems in review than in lint noise.

### 2. `ARG002` in `src` (36 directives)

Cause: the `equal(other, equal_nan)`, `summarize(registry, depth, max_depth)` and `handle(...)` signatures are shared by every implementation in a chain of responsibility, so implementations must accept args they do not use. The docstring in `equality/handler/base.py` explicitly decides on `# noqa` rather than reworking the interface.

Options, best first:

1. **Per-file ignore for the implementation modules** (`src/coola/equality/handler/*.py`, `src/coola/hashing/*.py`, `src/coola/summary/*.py`, `src/coola/io/*.py`, `src/coola/recursive/*.py`, `src/coola/iterator/*/*.py`). This removes about 36 noise comments but also hides _real_ unused args in these modules. Medium risk.
2. **Keep the directives** and accept the cost. They are accurate and documented. Reasonable if you prefer ARG002 to still catch accidents in these modules.
3. Use `del other, equal_nan` at the top of each method. This is noisier than `noqa`. Not recommended.

Recommendation: option 2 for `src`, unless the count grows. Do 1 for `tests/**` only.

### 3. `PT012` (5, tests)

Pattern: `with (pytest.raises(...), seed_ctx()):` followed by `msg = ...; raise ...`. The extra statement is just building the message. Fix by hoisting `msg = "Exception"` above the `with`, so the body is a single `raise RuntimeError(msg)`. That removes all 5 without a config change (`test_path.py:229`, `test_env_vars.py:266`, `random/test_{torch,numpy,interface}.py`).

### 4. `PLW1641` (3, `random/{torch,random,numpy}.py`)

`InlineDisplayMixin` + `BaseRandomManager` flag "defines `__eq__` without `__hash__`". Check which base class defines `__eq__`. If the managers are intentionally unhashable, set `__hash__ = None` explicitly, or define `__hash__` on the shared base class once. That is one change in `BaseRandomManager` instead of three directives.

### 5. `S105` / `S107` in `src/coola/hashing/pydantic.py` (6)

False positives: the strings are _policy names_ (`"reveal"`, `"exclude"`, `"error"`) compared against a parameter called `on_secret`. Rule S105/S107 triggers on the name containing "secret". Fix by renaming the parameter to e.g. `secret_policy`/`secret_mode`. This is a public-API rename, so it would be a breaking change. Alternative: define a `Literal` alias plus constants and compare against those, which does not fix S107 for the default value. Low priority, so keep the directives unless a rename is planned.

### 6. Keep (justified, 13 directives)

- `S301` `pickle.load` (`io/pickle.py`): the point of the loader. Add a short reason next to it.
- `S603` `subprocess.run` / `Popen` (`utils/git.py`, integration test): trusted fixed command.
- `PLC0415` ×2: deferred imports to avoid circular imports (also registered in importlinter `ignore_imports`).
- `PERF203` (`io/base.py:112`): try/except inside a lock-acquire retry loop. Cannot be hoisted.
- `TRY004` (`nested/flat.py:114`): `ValueError` is the right type, not `TypeError`. Add a reason.
- `PTH108` ×2: tests deliberately call the original `os.unlink`.
- `B024` ×2: abstract class without abstract methods is the test subject.
- `UP007` (`tests/unit/factory/test_config.py:16`): deliberately uses `Union[...]` to test the old syntax.

## Proposed change set (in order of value/effort)

1. Extend `per-file-ignores` for `tests/**` with `ARG`, `ANN`, `S105`, `S106`, `S311`, `NPY002`, and `BLE001` for `tests/integration/**`. This removes ~100 directives with a 5-line diff. Then run `ruff check --extend-select RUF100 --fix` to strip the now-unused comments.
2. Hoist `msg` in the 5 `PT012` tests (−5).
3. Fix `__hash__` once on the random manager base (−3).
4. Add a short reason to the kept directives (`# noqa: S301  # trusted local file`) and enable `RUF100` in the main `select` so stale directives are caught in CI. Ruff's `PGH004` already requires a code, so blanket noqa cannot sneak back in.

Expected result: 165 → ~35 directives, nearly all in `src/` with clear justification.

## Caveats

- I did not apply any change. Impact on remaining rules should be confirmed by running `ruff check .` after each step.
- The `ARG` test ignore also hides real unused fixtures/arguments in tests, so it is a trade-off.
