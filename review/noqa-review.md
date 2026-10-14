# Review of `# noqa` usage

Date: 2026-10-12 · Branch: `main` · ruff 0.17.0 · Python ≥ 3.11 (`requires-python = ">=3.11"`, `target-version = "py311"`)

## Summary

- 161 `# noqa` directives in Python code (`src/` = 51, `tests/` = 110). One more mention is in a docstring (`equality/handler/base.py:89`), and some are in generated `docs/site/`, which is not source.
- **No directive is unused.** `ruff check --extend-select RUF100 --no-fix .` passes, so nothing can be deleted blindly.
- No file-level or blanket `# noqa` is used. Every directive names a rule code, which is good.
- About **104 of the 161 (~65%) can be removed** with a few config changes and small refactors. The remaining 57 are justified or need a policy decision (see below).
- Raising the minimum Python version from 3.10 to 3.11 did not change any directive: none of them depended on 3.10 behaviour, and `UP007` still fires for `Union[...]` because `X | Y` has been available since 3.10. Since the previous review, `PERF203` is no longer needed (the lock-retry loop in `io/base.py` no longer triggers it), and `src/` gained one `ARG001` pair in `utils/fallback/factory.py` that was previously not counted.

(Note: `ruff check --select RUF100` alone wrongly reports all directives as unused, because it disables every other rule. Always use `--extend-select`.)

## Inventory

| Rule                                              | Total  | src                   | tests     | Verdict                                  |
| ------------------------------------------------- | ------ | --------------------- | --------- | ---------------------------------------- |
| ARG002 (unused method arg)                        | 69     | 37                    | 32        | Keep in `src`. Per-file ignore in tests. |
| S311 (`random` not crypto-safe)                   | 27     | 0                     | 27        | Per-file ignore for tests                |
| NPY002 (legacy `np.random`)                       | 14     | 0                     | 14        | Per-file ignore for tests                |
| S105 / S107 (hardcoded password)                  | 10 / 2 | 5 / 2                 | 5 / 0     | Mixed. See below.                        |
| BLE001 (blind except)                             | 6      | 0                     | 6         | Per-file ignore for `tests/integration`  |
| ARG001 / ARG005 (unused function / lambda arg)    | 6 / 4  | 2 / 0                 | 4 / 4     | Keep in `src`. Ignore `ARG` in tests.    |
| ANN001 / ANN201                                   | 6 / 1  | 0                     | 7         | Ignore in tests, or add annotations      |
| PT012 (`pytest.raises` with multi-statement body) | 5      | 0                     | 5         | Fix by restructuring                     |
| S603                                              | 2      | 1                     | 1         | Keep (subprocess calls)                  |
| PTH108                                            | 2      | 0                     | 2         | Keep (deliberate `os.unlink` patch)      |
| PLC0415 (import outside top level)                | 2      | 2                     | 0         | Keep (circular-import avoidance)         |
| B024                                              | 2      | 0                     | 2         | Keep, or restructure the test            |
| UP007, TRY004, S301                               | 1 each | 1 each (TRY004, S301) | 1 (UP007) | Keep (see below)                         |

## Recommendations

### 1. Tests: per-file ignores in `pyproject.toml` (~99 directives)

`[tool.ruff.lint.per-file-ignores]` already has a `tests/**` block (`D`, `PL`, `S101`) and a narrow `tests/unit/hashing/test_pydantic.py = ["S106"]` entry. These rules are noise in tests and are suppressed line by line:

- `S311` (27) and `NPY002` (14): tests of the random-seed managers _must_ use `random.*` and `np.random.*` to check seeding. Add both to `tests/**`.
- `S105` (5 in `test_password.py` / `test_pydantic.py`): the test strings are fixtures. Add `S105` and `S106` to `tests/**` and delete the narrower `tests/unit/hashing/test_pydantic.py` entry.
- `BLE001` (6, `tests/integration/**`): catching `BaseException` / `Exception` in worker threads to forward errors to the main thread is intended. Ignore it for `tests/integration/**`.
- `ANN001`, `ANN201` (7): the benchmark `benchmark` fixture arg (6, `tests/benchmarks/`) and one helper in `tests/unit/factory/test_config.py`. Either add `ANN` to the tests ignore list, or annotate (the pytest-benchmark fixture type is `BenchmarkFixture`, importable under `TYPE_CHECKING`). Ignoring `ANN` in tests is the simplest option.
- `ARG001`, `ARG002`, `ARG005` (40 in tests): stubs and fakes with a signature that must match the real one. Ignoring `ARG` for `tests/**` removes all of them. Narrow alternative: keep ARG enabled and rename to `_`-prefixed names where the arg is positional (lambdas, `failing_link(src, dst)`).

Why a global ignore is acceptable: these checks guard production code. Tests are more likely to hide real problems in review than in lint noise.

### 2. `ARG002` / `ARG001` in `src` (39 directives)

Cause: the `equal(other, equal_nan)`, `summarize(registry, depth, max_depth)`, `hash(registry, ignore_unhashable)` and `handle(...)` signatures are shared by every implementation in a chain of responsibility, so implementations must accept args they do not use. The docstring in `equality/handler/base.py` explicitly decides on `# noqa: ARG002` rather than reworking the interface. The `ARG001` pair is in `utils/fallback/factory.py` (fake `__init__` / function accepting `*args, **kwargs`).

Current distribution of `ARG002`: `hashing/` 14, `equality/handler/` 7, `summary/` 7, `io/` 4, `recursive/` 3, `iterator/{bfs,dfs}/` 2.

Options, best first:

1. **Keep the directives** and accept the cost. They are accurate and documented. Reasonable if you prefer ARG002 to still catch accidents in these modules.
2. **Per-file ignore for the implementation modules** (`src/coola/equality/handler/*.py`, `src/coola/hashing/*.py`, `src/coola/summary/*.py`, `src/coola/io/*.py`, `src/coola/recursive/*.py`, `src/coola/iterator/*/*.py`). This removes about 36 noise comments but also hides _real_ unused args in these modules. Medium risk.
3. Use `del other, equal_nan` at the top of each method. This is noisier than `noqa`. Not recommended.

Recommendation: option 1 for `src`, unless the count grows. Do the per-file ignore for `tests/**` only.

### 3. `PT012` (5, tests)

Pattern: `with (pytest.raises(...), seed_ctx()):` (parenthesized form, or the single-line `with a, b:` form in the `random` tests) followed by `msg = ...; raise ...`. The extra statement is just building the message. Fix by hoisting `msg = "Exception"` above the `with`, so the body is a single `raise RuntimeError(msg)`. That removes all 5 without a config change (`tests/unit/utils/test_path.py:229`, `tests/unit/utils/test_env_vars.py:266`, `tests/unit/random/test_{torch,numpy,interface}.py`).

### 4. `PLW1641` (fixed)

The three random managers (`random/{torch,random,numpy}.py`) defined `__eq__` (`type(other) is type(self)`) without `__hash__`. Each now defines `__hash__` returning `hash(type(self))`, consistent with `__eq__`, and the three directives are removed. `ruff check --preview --select PLW1641` is clean.

### 5. `S105` / `S107` in `src/coola/hashing/pydantic.py` (7)

False positives: the strings are _policy names_ (`"reveal"`, `"exclude"`, `"error"`) compared against a parameter called `on_secret`. Rule S105/S107 triggers on the name containing "secret". Lines 61, 200 (`S107`, default value) and 160, 163, 169, 180, 188 (`S105`, comparisons). Fix by renaming the parameter to e.g. `secret_policy` / `secret_mode`. This is a public-API rename, so it would be a breaking change. Alternative: define a `Literal` alias plus constants and compare against those, which does not fix S107 for the default value. Low priority, so keep the directives unless a rename is planned.

### 6. Keep (justified, 11 directives)

- `S301` `pickle.load` (`io/pickle.py:52`): the point of the loader. Add a short reason next to it.
- `S603` `subprocess.run` / `Popen` (`utils/git.py:59`, `tests/integration/io/test_base.py:187`): trusted fixed command.
- `PLC0415` ×2 (`equality/config.py:26`, `registry/base.py:178`): deferred imports to avoid circular imports. The `registry.base -> equality.interface` one is also registered in importlinter `ignore_imports`.
- `TRY004` (`nested/flat.py:114`): `ValueError` is the right type, not `TypeError`. Add a reason.
- `PTH108` ×2 (`tests/unit/io/test_base.py`): tests deliberately call the original `os.unlink`.
- `B024` ×2 (`tests/unit/registry/test_type.py`): abstract class without abstract methods is the test subject.
- `UP007` (`tests/unit/factory/test_config.py:16`): deliberately uses `Union[...]` to test the old syntax. Python 3.11 does not change this.

## Proposed change set (in order of value/effort)

1. Extend `per-file-ignores` for `tests/**` with `ARG`, `ANN`, `S105`, `S106`, `S311`, `NPY002`, and `BLE001` for `tests/integration/**`. This removes ~99 directives with a small diff. Then run `ruff check --extend-select RUF100 --fix` to strip the now-unused comments.
2. Hoist `msg` in the 5 `PT012` tests (−5).
3. ~~Fix `__hash__` on the random managers (−3).~~ Done.
4. Add a short reason to the kept directives (`# noqa: S301  # trusted local file`) and enable `RUF100` in the main `select` so stale directives are caught in CI. Ruff's `PGH004` already requires a code, so blanket noqa cannot sneak back in.

Expected result: 161 → ~57 directives, all in `src/` except 6 justified ones in tests. Of the 51 in `src/`, 39 are `ARG001` / `ARG002` (shared interface signatures) and 7 are the `pydantic` `S105` / `S107` false positives.

## Caveats

- I did not apply any change. Impact on remaining rules should be confirmed by running `ruff check .` after each step.
- The `ARG` test ignore also hides real unused fixtures/arguments in tests, so it is a trade-off.
