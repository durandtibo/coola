# Review of `# noqa` usage

Date: 2026-10-29 · Branch: `main` (7530c55a) · ruff 0.17.2 · Python ≥ 3.11 (`requires-python = ">=3.11"`, `target-version = "py311"`)

## Summary

- 215 lines with a `# noqa` directive in Python code (`src/` = 38, `tests/` = 177). The previous review counted 161 (`src/` = 51, `tests/` = 110). `src/` went down, `tests/` went up because of the new test suites (`jsonable`, `random`, `registry`, `hashing`, `summary`, `factory`, `tests/properties/`). Some mentions are in `docs/docs/dev/*.md`, which describe removed directives and are not source.
- **No directive is unused.** `ruff check --extend-select RUF100 --no-fix .` passes. `RUF` is selected in `pyproject.toml`, and `RUF100` is part of it, so a stale directive already fails CI.
- No file-level or blanket `# noqa` is used. Every directive names a rule code (`PGH004` is enabled through `PGH`).
- About **147 of the 215 (~63%) can be removed** with a few per-file ignores and small refactors. The remaining ~90 are justified or need a policy decision.
- Since the previous review, the `src/` count dropped from 51 to 38, because:
  - the `ARG002` in `hashing/`, `equality/handler/`, `io/`, `recursive/` and `iterator/` are gone (the signatures were reworked into a context object, see `docs/docs/dev/hashing.md` and `docs/docs/dev/io.md`);
  - the `PLW1641` directives were fixed (the random managers define `__hash__`);
  - `S105` / `S107` in `hashing/pydantic.py` went from 7 to 4;
  - `equality/handler/base.py` no longer exists, so its docstring mention is gone.
- New in `src/`: `PLC0415` went from 2 to 12 (deferred imports of optional libraries and of circular imports), `BLE001` from 0 to 5, plus `PLR0124` (`PLW2901` and `N807` were added and then fixed).

(Note: `ruff check --select RUF100` alone wrongly reports all directives as unused, because it disables every other rule. Always use `--extend-select`.)

## Inventory

A line with two codes (`# noqa: S311, NPY002`) counts once for each code.

| Rule                                    | Total  | src   | tests  | Verdict                                                          |
| --------------------------------------- | ------ | ----- | ------ | ---------------------------------------------------------------- |
| S311 (`random` not crypto-safe)         | 47     | 0     | 47     | Per-file ignore for tests                                        |
| NPY002 (legacy `np.random`)             | 37     | 0     | 37     | Per-file ignore for tests                                        |
| ARG002 (unused method arg)              | 27     | 10    | 17     | Keep in `src`. Per-file ignore in tests.                         |
| ARG001 (unused function arg)            | 16     | 2     | 14     | Keep in `src`. Per-file ignore in tests.                         |
| ARG005 (unused lambda arg)              | 12     | 0     | 12     | Per-file ignore in tests                                         |
| PLC0415 (import outside top level)      | 12     | 12    | 0      | Keep (lazy optional imports, circular imports)                   |
| BLE001 (blind except)                   | 15     | 5     | 10     | Keep in `src`. Per-file ignore for `tests/integration`.          |
| DTZ001 (naive `datetime`)               | 3      | 0     | 3      | Keep (naive vs aware datetime is the test subject)               |
| S105 / S106 / S107 (hardcoded password) | 9/2/2  | 2/0/2 | 7/2/0  | Mixed. See below.                                                |
| S301 (`pickle.load`)                    | 7      | 1     | 6      | Keep in `src`. Per-file ignore in tests.                         |
| S603 (`subprocess`)                     | 6      | 1     | 5      | Keep in `src`. Per-file ignore in tests.                         |
| E402 (import not at top)                | 5      | 0     | 5      | Keep (`tests/properties`, after `importorskip`)                  |
| F821 (undefined name)                   | 3      | 0     | 3      | Keep (deliberately unresolvable annotation)                      |
| ANN201                                  | 3      | 0     | 3      | Keep (tests of unannotated factories)                            |
| UP007 / UP045 / UP042                   | 2/2/2  | 0     | 6      | Keep (tests of the old `Union` / `Optional` / `str, Enum` forms) |
| PLR0124                                 | 2      | 2     | 0      | Keep (see below)                                                 |
| A001 / N801                             | 2 each | 0     | 2 each | Keep                                                             |
| TRY004                                  | 1      | 1     | 0      | Keep (see below)                                                 |

Where the tests are: `tests/unit` 132, `tests/integration` 28, `tests/properties` 12, `tests/package` 4, `tests/benchmarks` 1.

## Recommendations

### 1. Tests: per-file ignores in `pyproject.toml` (~137 directives)

`[tool.ruff.lint.per-file-ignores]` has a `tests/**` block (`D`, `S101`) and a narrow `tests/unit/hashing/test_pydantic.py = ["S106"]` entry. The `PL` rules are no longer ignored there. These rules are noise in tests and are suppressed line by line:

- `S311` (47) and `NPY002` (37): tests of the random-seed managers _must_ use `random.*` and `np.random.*` to check seeding. Add both to `tests/**`.
- `S105`, `S106` (9 in tests and 2 in `tests/unit/summary/test_pydantic.py`, `tests/integration/summary/test_interface.py`): the strings are fixtures. Add `S105` and `S106` to `tests/**` and delete the narrower `tests/unit/hashing/test_pydantic.py` entry.
- `S301` (6) and `S603` (5): tests pickle their own objects and run a fixed subprocess command. Add both to `tests/**`.
- `DTZ001` (3): was 10. The other 7 now use a timezone (`tzinfo=UTC`; the Hypothesis bounds in `tests/properties/test_equality.py` too). The 3 left (`tests/unit/equality/test_datetime.py` ×2, `tests/unit/hashing/test_collisions.py`) build a naive datetime on purpose, to test naive vs aware. Keep them; no per-file ignore needed.
- `BLE001` (10, `tests/integration/**` and `tests/package/checks.py`): catching `BaseException` / `Exception` in worker threads to forward errors to the main thread is intended. Ignore it for `tests/integration/**` and `tests/package/**`.
- `ARG001`, `ARG002`, `ARG005` (43 in tests): stubs and fakes with a signature that must match the real one. Ignoring `ARG` for `tests/**` removes all of them. Narrow alternative: keep ARG enabled and rename to `_`-prefixed names where the arg is positional (lambdas, `failing_link(src, dst)`).

Why a global ignore is acceptable: these checks guard production code. Tests are more likely to hide real problems in review than in lint noise.

### 2. `ARG002` / `ARG001` in `src` (12 directives)

This used to be 39 directives spread over `hashing/`, `equality/handler/`, `summary/`, `io/`, `recursive/` and `iterator/`. Reworking the signatures removed most of them. What remains:

- `summary/{default,mapping,metadata,object_fields,sequence,set}.py`: `summarize(data, config, *, compact)` where the implementation does not use `compact` (6).
- `summary/engine.py:131`: `_repr_str(data, level)` (1).
- `equality/tester/{pandas,polars}.py`: `same_structure(other, ctx)` (3).
- `utils/fallback/factory.py`: a fake `__init__` and a fake function accepting `*args, **kwargs` (2 × `ARG001`).

All are interface signatures shared by every implementation, so the unused argument is intended. Recommendation: keep the directives. The count is low, and `ARG002` still catches accidents in these modules. Do the per-file ignore for `tests/**` only.

### 3. `PT012` (fixed)

The three tests (`tests/unit/utils/test_env_vars.py`, `tests/unit/random/test_interface.py`, `tests/unit/random/test_base.py`) had a `with pytest.raises(...), ctx():` body of more than one statement. The `msg` is now built above the `with`, so the body is a single `raise`. In `test_random_seed_with_exception_raises_the_error` the unused `random.uniform` call was dropped (one `S311` less). No directive is left.

### 4. `S105` / `S107` in `src/coola/hashing/pydantic.py` (4)

False positives: the strings are _policy names_ (`"reveal"`, `"exclude"`, `"error"`) compared against a parameter called `on_secret`. S105/S107 trigger on the name containing "secret". Lines 114 and 154 (`S107`, default value) and 141, 143 (`S105`, comparisons). It was 7 directives; the refactor of the hasher removed three comparisons. Fix by renaming the parameter to e.g. `secret_policy`. This is a public-API rename, so it would be a breaking change. Low priority, so keep the directives unless a rename is planned.

### 5. Keep (justified, 22 directives in `src/`)

- `PLC0415` ×12: deferred imports. Two groups:
  - optional libraries imported lazily, as `CLAUDE.md` requires (`hashing/pyarrow.py` ×3, `utils/imports/optional.py` torch, `equality/_helpers/elementwise.py` jax, `random/interface.py` ×2, `equality/tester/table.py`);
  - circular-import avoidance (`equality/config.py`, `equality/tester/{registry,interface}.py`, `__init__.py` for `importlib.metadata`).

  `tests/` has no `PLC0415` left (local imports in tests are forbidden).

- `BLE001` ×5 (`utils/safe.py` ×2, `summary/engine.py` ×2, `random/registry.py:106`): the code must catch any error from user code (a `__repr__`, a manager) and carry on or re-raise after cleanup. `random/registry.py` already has a reason in the comment; the other four do not.
- `S301` `pickle.load` (`io/pickle.py:45`): the point of the loader. Add a short reason next to it.
- `S603` `subprocess.run` (`utils/git.py:59`): trusted fixed command.
- `PLR0124` ×2 (`jsonable/transformers.py:299`, `equality/_helpers/utils.py:81`): `value != value` is the NaN test. `math.isnan` does not work on arbitrary objects.
- `TRY004` (`nested/flat.py:278`): `ValueError` is the right type, not `TypeError`. Add a reason.
- Tests: `E402` ×5 (`tests/properties`, imports after `pytest.importorskip`), `F821` ×3 (deliberately undefined name, also `ty: ignore`), `ANN201` ×3, `UP007`/`UP045`/`UP042` ×6 (old syntax is the subject), `A001`/`N801` ×2 (a class named `int` shadows the builtin), `DTZ001` ×3 (naive vs aware).

## Proposed change set (in order of value/effort)

1. Extend `per-file-ignores` for `tests/**` with `ARG`, `S105`, `S106`, `S301`, `S311`, `S603` and `NPY002`, and `BLE001` for `tests/integration/**` and `tests/package/**`. This removes ~147 directives with a small diff. Then run `ruff check --extend-select RUF100 --fix` to strip the now-unused comments. Delete the `tests/unit/hashing/test_pydantic.py` entry.
2. ~~Restructure the 3 `PT012` tests (−3).~~ Done.
3. Add a short reason to the kept directives in `src` that have none (`# noqa: S301  # trusted local file`).

Expected result: 215 → ~85 directives, of which 38 are in `src/` (12 `PLC0415`, 12 `ARG001` / `ARG002`, 5 `BLE001`, 4 `S105` / `S107`, and 10 others) and ~40 are in tests where the old syntax or the unusual construct is the subject. `RUF100` is already active, so stale directives are caught in CI.

## Caveats

- I did not apply any change. Impact on remaining rules should be confirmed by running `ruff check .` after each step.
- The `ARG` test ignore also hides real unused fixtures/arguments in tests, so it is a trade-off.
- The per-rule counts come from `grep` on `# noqa`, not from ruff. The "~147 removable" figure is an estimate: only `RUF100` after the config change confirms it.
