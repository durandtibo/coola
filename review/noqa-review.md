# Review of `# noqa` usage

Date: 2026-10-29 · Branch: `main` (7530c55a) · ruff 0.17.2 · Python ≥ 3.11 (`requires-python = ">=3.11"`, `target-version = "py311"`)

## Summary

- 103 lines with a `# noqa` directive in Python code (`src/` = 34, `tests/` = 69). The previous review counted 161 (`src/` = 51, `tests/` = 110). `src/` went down, `tests/` went up because of the new test suites (`jsonable`, `random`, `registry`, `hashing`, `summary`, `factory`, `tests/properties/`). Some mentions are in `docs/docs/dev/*.md`, which describe removed directives and are not source.
- **No directive is unused.** `ruff check --extend-select RUF100 --no-fix .` passes. `RUF` is selected in `pyproject.toml`, and `RUF100` is part of it, so a stale directive already fails CI.
- No file-level or blanket `# noqa` is used. Every directive names a rule code (`PGH004` is enabled through `PGH`).
- About **40 of the 103 (~39%) can be removed** with a few per-file ignores and small refactors. The remaining ~63 are justified or need a policy decision.
- Since the previous review, the `src/` count dropped from 51 to 34, because:
  - the `ARG002` in `hashing/`, `equality/handler/`, `io/`, `recursive/` and `iterator/` are gone (the signatures were reworked into a context object, see `docs/docs/dev/hashing.md` and `docs/docs/dev/io.md`);
  - the `PLW1641` directives were fixed (the random managers define `__hash__`);
  - `S105` / `S107` in `hashing/pydantic.py` went from 7 to 0 (see section 4);
  - `equality/handler/base.py` no longer exists, so its docstring mention is gone.
- New in `src/`: `PLC0415` went from 2 to 12 (deferred imports of optional libraries and of circular imports), `BLE001` from 0 to 5, plus `PLR0124` (`PLW2901` and `N807` were added and then fixed).

(Note: `ruff check --select RUF100` alone wrongly reports all directives as unused, because it disables every other rule. Always use `--extend-select`.)

## Inventory

A line with two codes (`# noqa: A001, N801`) counts once for each code.

| Rule                               | Total  | src | tests  | Verdict                                                          |
| ---------------------------------- | ------ | --- | ------ | ---------------------------------------------------------------- |
| S311 (`random` not crypto-safe)    | 3      | 0   | 3      | Keep (one shared helper, see below)                              |
| NPY002 (legacy `np.random`)        | 5      | 0   | 5      | Keep (shared helpers, see below)                                 |
| ARG002 (unused method arg)         | 27     | 10  | 17     | Keep in `src`. Per-file ignore in tests.                         |
| ARG001 (unused function arg)       | 16     | 2   | 14     | Keep in `src`. Per-file ignore in tests.                         |
| PLC0415 (import outside top level) | 12     | 12  | 0      | Keep (lazy optional imports, circular imports)                   |
| BLE001 (blind except)              | 7      | 5   | 2      | Keep (see below)                                                 |
| DTZ001 (naive `datetime`)          | 3      | 0   | 3      | Keep (naive vs aware datetime is the test subject)               |
| S301 (`pickle.load`)               | 7      | 1   | 6      | Keep in `src`. Per-file ignore in tests.                         |
| S603 (`subprocess`)                | 6      | 1   | 5      | Keep in `src`. Per-file ignore in tests.                         |
| F821 (undefined name)              | 3      | 0   | 3      | Keep (deliberately unresolvable annotation)                      |
| ANN201                             | 3      | 0   | 3      | Keep (tests of unannotated factories)                            |
| UP007 / UP045 / UP042              | 2/2/2  | 0   | 6      | Keep (tests of the old `Union` / `Optional` / `str, Enum` forms) |
| PLR0124                            | 2      | 2   | 0      | Keep (see below)                                                 |
| A001 / N801                        | 2 each | 0   | 2 each | Keep                                                             |
| TRY004                             | 1      | 1   | 0      | Keep (see below)                                                 |

Where the tests are: `tests/unit` 53, `tests/integration` 5, `tests/properties` 1, `tests/package` 2, `tests/benchmarks` 1.

## Recommendations

### 1. Tests: per-file ignores in `pyproject.toml` (~40 directives)

`[tool.ruff.lint.per-file-ignores]` has a `tests/**` block (`D`, `S101`) (the narrow `tests/unit/hashing/test_pydantic.py = ["S106"]` entry was removed, because its 15 hidden `S106` are fixed). The `PL` rules are no longer ignored there. These rules are noise in tests and are suppressed line by line:

- `S311` (fixed, was 47): the tests of the random-seed managers must draw from `random` to check seeding. They now call `draw_python_random()` and `make_python_generator(seed)` from `tests/helpers.py`, which carry the only 2 directives. `tests/package/checks.py` is standalone (it does not import `tests`), so it has its own `_draw_python_random()` helper (1 directive).
- `NPY002` (fixed, was 37): same approach for `np.random.*`. The tests call `draw_numpy_rand`, `draw_numpy_randn`, `draw_numpy_randint`, `get_numpy_state` and `seed_numpy` from `tests/helpers.py`, which carry the only 5 directives (numpy is imported there only if available).
- `S105`, `S106` (fixed, was 9 + 2 in tests, plus 15 hidden by the per-file entry): the password fixtures use names the rule does not match (`with_spaces`, `special_chars`), objects are built with `Model.model_validate({...})` (and `make_creds()` in `test_pydantic.py`) instead of `password="..."`, and the private `_on_secret` checks became `repr(hasher) == "SecretHasher(on_secret='error')"`.
- `S301` (6) and `S603` (5): tests pickle their own objects and run a fixed subprocess command. Add both to `tests/**`.
- `DTZ001` (3): was 10. The other 7 now use a timezone (`tzinfo=UTC`; the Hypothesis bounds in `tests/properties/test_equality.py` too). The 3 left (`tests/unit/equality/test_datetime.py` ×2, `tests/unit/hashing/test_collisions.py`) build a naive datetime on purpose, to test naive vs aware. Keep them; no per-file ignore needed.
- `BLE001` (fixed in `tests/integration`, was 8): the worker threads and forked processes caught `BaseException` / `Exception` to forward the error to the main thread. They now use `with collect_errors(errors):` from `tests/integration/helpers.py`, which carries the only directive. `tests/package/checks.py` keeps its own (it collects import failures of any kind and does not import `tests`).
- `ARG005` (fixed, was 12): the unused lambda arguments are renamed to `_` (`lambda _: False`). `entry_points` is called with `group=`, so that stub is `lambda **_: eps`.
- `ARG001`, `ARG002` (31 in tests): stubs and fakes with a signature that must match the real one. Ignoring `ARG` for `tests/**` removes all of them. Narrow alternative: keep ARG enabled and rename to `_`-prefixed names where the arg is positional (`failing_link(src, dst)`).

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

### 4. `S105` / `S107` in `src/coola/hashing/pydantic.py` (fixed)

The strings are _policy names_ (`"reveal"`, `"exclude"`, `"error"`) compared with, and defaulted to, a parameter called `on_secret`, and the rules trigger on the name containing "secret". Renaming the parameter would break the public API, so the policies are now private constants (`_REVEAL`, `_EXCLUDE`, `_ERROR`, `Final`), used for the comparisons and the defaults. The signature, the `Literal` type and the `repr` are unchanged. The four directives are gone.

### 5. Keep (justified, 22 directives in `src/`)

- `PLC0415` ×12: deferred imports. Two groups:
  - optional libraries imported lazily, as `CLAUDE.md` requires (`hashing/pyarrow.py` ×3, `utils/imports/optional.py` torch, `equality/_helpers/elementwise.py` jax, `random/interface.py` ×2, `equality/tester/table.py`);
  - circular-import avoidance (`equality/config.py`, `equality/tester/{registry,interface}.py`, `__init__.py` for `importlib.metadata`).

  `tests/` has no `PLC0415` left (local imports in tests are forbidden).

- `BLE001` ×5 (`utils/safe.py` ×2, `summary/engine.py` ×2, `random/registry.py:106`): the code must catch any error from user code (a `__repr__`, a manager) and carry on or re-raise after cleanup. Each carries a short reason.
- `S301` `pickle.load` (`io/pickle.py:45`): the point of the loader. Add a short reason next to it.
- `S603` `subprocess.run` (`utils/git.py:59`): trusted fixed command.
- `PLR0124` ×2 (`jsonable/transformers.py:299`, `equality/_helpers/utils.py:81`): `value != value` is the NaN test. `math.isnan` does not work on arbitrary objects.
- `TRY004` (`nested/flat.py:278`): `ValueError` is the right type, not `TypeError`. Add a reason.
- Tests: `F821` ×3 (deliberately undefined name, also `ty: ignore`), `ANN201` ×3, `UP007`/`UP045`/`UP042` ×6 (old syntax is the subject), `A001`/`N801` ×2 (a class named `int` shadows the builtin), `DTZ001` ×3 (naive vs aware).

## Proposed change set (in order of value/effort)

1. Extend `per-file-ignores` for `tests/**` with `ARG`, `S301` and `S603`. This removes ~40 directives with a small diff. Then run `ruff check --extend-select RUF100 --fix` to strip the now-unused comments. Delete the `tests/unit/hashing/test_pydantic.py` entry.
2. ~~Restructure the 3 `PT012` tests (−3).~~ Done.
3. Add a short reason to the kept directives in `src` that have none (`# noqa: S301  # trusted local file`).

Expected result: 103 → ~63 directives, of which 34 are in `src/` (12 `PLC0415`, 12 `ARG001` / `ARG002`, 5 `BLE001`, and 10 others) and ~29 are in tests where the old syntax or the unusual construct is the subject. `RUF100` is already active, so stale directives are caught in CI.

## Caveats

- I did not apply any change. Impact on remaining rules should be confirmed by running `ruff check .` after each step.
- The `ARG` test ignore also hides real unused fixtures/arguments in tests, so it is a trade-off.
- The per-rule counts come from `grep` on `# noqa`, not from ruff. The "~40 removable" figure is an estimate: only `RUF100` after the config change confirms it.
