# Python 3.11 modernization review

Scope: `src/coola` (207 files) after the bump to `requires-python = ">=3.11"`
(`target-version = "py311"` in ruff).

Method: grep-based checks plus reading the relevant code. After implementing the changes
below I ran `ruff check`, `ruff format` and the unit tests (all pass). **mypy is not
installed here and the repo has no type-checker config, so none of the typing changes
(`Literal`, `assert_never`, `Never`) have been type-checked.**

The codebase was already modern: `from __future__ import annotations` + `X | Y`
everywhere, no `typing.List/Dict/Tuple`, no `typing_extensions`, no `sys.version_info`
branches, no `os.path`. The wins were small.

Note: PEP 695 syntax (`class Foo[T]`, `type X = ...`) is **3.12**, so the ~94
`TypeVar(...)` declarations must stay until the floor moves to 3.12.

## Status summary

| #     | Item                                               | Status                                                         |
| ----- | -------------------------------------------------- | -------------------------------------------------------------- |
| 1     | `zip(strict=True)`, drop `B905` ignore             | Done                                                           |
| 2     | `typing.Self`                                      | Done where it applied (see below); otherwise "use in new code" |
| 3     | `NoReturn` → `Never`                               | Done (src and tests)                                           |
| 4     | `assert_never` on `Literal` branches               | Done, with unit tests                                          |
| 5     | `ExceptionGroup` / `add_note` in plugin loader     | Done, with unit tests                                          |
| 6     | `dataclass(slots=True)`                            | Done (`EqualityConfig`, `ComparisonResult`)                    |
| 7     | `match` statements                                 | Skipped on purpose                                             |
| 8     | `isinstance(x, A \| B)`                            | Skipped on purpose                                             |
| 9     | `tomllib`                                          | Nothing to do                                                  |
| 10–13 | `StrEnum`, `LiteralString`, `Required`, asyncio    | Not applicable                                                 |
| 14    | Ruff rule sets                                     | Nothing to do (already enabled)                                |
| 15    | CI matrix / classifiers                            | Partly verified; CI matrix lives outside the repo              |
| 16    | Docs wording                                       | Done (3 pages)                                                 |
| F4-5  | Slots benchmark, `functools.cache`, `Generator[X]` | Done                                                           |

## 1. Implemented

1. **`zip(..., strict=True)`.** Added to `equality/handler/sequence.py` (lengths are
   checked just above) and to both `zip` calls in `nested/conversion.py` (lengths are
   validated before). Removed the stale `B905` ignore from `pyproject.toml`, so new code is
   checked.

2. **`typing.Self`.** No library method returns its own class by name, so there was nothing
   to convert in `src/`. It is now used in the test helper `_PassesValidationOnly.__new__`
   (`tests/unit/nested/test_mapping.py`), at ruff's suggestion. Use it in new code.

3. **`NoReturn` → `Never`.** Replaced in 15 `src` files and 5 test files; imports remain
   sorted.

4. **`assert_never` for exhaustive `Literal` branches.**
   - `hashing/pydantic.py` (`unwrap_secrets`): the `"error"` case is now an explicit
     branch, followed by `assert_never(on_secret)`.
   - `nested/mapping.py`: `on_duplicate` is now typed as a `Literal` in `flatten_mapping`
     (`raise|first|last|prefix`) and `merge_mappings` (`raise|first|last|suffix`), each with
     an `assert_never` for the last case. Runtime validation is unchanged.
   - Behavior notes: an invalid `on_secret` passed directly to `unwrap_secrets` now raises
     `AssertionError` instead of `ValueError`. Callers passing a plain `str` for
     `on_duplicate` will fail type checking.
   - The `assert_never` lines are unreachable through the public API (validation runs
     first), so the tests reach them with an invalid value (`unwrap_secrets`) and a
     `str` subclass that passes validation once (`nested/mapping.py`).

5. **`ExceptionGroup` / `add_note` in the plugin loader (`utils/singleton.py`).** In strict
   mode (`COOLA_STRICT_PLUGINS`) every plugin is now attempted and all failures are raised
   together as one `ExceptionGroup`, each annotated with the plugin name and group.
   `_load_plugin` now appends to an `errors` list instead of re-raising, which also removed
   the need for a `# noqa: BLE001`. Non-strict mode (warn and skip) is unchanged.
   **Breaking for strict mode:** a single failing plugin now raises `ExceptionGroup`, not the
   original exception, so `except RuntimeError` around it no longer catches it.
   The strict-mode tests use `pytest.RaisesGroup` / `pytest.RaisesExc` to assert the group
   message, the member exceptions and the per-plugin note.

6. **`@dataclass(slots=True)`.** Applied to `EqualityConfig` and `ComparisonResult` (the
   latter as `frozen=True, slots=True`). Nothing in `src` relied on `__dict__` or ad-hoc
   attributes. `kw_only=True` was not applied because it would break positional callers.
   Micro-benchmark (5 runs, best): creating 100k `EqualityConfig` took 0.039 s with slots vs 0.043 s
   without (~10% faster); the instance has no `__dict__` (88 B vs 48 B plus a dict). Comparison
   speed was unchanged within noise.

7. **Small idioms (not 3.11-specific).** Zero-argument `@lru_cache(maxsize=1)` / `@lru_cache(1)`
   functions (`utils/tensor.py` x3, `equality/tester/jax.py`, `identifier/objectid.py`,
   `identifier/snowflake.py`, `nested/mapping.py`) now use `@functools.cache`; `cache_clear()`
   is still available, which the tests rely on. `Generator[X, None, None]` annotations were
   shortened to `Generator[X]` (already the style in `utils/path.py`). The functions with
   arguments (`get_package_version`, polars `is_new_naming`) keep `lru_cache`.

8. **Docs.** `docs/docs/get_started.md`, `docs/docs/dev/development.md` and
   `docs/docs/dev/architecture.md` now say Python 3.11. Left alone on purpose: older rows of
   the README compatibility table (they record past releases), the sample environment in
   `CONTRIBUTING.md`, and the generated, git-ignored `docs/site/`.

## 2. Reviewed and deliberately not changed

- **`match` statements.** The `if` chains are clearer, and `assert_never` already gives
  exhaustiveness checking.
- **`isinstance(x, A | B)`.** About 10 tuple sites remain. The union form is slower at
  runtime and ruff's `UP038` was deprecated for that reason. Keep the tuples.
- **`tomllib`.** The only TOML reference outside `src` is a `pyproject.toml` path in
  `dev/generate_versions.py`, which does not parse it with a third-party package.
- **`StrEnum`, `LiteralString`, `Required`/`NotRequired`, `asyncio.TaskGroup`.** No `Enum`,
  `TypedDict`, SQL-like string building or asyncio in `src/`. Options are `Literal` strings,
  which should stay. If an enum is ever introduced, use `StrEnum` rather than `(str, Enum)`.
- **Ruff.** `UP`, `FURB` and `PERF` are already enabled and `UP` has no ignores.

## 3. Open follow-ups

1. **CI Python matrix.** It is read from the external `durandtibo/workflow-config-action`
   (see `.github/workflows/lib-get-test-matrix.yaml`), so it is not in this repo. Confirm
   that shared config no longer lists 3.10. `pyproject.toml` classifiers (3.11–3.14) and the
   README `main` row (`>=3.11`) are already correct.
2. **Configure a type checker.** `pyproject.toml` has no mypy/pyright section and CI does
   not appear to run one. Without it, `assert_never`, the new `Literal` parameters and
   `Never` are not enforced. Adding mypy (or pyright) with `python_version = "3.11"` would
   make these changes pay off, and would also check the `# type: ignore[arg-type]` comments
   in the new tests.
3. **Simplify the `_PassesValidationOnly` test helper.** It is a hack needed only because
   `nested/mapping.py` validates `on_duplicate` up front _and_ has a defensive branch. One
   option is to drop the up-front set check and let the final `else: assert_never` /
   `ValueError` raise. That would change when an invalid value is reported (only when a
   conflict occurs), so the current approach is the safer one. Revisit only if the helper
   becomes a burden.
4. **Prepare for 3.12 (when the floor moves).** Convert `TypeVar` declarations and
   `Generic[...]` classes to PEP 695 syntax, use `type X = ...` aliases, add
   `@typing.override` on the many `abstractmethod` implementations (44 abstract methods),
   and consider `itertools.batched` for `utils.batching.batchify` (it would still need the
   `size >= 1` validation and a tuple-yielding wrapper).

## Suggested order for what remains

1. Follow-ups 1 (confirm CI matrix) and 2 (type checker). They are cheap and verify the rest.
2. Follow-up 4 when the project drops 3.11.
