# Python 3.11 modernization review

Scope: `src/coola` (207 files) after the bump to `requires-python = ">=3.11"`
(`target-version = "py311"` in ruff).

Method: grep-based checks plus reading the relevant code. After implementing the changes
below I ran `ruff check`, `ruff format` and the unit tests (all pass). **The typing
changes (`Literal`, `assert_never`, `Never`) were not type-checked in that session.** The
repo does have a type checker, [`ty`](https://github.com/astral-sh/ty) (`[tool.ty.src]` in
`pyproject.toml`, run by `.github/workflows/ci-type-checking.yaml` plus `inv types.check`),
so run `ty check` to verify them (the earlier version of this document wrongly said there
was none).

The codebase was already modern: `from __future__ import annotations` + `X | Y`
everywhere, no `typing.List/Dict/Tuple`, no `typing_extensions`, no `sys.version_info`
branches, no `os.path`. The wins were small.

Note: PEP 695 syntax (`class Foo[T]`, `type X = ...`) is **3.12**, so the ~94
`TypeVar(...)` declarations (51 real ones, in 14 `Generic[...]` classes and many generic
functions) must stay until the floor moves to 3.12.

A second full pass over `src/`, `tests/`, `dev/`, `debug/`, `pyproject.toml` and the CI
workflows was done afterwards; its extra findings are in section 4.

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
   that shared config no longer lists 3.10. `pyproject.toml` classifiers (3.11–3.15) and the
   README `main` row (`>=3.11`) are already correct.
2. **Run `ty check` against the typing changes.** A type checker is already configured
   (`ty`, CI job "Type Checking"), so no new tool is needed. Confirm that `ty` passes with the
   `assert_never`, `Literal` and `Never` changes, and that the `# type: ignore[arg-type]`
   comments in the new tests are understood by `ty` (it honours `# type: ignore` but prefers
   `# ty: ignore[...]`). `[tool.ty]` has no `environment.python-version`; add
   `python-version = "3.11"` so `ty` checks against the floor and not the interpreter in use.
3. **Simplify the `_PassesValidationOnly` test helper.** It is a hack needed only because
   `nested/mapping.py` validates `on_duplicate` up front _and_ has a defensive branch. One
   option is to drop the up-front set check and let the final `else: assert_never` /
   `ValueError` raise. That would change when an invalid value is reported (only when a
   conflict occurs), so the current approach is the safer one. Revisit only if the helper
   becomes a burden.
4. **Prepare for 3.12 (when the floor moves).** Convert `TypeVar` declarations and
   `Generic[...]` classes to PEP 695 syntax, use `type X = ...` aliases, add
   `@typing.override` on the implementations of the 33 `@abstractmethod`s (none use it today),
   and consider `itertools.batched` for `utils.batching.batchify` (it would still need the
   `size >= 1` validation and a tuple-yielding wrapper).

## 4. Additional suggestions from a full-codebase pass

Found by grepping and reading all of `src/` (207 files), plus `tests/`, `dev/`, `debug/`,
`pyproject.toml` and the workflows. All of these work on 3.11. Ordered by value/effort.

### A. `Path.open(path, ...)` → `path.open(...)` (6 sites) — Done

`io/text.py:53,96`, `io/json.py:46,90`, `io/pickle.py:51,89` call the unbound method
`Path.open(path, mode=...)`. It works but reads oddly and bypasses subclass dispatch. Use
`path.open(...)` (or `path.read_text/write_text/read_bytes/write_bytes` where there is no
special handling). Add an `encoding=` to any text open that lacks one (`EncodingWarning`
is a 3.10+ check; `PLW1514` in ruff enforces it).

### B. Reduce the `cast(...)` workarounds (19 casts) — Partly done

Done: the four `cast("Callable[..., Any]", type(data))` copies in `recursive/` now use the new
`utils.introspection.get_type_constructor`. The other casts (optional-dependency `cast("Any", ...)`)
were left: they are needed by `ty` and removing them is a separate experiment.

`equality/handler/pandas.py:137-180`, `equality/tester/interface.py:330`,
`equality/tester/jax.py:107`, `utils/tensor.py:112` and `display/pydantic.py:73` use
`cast("Any", pd)` to silence the checker on optional dependencies. These are
workarounds for optional dependencies, not 3.11 features. Review which are still needed
once `ty` runs clean (see follow-up 2); some may be removable. The
`cast("Callable[..., Any]", type(data))` pattern repeated in `recursive/{set,sequence,mapping,key}.py`
could become one shared helper (or a `Protocol` with `__call__`) instead of four copies.

### C. `typing.Union` in `factory/config.py` — Reviewed, kept

`factory/config.py:14,55` imports `Union` only to compare `get_origin(target)` with
`(Union, UnionType)`. That is correct and must stay for `typing.Union[...]` annotations
coming from user code, so keep it, but make the intent explicit (the comment already
says so). No other `Optional`/`Union`/`List`/`Dict` exist in `src/` or `tests/`.

### D. Mark overrides with `@typing.override` — Blocked (3.12)

0 uses today across 33 abstract methods and every subclass of `BaseTester`, `BaseHasher`,
`BaseTransformer`, `BaseLoader`/`BaseSaver`, `BaseReducer`, `BaseSummarizer`, `BaseHandler`,
`BaseRandomManager` and the `Registry` classes. When the floor becomes 3.12 add `@override` from `typing`.
Do not backport it with `typing_extensions` now: `dependencies = []` is a feature.

### E. Dedupe the 11 `utils/imports/<pkg>.py` modules — Decided against

`numpy.py`, `torch.py`, `pandas.py`, `polars.py`, `jax.py`, `pyarrow.py`, `pydantic.py`,
`packaging.py`, `colorlog.py`, `rich.py`, `xarray.py` are near-identical (96 lines each; they
differ only in the package name and the `__all__` order). Each repeats `F = TypeVar("F",
bound="Callable[..., Any]")`, a `@lru_cache(1)` availability check, `check_*`,
`*_available`, `raise_*_missing_error`. A small factory in `utils/imports/universal.py`
(for example `make_package_helpers("numpy")`) would remove ~1000 lines. **Decision: not done.** The
modules are the public, statically named API (`check_numpy`, `is_numpy_available`, ...) re-exported
from `utils/imports/__init__.py`, with per-function docstrings and doctests. Generating them from a
factory would hide those names from `ty` and IDEs and change the docs. Changing the decorator's
return type to `Callable[P, R | None]` would also make every decorated function look `Optional` to
callers. The duplication is the price of explicit names; revisit only with a stub-generating approach.
Original note: this is not purely a modernization item, but it is the largest remaining duplication. On 3.12 the repeated
`F` becomes `ParamSpec`-based (`def decorator[**P, R](fn: Callable[P, R]) -> Callable[P, R | None]`),
which also removes the `cast("F", inner)` in `universal.py:136`. On 3.11 the same can be
done now with `ParamSpec` + `TypeVar` from `typing` (3.10+); the `@wraps` wrapper then returns
`Callable[P, R | None]`, which is truthfully typed because the wrapper returns `None` when
the package is missing. The current `F`-typed signature claims the return type is unchanged.

### F. `@functools.cache` for the remaining zero-argument `@lru_cache(1)` (11 files) — Done

The doc's section 1 item 7 converted 7 sites, but all the `utils/imports/<pkg>.py` files still
use `@lru_cache(1)` (`pyarrow`, `pandas`, `numpy`, `torch_numpy`, `polars`, `pydantic`,
`packaging`, `jax`, `torch`, `colorlog`, `rich`, `xarray`). These are the same zero-argument
case; switch to `@cache`. `utils/imports/universal.py:25,50` use bare `@lru_cache`
(default `maxsize=128`) on functions taking a `str`; they are fine, but `UP033` /
`FURB` would flag `@lru_cache(maxsize=None)` if it ever appears. Tests that call
`cache_clear()` keep working.

### G. Logging: lazy `%s` formatting instead of f-strings (7 sites) — Done

Done: 12 `src/` sites converted to lazy `%s` arguments, `logger.error(..., exc_info=True)` became
`logger.exception(...)` in `utils/singleton.py`, and ruff `G` is enabled (with `G004` ignored for the
`debug/` and `dev/` scripts). The `format_value_difference(...)` calls are already guarded by
`config.show_difference`, so they were left as they are.

`io/base.py:84`, `utils/env_vars.py:58,60`, `equality/handler/native.py:174,205` and
others pass an f-string to `logger.info/warning`. This formats the message even when the
level is disabled; `logger.info("... %s", value)` defers it. Ruff `G004` flags it; the rule
set `LOG` is on but `G` is not in `lint.select`. Enable `G` (flake8-logging-format). The
`equality/handler/*.py` sites that call `format_value_difference(...)` eagerly are the
hot ones: wrap in `if logger.isEnabledFor(logging.INFO):` or pass a lazy object, since they
run on every mismatch during equality checks.

### H. Small readability items — Mostly done

Done: `_iter_items` helper in `utils/format.py`, `removesuffix` in `io/utils.py`,
`next(iter(data.values()))` in `nested/mapping.py`. Not done: the `isinstance(other, type(self))` clause
is redundant at runtime but narrows `other` for the type checker, so it stays; `time_ns() // 1_000_000`
appears only twice in `snowflake.py`, not worth a helper.

- Time sources are already correct: `perf_counter` for elapsed time, `monotonic` for
  deadlines, `time.time()` only where compared with file mtimes or ID timestamps.
- `io/utils.py:41-48` builds the extension with `"".join(path.suffixes)[1:]` and then slices
  the name by length. Using `path.name.removesuffix(extension)` (3.9+) is clearer.
- `utils/format.py:136,168,286,318` repeat `sorted(mapping.items()) if sorted_keys else
  mapping.items()` four times; extract a helper.
- `equality/handler/mixin.py:117`, `equality/tester/handler.py:62`, `registry/base.py:180`
  and `summary/collection.py:64` repeat `type(other) is not type(self) or not
  isinstance(other, type(self))`. The second clause is redundant once the first holds;
  keep one.
- `nested/mapping.py:212` `data[next(iter(data))]` can be `next(iter(data.values()))`.
- `identifier/*`: `time.time_ns() // 1_000_000` appears three times (`snowflake.py`);
  extract a small `_now_ms()`.
- `debug/equality_examples.py`: `logger.info(f"...")` and module-level conditional imports;
  fine for a scratch script, excluded from the above.

### I. Checked and nothing to change

- `datetime`: already uses `datetime.UTC` (3.11) and `ZoneInfo`; no `utcnow`/`utcfromtimestamp`.
- `contextlib`: `chdir` (3.11) is already used in `utils/path.py`; `suppress` is used in place
  of `try/except/pass` everywhere.
- `raise ... from`: all re-raises chain correctly (`from error`/`from None`).
- `zip`: every call has `strict=`.
- `dataclass`: only 2 dataclasses exist, both already `slots=True`. Plain classes with
  `__init__` (`TimingResult`, `LRUCache`, the `random` managers) are mutable or define
  custom `__eq__`/`__hash__`, so converting them gains nothing.
- `Final`: used twice (`MAX_PASSWORD_LENGTH`, `DEFAULT_DATE_FORMAT`); other constants are
  private `_UPPER_CASE` names and could take `Final` but ruff does not require it.
- `Protocol`: 12 protocols, all already `runtime_checkable` where `isinstance` is used.
- `TypeAlias`/`TypedDict`/`NamedTuple`/`Enum`: none, nothing to convert.
- `tomllib`, `asyncio.TaskGroup`, `StrEnum`: no use case (see section 2).
- Ruff `lint.select` already includes `UP`, `FURB`, `PERF`, `PTH`, `SIM`, `RUF`, `TC`, `LOG`,
  `TRY`; `target-version = "py311"` is set. `G` (logging format) is the only
  relevant family that is not selected (`FA` is already selected).
- Tests: `tmp_path` fixtures and `monkeypatch` are used throughout; `tmpdir` and
  `unittest.mock` appear in only 3 files (`tests/unit/io/test_utils.py`, `test_base.py`, guide).
  `pytest.RaisesGroup` is used for the new plugin tests. No deprecated typing imports in tests.

## Suggested order for what remains

1. Follow-ups 1 (confirm CI matrix) and 2 (`ty check`). They are cheap and verify the rest.
2. Section 4, items A–C (cheap, behaviour-preserving, 3.11-compatible).
3. Section 4, items D–H as time allows.
4. Follow-up 4 when the project drops 3.11.
