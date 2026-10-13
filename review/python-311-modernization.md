# Python 3.11 modernization review

Scope: `src/coola` (207 files) after the bump to `requires-python = ">=3.11"`
(`target-version = "py311"` in ruff). I used grep-based checks and read the relevant spots.
I did not run tests, ruff or mypy. Treat each item as a lead to verify.

Overall the codebase is already modern: `from __future__ import annotations` + `X | Y`
everywhere, no `typing.List/Dict/Tuple`, no `typing_extensions`, no `sys.version_info`
branches, no `os.path`. There is little dead compat code left. The wins below are small.

Note: PEP 695 syntax (`class Foo[T]`, `type X = ...`) is **3.12**, so the ~94
`TypeVar(...)` declarations must stay until the floor moves to 3.12.

## 1. Worth doing

1. **`zip(..., strict=...)` and drop the `B905` ignore.** `pyproject.toml:194` ignores B905
   with a comment about 3.10, which no longer applies. Two call sites lack `strict`:
   - `src/coola/equality/handler/sequence.py:69` — lengths are checked just above, so
     `strict=True` is a free safety net (use `strict=True`).
   - `src/coola/nested/conversion.py:86` — `dict(zip(mapping_of_seqs, seqs))` pairs keys with
     one value per sequence; `strict=True` is correct. The outer `zip(*values)` is
     intentionally truncating only if lengths may differ; check the docstring and decide
     (probably `strict=True` too, since unequal lengths silently drop data).
     Then remove the ignore so new code is checked.

2. **`typing.Self` for fluent/copy-style returns.** Available since 3.11. Grep found no
   existing `Self` use. Candidates are methods annotated with the concrete class name or
   `-> "ClassName"` (e.g. `clone`/`copy`-like methods, `__enter__`, registry `__new__`
   helpers). I found few obvious hits, so this is a "use it going forward" item; check
   `utils/lru.py`, `utils/bloom_filter.py:84` (`@classmethod` constructor) and the `io`
   serializer classes, which return `self._kwargs`/copies.

3. **`NoReturn` → `Never`.** 15 files import `NoReturn` (mostly `utils/imports/*` and
   `utils/fallback/*`). `typing.Never` (3.11) is the preferred spelling for the bottom
   type. Purely cosmetic; no behavior change. Low priority, but a mechanical sed.

4. **`typing.assert_never` for exhaustive `Literal` branches.** `hashing/pydantic.py:160-186`
   and `nested/mapping.py:119` branch on `on_secret` / `on_duplicate` string literals. If the
   final branch is an implicit fall-through or a generic `ValueError`, adding
   `assert_never(x)` after the last `Literal` case gives mypy exhaustiveness checking when
   a new option is added.

5. **`ExceptionGroup` / `add_note` in the plugin loader.** `utils/singleton.py:_load_plugin`
   catches each plugin failure separately and then warns or re-raises (strict mode).
   In strict mode it aborts on the first failure. Collecting failures and raising one
   `ExceptionGroup("plugin load failed", errors)` would report all broken plugins at once.
   Alternatively `exc.add_note(f"while loading plugin {name!r} of group {group!r}")` before
   re-raising keeps the original traceback but adds the context that is now only in the log
   line. This is a behavior change for strict mode, so check tests and docs first.

## 2. Optional / judgement calls

6. **`@dataclass(slots=True, kw_only=True)`** (3.10). `EqualityConfig`
   (`equality/config.py:31`) and the frozen result dataclass (`equality/result.py:29`) can
   take `slots=True` for memory and a typo guard on attributes. `EqualityConfig` is created
   per comparison, so `slots=True` is a cheap win. Check that nothing sets ad-hoc attributes
   or relies on `__dict__`, and that pickling or `copy` still work. `kw_only=True` would be
   a public API break for positional callers; skip unless intended.

7. **`match` statements** (3.10) for `on_secret` / `on_duplicate` string dispatch. Readable,
   but the current `if` chains are fine; low value, and ruff/coverage tooling handles `if`
   better. I would skip it.

8. **`isinstance(x, A | B)`** (3.10) instead of tuples. About 10 sites, e.g.
   `recursive/key.py:127`, `hashing/pydantic.py:159`, `nested/polars.py:46`. Ruff rule
   `UP038` used to enforce this, but it was **deprecated/removed** because `A | B` is
   slower than a tuple at runtime. Keep the tuples.

9. **`tomllib`** (3.11, stdlib). Not used in `src/` today. If any tooling in `tasks.py` or
   `dev/` parses `pyproject.toml` with a third-party package, replace it with `tomllib`.
   I did not find a `tomli` import in `src/`; check `tasks.py` and `dev/`.

10. **`enum.StrEnum`.** There are no `Enum` classes in `src/`; options are `Literal` strings.
    Keep `Literal`. If an enum is ever introduced, use `StrEnum` rather than
    `(str, Enum)`.

11. **`typing.LiteralString`.** Only useful if coola builds SQL-like strings. Not
    applicable.

12. **`typing.Required` / `NotRequired` for `TypedDict`.** No `TypedDict` use found.
    Not applicable today.

13. **`asyncio.TaskGroup` / `timeout`.** No asyncio in `src/`. Not applicable.

## 3. Tooling / config follow-ups

14. **Ruff `UP` is already enabled**, so most pyupgrade modernizations are already enforced.
    Consider `FURB` (refurb) and `PERF` to catch further idioms. Check that `UP` has no
    ignores hiding 3.11 items.
15. **CI matrix and classifiers.** Confirm the test matrix (`lib-get-test-matrix.yaml` input)
    and the `pyproject.toml` classifiers no longer list 3.10, and that mypy's
    `python_version` (if set) is 3.11.
16. **Docs wording.** `grep` found no remaining "3.9/3.10" mentions in `src` or `docs`
    during my pass, so I believe docs are consistent; re-grep README/CONTRIBUTING.

## Suggested order

1. Items 1 and 3 (mechanical, safe). Drop the B905 ignore with item 1.
2. Item 6 (`slots=True` on `EqualityConfig`) with a benchmark from `tests/benchmarks`.
3. Items 4, 5 as small, separate PRs because they touch behavior or typing strictness.
4. Revisit PEP 695 when the floor becomes 3.12.
