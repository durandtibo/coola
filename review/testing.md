# `coola.testing`: target state and the changes to reach it

This document describes the design that `coola.testing` should have, then the changes to make to the
current implementation (`main`, `24913fab`) to get there. The design follows the rules of the user
guide (`docs/docs/uguide/testing.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: lazy imports, optional libraries, process-wide registries), and it
stays aligned with `coola.utils.imports` (the availability checks) and `coola.utils.singleton` (the
default registries), which it builds on. Its aim is a small package that is easy to maintain: **one
module for the marks, one module for the fixtures, one table row per optional dependency**. Breaking
changes are accepted, and the future state is more important than the past one.

`coola.testing` is the one sub-package that is not a registry: it has no engine and no extension
point of the `equality`/`hashing`/`recursive` kind. Its job is to let a test suite (coola's own and
the one of a downstream library) say "this test needs library X" and "this test changes the default
registries" in one line. The review therefore looks at the properties that matter for a test helper:
the cost of importing it, the number of places to touch to add a dependency, the agreement with the
code under test, and the cost of its own tests.

Evidence comes from probes run on `main` (Python 3.14, scripts kept outside the repository). Items
marked **[verified]** were reproduced; **[reading]** comes from the code. Nothing in `src`, `tests`
or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 245 lines of source, 336 lines of tests (195
   tests) and 605 lines of user guide. Adding a dependency touches five places of one file
   (`__all__` twice, two annotations, one table row) and the developer guide describes a sixth way
   that no longer works (F3). The target removes the code that exists only to work around the
   design (the 28 annotations, the hand-written `__all__`, the `plural` flag, the conftest
   re-export) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period. The two modules are renamed (decision D1) and the release notes list the
   changes. The mark names (`numpy_available`, ...) are **kept**: they are what the 93 test files
   of the repository, and every downstream suite, import, and they read well.
3. **No cost for the suites that do not use it.** Importing `coola.testing.marks` must not import an
   optional library (checking that `torch` is installed imports it: 1 s). This is the reason for
   the lazy marks and it must stay true by construction (section 1).
4. **Consistent with the library.** A test is skipped if and only if the library takes its fallback
   path for the same dependency. The marks and the library must use the same check.
5. **Pytest only here.** `pytest` is imported by `coola.testing` and by nothing else in `coola`
   (the import-linter contract already forbids the reverse direction); it is the `testing` extra.

## 1. Properties of the sub-package

These are the rules of the user guide (the table is rewritten in step 6; the current four rows are
kept and four are added). Every design choice below exists to make one of them true **by
construction** instead of by the discipline of the author of a test.

| Property                | Statement                                                                                                                                                                                                                                                           | Enforced by                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Paired marks            | Each dependency has `<name>_available` (run the test only if it is installed) and `<name>_not_available` (run it only if it is not), built together.                                                                                                                | the single table, `requires`                       |
| Skip, not fail          | A test whose condition is not met is **skipped** with a reason that names the dependency (`Requires NumPy`, `Skip if NumPy is available`). It is never collected as a pass and never silently dropped.                                                              | `requires`, `pytest.mark.skipif`                   |
| Standard pytest         | A mark is a `pytest.mark.skipif(...)` decorator: it works on a function, a class, a module (`pytestmark`), in `pytest.param(..., marks=...)`, and combines with any other mark. No plugin is needed.                                                                | `requires`                                         |
| Lazy                    | Importing `coola.testing.marks` imports no optional library and runs no check. The check of a dependency runs when its mark is first used, once, and a suite that uses only `numpy_available` never pays for `torch`, `jax` or the 1.2 s MPS probe.                 | module `__getattr__`                               |
| Agrees with the library | A mark calls the `is_<name>_available` function of `coola.utils.imports` (or the device probe of `coola.utils.tensor`), the function the code under test uses. A test runs the fallback path exactly when the library does.                                         | the table                                          |
| Extensible              | A downstream suite creates the pair for another dependency with `requires(is_available, name)`; coola adds a built-in one with one table row.                                                                                                                       | `requires`, the table                              |
| Isolated registries     | The fixture `isolated_default_registries` runs a test on copies of every default registry (equality, hashing, summary, iterator, recursive, random) and restores the originals when the test ends, **even if it fails**. Nothing a test registers leaks to another. | `temporary_default_registries`, `try`/`finally`    |
| Opt-in                  | Nothing in `coola` outside `coola.testing` imports it, the fixture is not autouse, and loading the plugin (`pytest_plugins`) changes no test that does not request the fixture.                                                                                     | the import-linter contract, the plugin has no hook |
| No state of its own     | The module has no global state besides the cache of the marks it built. It changes no environment variable, no `sys.modules` entry that is not a consequence of the check, no registry at import.                                                                   | review, the integration test of section 8          |

Not guaranteed, and documented as such:

- **The fixture is not thread-safe.** The default registries are process-wide and the context
  manager that backs the fixture swaps them: do not request it from tests that run in threads of
  the same process (`pytest-xdist` workers are processes and are fine).
- **Other process-wide state.** Only the six default registries built by
  `make_default_registry_singleton` are isolated. The random seeds, `sys.modules`, the environment
  and the availability caches are not (`clear_availability_caches` is the tool for the last one).
- **A broken install.** A package that is installed but cannot be imported is reported as not
  available, because that is what the library does (property "Agrees with the library").
- **Typos in a mark name** are detected when the test module is imported (`AttributeError` that
  lists the valid names), not by the type checker (F5).

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Status         |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The module names say the opposite of the content.** `coola.testing.fixtures` holds the **marks** (`numpy_available`, `requires`; there is no fixture in it) and `coola.testing.registries` holds the **fixture** (`isolated_default_registries`). The user guide has to say "all fixtures are defined in `coola.testing.fixtures`" and then "the `isolated_default_registries` fixture of `coola.testing.registries`".                                                                                                                                                   | **[reading]**  |
| F2  | **A fixture that must be re-exported.** A fixture imported in a test module is not found by the other modules, so each suite re-exports it from a `conftest.py` with an `__all__` that exists only to silence the unused-import lint (`tests/integration/conftest.py`, `tests/unit/testing/conftest.py`, and the same two lines in the user guide). Every downstream project copies this boilerplate; the supported mechanism for sharing a fixture is a plugin loaded with `pytest_plugins`.                                                                              | **[reading]**  |
| F3  | **Adding a dependency touches five places, and the developer guide is wrong.** In `fixtures.py` a new dependency needs its two names in `__all__`, its two annotations and one `_REQUIREMENTS` row. Step 4 of `docs/docs/dev/add_backend.md` still says to add `<pkg>_available, <pkg>_not_available = requires(is_<pkg>_available(), "<pkg>")`: that line calls the check at import and imports the library (the sum of the checks is 2.2 s: `torch` 0.98 s, `jax` 0.50 s, `pandas` 0.44 s).                                                                              | **[verified]** |
| F4  | **`plural` exists for a grammar problem.** `requires(..., plural=True)` and a third field in `_REQUIREMENTS` exist only so that the reason of `torch_cuda_not_available` reads "Skip if PyTorch and CUDA **are** available". A singular name ("PyTorch with CUDA") removes the flag, the third tuple element and the two documentation sentences.                                                                                                                                                                                                                          | **[reading]**  |
| F5  | **28 annotations duplicate the table.** `colorlog_available: pytest.MarkDecorator` ... exist so the type checker accepts `from coola.testing.fixtures import numpy_available`. A module-level `__getattr__ -> pytest.MarkDecorator` gives the same result: `ty` resolves the import and types the name through it (the price is that a typo is not caught statically). The 28 lines and the 28 names of `__all__` are a second copy of the table.                                                                                                                          | **[verified]** |
| F6  | **Unused and re-implemented marks.** No test outside `tests/unit/testing` uses `torch_cuda_*`, `torch_mps_*` or `torch_numpy_*` (six names, two probes and the `plural` flag), while three tests rebuild the same condition by hand: `tests/unit/summary/test_torch.py:14` (`cuda_available`, with a `Mock` fallback for `torch`), `tests/integration/random/test_torch.py:35` and `:48` (`skipif(not is_torch_available() or not torch.backends.mps.is_available())`). The hand-written versions evaluate `torch` at import of the test module, which the marks avoid.    | **[verified]** |
| F7  | **Two things with the same name.** `coola.utils.imports.numpy_available` is a **function decorator** that replaces the function by `None` when NumPy is missing (it is meant for library code); `coola.testing.fixtures.numpy_available` is a **pytest mark**. They are one import line apart. A test decorated with the first one is not skipped: it **disappears** from the run when the package is missing, with no skip report, which is the failure the second one exists to prevent. `tests/package_checks.py` imports the first kind.                               | **[reading]**  |
| F8  | **The tests of the marks are slow, repetitive and partly about the structure.** 195 tests in 1.5 s; the three slowest import `torch` (0.66 s), `jax` (0.28 s) and `pandas` (0.25 s) to call `getattr`. The same 14-name list is pasted five times; `test_all_exports_new_not_available_fixtures`, `test_requires_is_in_all` and `test_requires_is_callable` test `__all__` and `callable` (excluded by `CLAUDE.md`); `len(MARK_NAMES) == 28` fails each time a dependency is added; `test_all_names_are_markers` is the same test as `test_all_marks_are_mark_decorators`. | **[verified]** |
| F9  | **Nothing proves that a mark skips.** Every test inspects `mark.args[0]` and `mark.kwargs["reason"]`, the internals of `skipif`. No test runs a test under pytest and checks the outcome (skipped / passed), the one behaviour that matters. `CLAUDE.md` asks for an integration test for a behaviour that is only checked on a stand-in.                                                                                                                                                                                                                                  | **[reading]**  |
| F10 | **The laziness is a property that is not tested.** The module is lazy so that `import coola.testing.fixtures` does not import `torch` (**[verified]**: after the import `torch`, `jax` and `numpy` are not in `sys.modules`; the first access to `torch_mps_available` takes 1.2 s and imports `torch`). No test would fail if a future edit made the checks eager, and `tests/integration/test_lazy_imports.py` does not cover this module.                                                                                                                               | **[verified]** |
| F11 | **The user guide is long and repeats itself.** 605 lines: ten near-identical sections (`### NumPy Markers`, `#### numpy_available`, `#### numpy_not_available`, ...) that differ only by the library name, then "Usage Examples" and "Common Use Cases" that repeat them, and a "Design Principles" list that duplicates the **Properties** table. It does not say that the marks are lazy, that the fixture is not thread-safe, that only six registries are isolated, or how the two `numpy_available` differ (F7).                                                      | **[reading]**  |
| F12 | **The fixture docstring has no example that runs.** The doctest of `isolated_default_registries` only imports the fixture. The behaviour (the copy, the restore on error, the six registries) is in the docstring of `temporary_default_registries` and in a test file in `tests/integration/test_default_registries.py`; the fixture is tested by one unit test and two integration tests.                                                                                                                                                                                | **[reading]**  |
| F13 | **Two `noqa: PLC0415`** (`_is_cuda_available`, `_is_mps_available`) exist because `coola.utils.tensor` imports `torch` at module level when it is installed, so importing it eagerly would defeat the laziness. They disappear because the table stores the probe as a lazy callable that imports the function on first call (section 3.1); the device marks are kept (decision D3).                                                                                                                                                                                       | **[reading]**  |

What is already right and must not regress: the lazy creation of the marks through a module
`__getattr__` (PEP 562) with a cache; the pair built by one call (`requires`); the reasons that name
the dependency; the fixture built on `temporary_default_registries`, which restores on error; the
`testing` extra that pins `pytest`; the import-linter contract that forbids the library to import
`coola.testing`; the rule that `coola.testing.__all__` is empty (the user imports from the module
that holds the name).

## 3. Architecture

```
coola/testing/
    __init__.py     package docstring (properties), __all__ = []
    marks.py        requires(), the table of dependencies, the lazy marks     (about 90 lines)
    fixtures.py     isolated_default_registries                               (about 40 lines)
```

```
test module                      coola.testing.marks                          coola.utils.imports
-----------                      -------------------                          -------------------
from ...marks import             module __getattr__("numpy_available")
    numpy_available   ------->     look up "numpy" in _DEPENDENCIES
                                   requires(is_numpy_available(), "NumPy") ->  importlib check (once, cached)
                                   cache both marks in the module globals
@numpy_available       <-------   pytest.mark.skipif(not available, reason=...)
```

```
conftest.py (top of the suite)           coola.testing.fixtures
------------------------------           ----------------------
pytest_plugins = ["coola.testing.fixtures"]  ->  @pytest.fixture isolated_default_registries
                                                    with temporary_default_registries(): yield
```

### 3.1 `coola.testing.marks`

```python
__all__ = ["requires", *sorted(...)]  # built from the table, see below


class _Dependency(NamedTuple):
    is_available: Callable[[], bool]  # the check of the library, called on first use
    name: str  # singular, used in the reasons


def requires(is_available: bool, name: str) -> tuple[pytest.MarkDecorator, pytest.MarkDecorator]:
    """The (available, not_available) marks for one dependency."""
    return (
        pytest.mark.skipif(not is_available, reason=f"Requires {name}"),
        pytest.mark.skipif(is_available, reason=f"Skip if {name} is available"),
    )


# One row per dependency: the only place to edit to support a new one.
_DEPENDENCIES: dict[str, _Dependency] = {
    "colorlog": _Dependency(is_colorlog_available, "colorlog"),
    "jax": _Dependency(is_jax_available, "JAX"),
    ...
    "torch_cuda": _Dependency(lambda: is_cuda_available(), "PyTorch with CUDA"),  # device marks are kept (D3)
    "torch_mps": _Dependency(lambda: is_mps_available(), "PyTorch with MPS"),
    "torch_numpy": _Dependency(is_torch_numpy_available, "PyTorch with NumPy"),
}


def __getattr__(name: str) -> pytest.MarkDecorator:
    for suffix in ("_available", "_not_available"):
        key = name.removesuffix(suffix)
        if name.endswith(suffix) and key in _DEPENDENCIES:
            dependency = _DEPENDENCIES[key]
            available, not_available = requires(dependency.is_available(), dependency.name)
            globals()[f"{key}_available"] = available  # cached: __getattr__ is not called again
            globals()[f"{key}_not_available"] = not_available
            return globals()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
```

Design points:

- **One source of truth.** `_DEPENDENCIES` is the table. `__all__` is computed from it
  (`["requires", *(f"{k}{s}" for k in _DEPENDENCIES for s in ("_available", "_not_available"))]`),
  so `from coola.testing.marks import *` still works and there is no second list. The 28
  annotations are deleted: the return type of the module `__getattr__` types the names (F5, checked
  with `ty` on a prototype). The `add_backend.md` step becomes "add one row to `_DEPENDENCIES`".
- **Lazy by construction.** The table stores the **function**, not its result. Nothing runs at
  import. The device probes are wrapped in a lambda that imports `coola.utils.tensor` on first
  call, which removes the two `noqa: PLC0415` helpers (F13).
- **Singular names.** `plural` is deleted from `requires` and from the table (F4). The reasons are
  "Requires PyTorch with CUDA" and "Skip if PyTorch with CUDA is available".
- **The error lists what exists.** An unknown name raises `AttributeError` whose message ends with
  the valid keys (`did you mean ...`), which is the substitute for the static check that a
  hand-written `__all__` gave (F5).
- **`requires` keeps its position** `(is_available, name)` and stays public: it is the documented
  way for a downstream suite to define its own pair (`dask_available, dask_not_available =
requires(package_available("dask"), "Dask")`).
- `__dir__` returns `__all__` so that completion in an editor shows the marks.

### 3.2 `coola.testing.fixtures`

```python
@pytest.fixture
def isolated_default_registries() -> Generator[None]:
    with temporary_default_registries():
        yield
```

The code does not change; the module name does (decision D1) and the way it is loaded changes
(decision D2): a suite loads it with one line in its top-level `conftest.py`,

```python
# tests/conftest.py
pytest_plugins = ["coola.testing.fixtures"]
```

and requests it by name (`@pytest.mark.usefixtures("isolated_default_registries")`, or as an
argument). No `__all__` hack, no re-export in each sub-directory `conftest.py`, and the fixture is
visible in `pytest --fixtures` with its docstring. The docstring gets a runnable example of the
behaviour (a `pycon` block cannot run a pytest fixture, so the example shows the test and the
restore; the doctest of `temporary_default_registries` covers the runnable part).

### 3.3 The package docstring

`coola/testing/__init__.py` carries the **Properties** table (condensed) and the two lines "import
a mark from `coola.testing.marks`; load the fixtures with `pytest_plugins`", as `coola.recursive`
does. `__all__` stays empty.

## 4. Public API after the change

```python
from coola.testing.marks import (
    requires,  # (is_available: bool, name: str) -> (available, not_available)
    numpy_available,
    numpy_not_available,  # one pair per dependency of the table
    # colorlog, jax, numpy, packaging, pandas, polars, pyarrow, pydantic, rich, torch,
    # torch_cuda, torch_mps, torch_numpy, xarray
)

# in tests/conftest.py
pytest_plugins = ["coola.testing.fixtures"]  # provides: isolated_default_registries
```

Removed or changed: `coola.testing.fixtures` (the marks) is now `coola.testing.marks`;
`coola.testing.registries` (the fixture) is now `coola.testing.fixtures`; `requires(..., plural=)`
is removed; the re-export of the fixture in a `conftest.py` is replaced by `pytest_plugins`.

## 5. Dependents

| Dependent                                                                    | Change                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 93 test files importing `coola.testing.fixtures`                             | Mechanical: `from coola.testing.fixtures import` becomes `from coola.testing.marks import` (one `sed`; no name changes). The import-sorting of `ruff` is run after.                                                                                                                      |
| `tests/integration/conftest.py`, `tests/unit/testing/conftest.py`            | Deleted, replaced by one `tests/conftest.py` with `pytest_plugins = ["coola.testing.fixtures"]`. (To verify in step 3: `pytest_plugins` must be in the top-level conftest of the run; `testpaths = ["tests/"]` makes `tests/conftest.py` one.)                                           |
| `tests/unit/summary/test_torch.py`, `tests/integration/random/test_torch.py` | Use `torch_cuda_available` / `torch_mps_available` instead of the three hand-written conditions (F6). The `Mock()` fallback for `torch` in `test_torch.py` goes away if the module only uses `torch` inside tests guarded by a mark.                                                     |
| `docs/docs/dev/add_backend.md`, `docs/docs/dev/architecture.md`              | Step 4 of `add_backend.md` becomes "add a row to `_DEPENDENCIES` in `src/coola/testing/marks.py`" (and the table row of the summary at the end of the page). `architecture.md` mentions `testing/` in the package tree and in the import-linter paragraph: the module names are updated. |
| `docs/docs/refs/testing.md`, `docs/mkdocs.yml`                               | `::: coola.testing.marks` and `::: coola.testing.fixtures` replace the two current directives.                                                                                                                                                                                           |
| `pyproject.toml`                                                             | The comment of the `testing` extra names the new modules. The import-linter contract (`coola.testing` forbidden for the library) is unchanged.                                                                                                                                           |
| `tests/TESTING_GUIDE.md`                                                     | Its import line (`from coola.testing.fixtures import numpy_available`) is updated and a paragraph on the fixture is added.                                                                                                                                                               |
| Downstream suites                                                            | Two import lines to change, listed in the migration page. No behaviour change for the marks.                                                                                                                                                                                             |

## 6. Evidence

All on `main`, Python 3.14, Apple M1. **[verified]**

| Probe                                                                                                | Result                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `import coola.testing.fixtures`, then `"torch" in sys.modules`                                       | 149 ms (pytest and coola), `torch`, `jax` and `numpy` **not** imported: the laziness works today.                                                                     |
| First access to `numpy_available`                                                                    | 71 ms (imports NumPy). `torch_mps_available`: **1.2 s** (imports `torch`, allocates a tensor on `mps`).                                                               |
| Eager alternative: call the 11 `is_*_available` checks at import                                     | `torch` 981 ms, `jax` 499 ms, `pandas` 444 ms, `polars` 134 ms, `xarray` 90 ms, `numpy` 65 ms, `pydantic` 28 ms: about 2.3 s.                                         |
| `ty check` on a module with `def __getattr__(name) -> pytest.MarkDecorator` and a computed `__all__` | `from m import numpy_available` resolves, no diagnostic. The 28 annotations are not needed.                                                                           |
| `pytest tests/unit/testing --cov=coola.testing`                                                      | 195 passed in 1.5 s, 100 % of the three modules. The slow tests are the `getattr` of `torch`, `jax`, `pandas`.                                                        |
| Users of the marks outside `tests/unit/testing`                                                      | 93 files import `coola.testing.fixtures`; `*_not_available` is used by at most two files per dependency; `torch_cuda_*`, `torch_mps_*`, `torch_numpy_*` by none (F6). |
| Registries covered by the fixture                                                                    | the six created with `make_default_registry_singleton`: equality, hashing, iterator, random, recursive, summary.                                                      |

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                              | Breaking           |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| 1    | **Fix the guides (F3, F11 first part).** `add_backend.md` step 4 describes the lazy table instead of the eager `requires(...)` line; the user guide gains the **Properties** of section 1 (lazy, agrees with the library, not thread-safe, six registries) and the note on the two `numpy_available` (F7). Documentation only.                                                                      | no                 |
| 2    | **Simplify the marks (F3, F4, F5, F13).** One table, computed `__all__` and `__dir__`, the module `__getattr__` with the typed return, no annotations, no `plural`, singular names, no `PLC0415` helpers, an `AttributeError` that lists the valid names. Behaviour of every existing mark is unchanged (same skip conditions, reasons for the two-word names change). Tests rewritten (section 8). | `plural=` removed  |
| 3    | **Rename the modules and load the fixture as a plugin (F1, F2).** `fixtures.py` becomes `marks.py`, `registries.py` becomes `fixtures.py`, the 93 imports are updated, the two `conftest.py` files become one `tests/conftest.py` with `pytest_plugins`, the docs directives and the status table are updated. No logic change: a pure move, reviewable with `git diff -M`.                         | yes (import paths) |
| 4    | **Use the marks everywhere (F6).** Replace the three hand-written CUDA/MPS conditions by the marks; remove the `Mock` fallback if it becomes unused. All three device marks (`torch_cuda_*`, `torch_mps_*`, `torch_numpy_*`) are kept (decision D3).                                                                                                                                                | no                 |
| 5    | **Tests (F8, F9, F10, F12).** Replace the structural tests by the table-driven ones of section 8; add the `pytester` tests that run a test and check the outcome, the subprocess test of the laziness, and the integration test of the fixture through a real pytest run.                                                                                                                           | no                 |
| 6    | **Documentation (F11).** Rewrite `uguide/testing.md` (about 150 lines: the table of marks, three short examples, the extension with `requires`, the fixture and `pytest_plugins`, the Properties), the package docstring, `refs/testing.md`, the migration page (`migration/v0.11_to_v1.0.md`: the two renamed modules, `plural`), and the status table of `review/suggestions.md`.                 | no                 |

Steps 2 and 3 can be one pull request if the reviewer prefers a single breaking change; they are
separate here because step 3 is a pure move and step 2 is a rewrite, and mixing them hides the
rewrite in a 100-file diff.

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/testing/test_marks.py`,
`tests/unit/testing/test_fixtures.py`); the conventions of `CLAUDE.md` apply (one behaviour per
test, literal expected values, a regression test for every bug, no local imports, no sleeps).

- **`requires`** (`test_marks.py`): the available mark does not skip when the dependency is
  available and skips when it is not; the not-available mark is the opposite; the reasons are the
  two literals (`"Requires NumPy"`, `"Skip if NumPy is available"`); both marks are `skipif`.
  (Kept: these are fast and about the function.)
- **The table**: **one** parametrised test per property over `_DEPENDENCIES` (not five pasted
  lists): every key gives a pair of marks; the two marks are opposite (`args[0]` differs); the
  reasons start with `Requires ` / `Skip if ` and the second ends with ` is available`. To stay
  fast the test patches the check of the key (`monkeypatch.setitem(_DEPENDENCIES, key,
_Dependency(lambda: True, "X"))`) instead of importing `torch`, `jax` and `pandas`; the real
  checks are covered by the integration test below.
- **The lazy `__getattr__`**: an unknown name raises `AttributeError` that names the module and
  the attribute (kept); the message lists the valid names (new); the marks are cached and created
  together (kept, reduced to two tests); `__all__` equals `["requires"]` plus the two names of every
  row, asserted against a fully-specified literal built from a **two-row table** patched into the
  module (no assertion on `len == 28`, which fails each time a dependency is added); `dir()` contains
  the marks.
- **Removed**: `test_all_exports_new_not_available_fixtures`, `test_requires_is_in_all`,
  `test_requires_is_callable` (tests of `__all__` and `callable`), `test_all_names_are_markers`
  (duplicate), `len(MARK_NAMES) == 28`, the three `*_is_opposite_of_*` tests (covered by the
  parametrised one).
- **The marks skip, for real** (`tests/integration/testing/test_marks.py`, new, with `pytester`):
  a generated test file with one test decorated by a mark of an available dependency and one by a
  not-available dependency (`requires(True, "X")`, `requires(False, "X")` imported in the generated
  file) is run, and `result.assert_outcomes(passed=1, skipped=1)`; the skip reason is in the report
  (`-rs`). Run on `pytestmark` of a module, on a class and on `pytest.param(..., marks=...)` (one
  test each). Covers F9.
- **Lazy import** (same folder, subprocess like `tests/integration/utils/test_torch_numpy.py`): a
  fresh interpreter imports `coola.testing.marks` and prints whether `torch`, `jax`, `numpy` and
  `pandas` are in `sys.modules` (expected: none); a second one accesses `numpy_available` and
  checks that `torch` is still absent. Covers F10 and is the guard of the **Lazy** property.
- **Agrees with the library** (one parametrised test over the table, in the integration folder):
  for every key whose library is installed, the skip condition of `<key>_available` equals
  `not is_<key>_available()`. Cheap because the checks are cached by the time the suite ran.
- **The fixture** (`tests/unit/testing/test_fixtures.py` and `tests/integration/`): keep the unit
  test (registers a type inside) and the integration tests (does not leak to the next test, restores
  a registry that was cleared, restores the hasher registry). Add: the original registry is
  restored when the test fails (`pytester`: a generated test registers a type then raises; a second
  test of the same run asserts the type is gone); a registry of each of the six packages is covered
  (one parametrised test over the six `get_default_registry` functions); the fixture is found
  through `pytest_plugins` with no re-export (the generated `conftest.py` of the `pytester` run
  contains only that line).
- **Doctests** run as part of the docstrings; the `requires` example is kept and the guide examples
  that do not need a pytest run are doctests.
- Coverage: 100 % of `src/coola/testing` (`--cov=coola`), no `# pragma: no cover`.
- **Gate on the cost.** `pytest tests/unit/testing` must not import `torch`, `jax` or `pandas`
  (check with `python -X importtime` or the subprocess test above) and must run in well under the
  1.5 s of today (target: under 0.5 s).

## 9. Decisions for the maintainers

All seven decisions are settled (the answer is in bold under each one) and the sections above
reflect them.

**D1. Rename the two modules?** _Recommended: yes._ `marks.py` for the marks and `fixtures.py` for
the fixture say what they contain; today the names are swapped (F1), and the user guide has to
apologise for it. The cost is one `sed` over 93 files and two lines in the migration page, in a
major release. Alternative: keep `fixtures` (marks) and `registries` (fixture); no churn, and the
confusion stays for every new reader.

**Decision: yes.** Rename `fixtures.py` to `marks.py` and `registries.py` to `fixtures.py` (step 3).

**D2. How a suite gets the fixture.** _Recommended: `pytest_plugins = ["coola.testing.fixtures"]`
in the top-level `conftest.py`, documented._ It is the supported mechanism, needs no `__all__`
hack and no re-export in each directory. Alternatives: (a) keep the re-export in `conftest.py`
(works, F2 boilerplate); (b) register `coola.testing.fixtures` through a `pytest11` entry point so it
is loaded in every session where `coola` and `pytest` are installed. (b) is the least typing but
**intrusive**: `coola` is a dependency of ML code, and a library should not add a plugin and a
fixture name to the sessions of projects that did not ask for it. Not recommended. The fixture is
**not** autouse in any case: isolation costs a copy of six registries per test and most tests do not
register anything.

**Decision: follow the recommendation.** `pytest_plugins` in the top-level `conftest.py`; no entry point, not autouse.

**D3. Keep the device marks (`torch_cuda_*`, `torch_mps_*`, `torch_numpy_*`)?** Three options were
considered: keep all three, keep `torch_cuda_*` and `torch_mps_*` and drop `torch_numpy_*` (no user
in the repository; a downstream suite could write `requires(is_torch_numpy_available(), "PyTorch
and NumPy")`), or drop all of them (the MPS probe, 1.2 s, would then be paid at the import of each
test module that rebuilds the condition by hand).

**Decision: keep all three.** `torch_numpy_*` stays in the table (one row, singular name "PyTorch with NumPy"); the three hand-written CUDA/MPS conditions are replaced by the marks (step 4).

**D4. Keep the `*_not_available` marks?** _Recommended: yes._ They are the property "Paired marks",
they cost nothing in the table (built by the same call), and a fallback path that is never tested is
a fallback path that breaks. They are used by few files today, which says that the fallbacks are
under-tested rather than that the marks are useless. Alternative: generate only the `_available`
marks and document `requires` for the opposite; saves 14 names and weakens the guide.

**Decision: yes.** Keep the `*_not_available` marks.

**D5. Rename the marks to remove the clash with `coola.utils.imports` (F7)?** _Recommended: no,
document and lint instead._ `numpy_available` reads well on a test, and the module that imports it
(`coola.testing.marks`) is what differs. A silent disappearance of a test is the risk, so add a
`flake8-tidy-imports` `banned-api` entry in `pyproject.toml` that forbids
`coola.utils.imports.<pkg>_available` in `tests/` (a per-file ignore for `src/`), and say it in the
guide. Alternative: rename the marks `requires_numpy` / `without_numpy`, which makes the clash
impossible but changes 93 files by hand-written names instead of a path.

**Decision: no.** Keep the names; the clash is handled by the guide and the `banned-api` lint rule, as recommended.

**D6. Catch the missing `pytest` with a friendly error?** _Recommended: no._ `import pytest` at the
top of the module raises `ModuleNotFoundError: No module named 'pytest'`, which a test author
understands at once; the package docstring and the guide name the `testing` extra. A wrapper that
catches and re-raises is code for a message nobody needs.

**Decision: no.** No friendly error for a missing `pytest`.

**D7. A mark that evaluates at run time (`@pytest.mark.requires("numpy")` with a plugin hook)?**
_Recommended: no._ It would be the shortest to extend (one dict, no `__getattr__`) and it moves the
check to the run phase, but it needs a plugin in every suite (the marks work with no plugin today),
loses the importable and greppable names, and changes 93 files in a way that is not mechanical for
`pytestmark = [a, b]` and `pytest.param(marks=...)`. The module `__getattr__` costs 10 lines and
keeps the property "Standard pytest".

**Decision: no.** Keep the importable marks and the module `__getattr__`.

## 10. Not covered

- **Other fixtures.** A fixture that seeds the random generators, one that clears the availability
  caches, one that provides a temporary registry of each package: all are small and none has a user
  today. They belong in `coola.testing.fixtures` when a second test file needs one.
- **Parametrisation helpers** (a list of `(numpy_array, torch_tensor)` pairs, `NATIVE_SAMPLES`):
  they are test data of `equality`/`hashing` and live next to those tests (`tests/helpers.py`).
- **Thread-safe isolation.** Making `temporary_default_registries` safe for several threads would
  need a lock or a context variable in every default registry; the fixture is for the main thread
  of a test, and the thread-safety tests of the registries use their own registry
  (`tests/integration/helpers.py`).
- **Python versions and platforms.** The marks do not know about the Python version or the OS
  (`skipif(sys.platform == ...)` is one line in the test and not worth a mark).
- **Strict mode.** A way to make the suite **fail** when a dependency is missing in CI (so a broken
  environment cannot turn the whole numpy suite into skips): an environment variable read by
  `requires` that turns `skipif` into a failure is attractive, and is a separate item with its own
  design (CI configuration, naming of the variable).

## Appendix: effect on the size of the package

| Module / file                           | Lines today | Expected after | Why                                                                                                   |
| --------------------------------------- | ----------- | -------------- | ----------------------------------------------------------------------------------------------------- |
| `fixtures.py` (becomes `marks.py`)      | 191         | about 90       | no annotations, no hand-written `__all__`, no `plural`, no `PLC0415` helpers; the table is the source |
| `registries.py` (becomes `fixtures.py`) | 39          | about 40       | same fixture, renamed, longer docstring                                                               |
| `__init__.py`                           | 15          | about 30       | the Properties table in the package docstring                                                         |
| `tests/unit/testing/` (source tests)    | 336         | about 160      | one parametrised test per property, no tests of `__all__`, no pasted lists                            |
| `tests/integration/testing/` (new)      | 0           | about 120      | `pytester` outcome tests, subprocess laziness test, agreement with the library, fixture on failure    |
| `tests/*/conftest.py` re-exports        | 10          | 4              | one `pytest_plugins` line                                                                             |
| `docs/docs/uguide/testing.md`           | 605         | about 150      | a table instead of ten sections, three short examples                                                 |

The line counts are estimates from the code to delete and the structure above; the gates are the
behaviour (sections 1 and 8), not the size.
