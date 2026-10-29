# Architecture of `coola.testing`

:book: This page describes the design of the `coola.testing` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/testing.md#properties); the API is in the [reference](../refs/testing.md).

## Constraints

1. **Simple and easy to maintain.** The package is 245 lines of source, 336 lines of tests (195
   tests) and 605 lines of user guide. Adding a dependency touches five places of one file
   (`__all__` twice, two annotations, one table row) and the developer guide describes a sixth way
   that no longer works. The target removes the code that exists only to work around the
   design (the 28 annotations, the hand-written `__all__`, the `plural` flag, the conftest
   re-export) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period. The two modules are renamed and the release notes list the
   changes. The mark names (`numpy_available`, ...) are **kept**: they are what the 93 test files
   of the repository, and every downstream suite, import, and they read well.
3. **No cost for the suites that do not use it.** Importing `coola.testing.marks` must not import an
   optional library (checking that `torch` is installed imports it: 1 s). This is the reason for
   the lazy marks and it must stay true by construction.
4. **Consistent with the library.** A test is skipped if and only if the library takes its fallback
   path for the same dependency. The marks and the library must use the same check.
5. **Pytest only here.** `pytest` is imported by `coola.testing` and by nothing else in `coola`
   (the import-linter contract already forbids the reverse direction); it is the `testing` extra.

## Architecture

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

### `coola.testing.marks`

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
    "torch_cuda": _Dependency(lambda: is_cuda_available(), "PyTorch with CUDA"),  # device marks are kept
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
  annotations are deleted: the return type of the module `__getattr__` types the names (checked
  with `ty` on a prototype). The `add_backend.md` step becomes "add one row to `_DEPENDENCIES`".
- **Lazy by construction.** The table stores the **function**, not its result. Nothing runs at
  import. The device probes are wrapped in a lambda that imports `coola.utils.tensor` on first
  call, which removes the two `noqa: PLC0415` helpers.
- **Singular names.** `plural` is deleted from `requires` and from the table. The reasons are
  "Requires PyTorch with CUDA" and "Skip if PyTorch with CUDA is available".
- **The error lists what exists.** An unknown name raises `AttributeError` whose message ends with
  the valid keys (`did you mean ...`), which is the substitute for the static check that a
  hand-written `__all__` gave.
- **`requires` keeps its position** `(is_available, name)` and stays public: it is the documented
  way for a downstream suite to define its own pair (`dask_available, dask_not_available =
requires(package_available("dask"), "Dask")`).
- `__dir__` returns `__all__` so that completion in an editor shows the marks.

### `coola.testing.fixtures`

```python
@pytest.fixture
def isolated_default_registries() -> Generator[None]:
    with temporary_default_registries():
        yield
```

The code does not change; the module name does and the way it is loaded changes
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

### The package docstring

`coola/testing/__init__.py` carries the **Properties** table (condensed) and the two lines "import
a mark from `coola.testing.marks`; load the fixtures with `pytest_plugins`", as `coola.recursive`
does. `__all__` stays empty.

## Public API

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

## Dependents

| Dependent                                                                    | Change                                                                                                                                                                                                                                  |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 93 test files importing `coola.testing.fixtures`                             | Mechanical: `from coola.testing.fixtures import` becomes `from coola.testing.marks import` (one `sed`; no name changes). The import-sorting of `ruff` is run after.                                                                     |
| `tests/integration/conftest.py`, `tests/unit/testing/conftest.py`            | Deleted, replaced by one `tests/conftest.py` with `pytest_plugins = ["coola.testing.fixtures"]`. (To verify in: `pytest_plugins` must be in the top-level conftest of the run; `testpaths = ["tests/"]` makes `tests/conftest.py` one.) |
| `tests/unit/summary/test_torch.py`, `tests/integration/random/test_torch.py` | Use `torch_cuda_available` / `torch_mps_available` instead of the three hand-written conditions. The `Mock()` fallback for `torch` in `test_torch.py` goes away if the module only uses `torch` inside tests guarded by a mark.         |
| `pyproject.toml`                                                             | The comment of the `testing` extra names the new modules. The import-linter contract (`coola.testing` forbidden for the library) is unchanged.                                                                                          |
| `tests/TESTING_GUIDE.md`                                                     | Its import line (`from coola.testing.fixtures import numpy_available`) is updated and a paragraph on the fixture is added.                                                                                                              |
| Downstream suites                                                            | Two import lines to change, listed in the migration page. No behaviour change for the marks.                                                                                                                                            |

## Decisions for the maintainers

All seven decisions are settled (the answer is in bold under each one) and the sections above
reflect them.

**D1. Rename the two modules?** _Recommended: yes._ `marks.py` for the marks and `fixtures.py` for
the fixture say what they contain; before the names are swapped, and the user guide has to
apologise for it. The cost is one `sed` over 93 files and two lines in the migration page, in a
major release. Alternative: keep `fixtures` (marks) and `registries` (fixture); no churn, and the
confusion stays for every new reader.

**Decision: yes.** Rename `fixtures.py` to `marks.py` and `registries.py` to `fixtures.py`.

**D2. How a suite gets the fixture.** _Recommended: `pytest_plugins = ["coola.testing.fixtures"]`
in the top-level `conftest.py`, documented._ It is the supported mechanism, needs no `__all__`
hack and no re-export in each directory. Alternatives: (a) keep the re-export in `conftest.py`
(works, boilerplate); (b) register `coola.testing.fixtures` through a `pytest11` entry point so it
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

**Decision: keep all three.** `torch_numpy_*` stays in the table (one row, singular name "PyTorch with NumPy"); the three hand-written CUDA/MPS conditions are replaced by the marks.

**D4. Keep the `*_not_available` marks?** _Recommended: yes._ They are the property "Paired marks",
they cost nothing in the table (built by the same call), and a fallback path that is never tested is
a fallback path that breaks. They are used by few files before, which says that the fallbacks are
under-tested rather than that the marks are useless. Alternative: generate only the `_available`
marks and document `requires` for the opposite; saves 14 names and weakens the guide.

**Decision: yes.** Keep the `*_not_available` marks.

**D5. Rename the marks to remove the clash with `coola.utils.imports`?** _Recommended: no,
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

## Not covered

- **Other fixtures.** A fixture that seeds the random generators, one that clears the availability
  caches, one that provides a temporary registry of each package: all are small and none has a user
  before. They belong in `coola.testing.fixtures` when a second test file needs one.
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
