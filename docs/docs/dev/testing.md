# Architecture of `coola.testing`

:book: This page describes the design of the `coola.testing` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/testing.md#properties); the API is in the [reference](../refs/testing.md).

## Constraints

1. **Simple and easy to maintain.** Adding a dependency is one row of one table in
   `coola.testing.marks`. There is no hand-written `__all__`, no annotation per mark and no re-export
   of the fixture in each `conftest.py`.
2. **Stable mark names.** The mark names (`numpy_available`, ...) are what every test file of the
   repository, and every downstream suite, import, and they read well.
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
    marks.py        requires(), the table of dependencies, the lazy marks
    fixtures.py     isolated_default_registries
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
    "torch_cuda": _Dependency(_is_cuda_available, "PyTorch with CUDA"),
    "torch_mps": _Dependency(_is_mps_available, "PyTorch with MPS"),
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
  (`["requires", *(f"{k}{s}" for k in _DEPENDENCIES for s in ("_available", "_not_available"))]`), so
  `from coola.testing.marks import *` works and there is no second list. The return type of the module
  `__getattr__` types the names, so there is no annotation per mark. Supporting a new dependency is
  "add one row to `_DEPENDENCIES`" (see [add_backend.md](add_backend.md)).
- **Lazy by construction.** The table stores the **function**, not its result. Nothing runs at import.
  The device probes are small functions that import `coola.utils.tensor` on first call.
- **Singular names.** The reasons are "Requires PyTorch with CUDA" and "Skip if PyTorch with CUDA is
  available".
- **The error lists what exists.** An unknown name raises `AttributeError` whose message ends with the
  valid keys (`did you mean ...`), the substitute for the static check that a hand-written `__all__`
  would give.
- **`requires(is_available, name)` is public.** It is the documented way for a downstream suite to
  define its own pair (`dask_available, dask_not_available = requires(package_available("dask"),
"Dask")`).
- `__dir__` returns `__all__` so that completion in an editor shows the marks.
- **The device marks** (`torch_cuda_*`, `torch_mps_*`, `torch_numpy_*`) are in the table, so a test
  does not rebuild the condition by hand. Without them, the MPS probe (1.2 s) would be paid at the
  import of each test module that writes its own.
- **The `*_not_available` marks** exist for every dependency: a fallback path that is never tested
  breaks. They cost nothing in the table (built by the same call).
- **The marks are not renamed to remove the clash with `coola.utils.imports`.** `numpy_available`
  reads well on a test, and the module that imports it (`coola.testing.marks`) is what differs. The
  risk is a silent disappearance of a test that imports the function `is_numpy_available` instead of
  the mark; the guide says so. Renaming the marks `requires_numpy` / `without_numpy` would make the
  clash impossible at the cost of every test file.
- **Import-time marks, not a plugin.** A mark that evaluates at run time (`@pytest.mark.requires("numpy")`
  with a plugin hook) would be the shortest to extend, but it needs a plugin in every suite (the marks
  work with no plugin), loses the importable and greppable names, and is not mechanical for
  `pytestmark = [a, b]` and `pytest.param(marks=...)`. The module `__getattr__` costs 10 lines and keeps
  the property "Standard pytest".
- **No friendly error for a missing `pytest`.** `import pytest` raises `ModuleNotFoundError: No module
named 'pytest'`, which a test author understands at once; the package docstring and the guide name the
  `testing` extra.

### `coola.testing.fixtures`

```python
@pytest.fixture
def isolated_default_registries() -> Generator[None]:
    with temporary_default_registries():
        yield
```

A suite loads it with one line in its top-level `conftest.py`,

```python
# tests/conftest.py
pytest_plugins = ["coola.testing.fixtures"]
```

and requests it by name (`@pytest.mark.usefixtures("isolated_default_registries")`, or as an
argument). There is no `__all__` hack and no re-export in each sub-directory `conftest.py`, and the
fixture is visible in `pytest --fixtures` with its docstring. `pytest_plugins` must be in the top-level
`conftest.py` of the run (`testpaths = ["tests/"]` makes `tests/conftest.py` one).

The fixture is **not** autouse: isolation costs a copy of the default registries per test and most
tests do not register anything. It is also **not** registered through a `pytest11` entry point, which
would be the least typing but intrusive: `coola` is a dependency of ML code, and a library should not
add a plugin and a fixture name to the sessions of projects that did not ask for it.

### The package docstring

`coola/testing/__init__.py` carries the **Properties** table (condensed) and the two lines "import a
mark from `coola.testing.marks`; load the fixtures with `pytest_plugins`", as `coola.recursive` does.
`__all__` is empty.

## Public API

```python
from coola.testing.marks import (
    requires,  # (is_available: bool, name: str) -> (available, not_available)
    numpy_available,
    numpy_not_available,  # one pair per dependency of the table
    # colorlog, jax, numpy, packaging, pandas, polars, pyarrow, pydantic, rich, scipy, torch,
    # torch_cuda, torch_mps, torch_numpy, xarray
)

# in tests/conftest.py
pytest_plugins = ["coola.testing.fixtures"]  # provides: isolated_default_registries
```

The module `marks` holds the marks and the module `fixtures` holds the fixture, so each name says what
it contains.

## Dependents

| Dependent                                                                    | Use                                                                                                                                                                |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The test files of the repository                                             | `from coola.testing.marks import numpy_available, ...`                                                                                                             |
| `tests/conftest.py`                                                          | `pytest_plugins = ["coola.testing.fixtures"]`.                                                                                                                     |
| `tests/unit/summary/test_torch.py`, `tests/integration/random/test_torch.py` | `torch_cuda_available` / `torch_mps_available` instead of hand-written conditions.                                                                                 |
| `pyproject.toml`                                                             | The comment of the `testing` extra names the modules. The import-linter contract (`coola.testing` is forbidden for the library) keeps `pytest` out of the library. |
| `tests/TESTING_GUIDE.md`                                                     | Documents the marks and the fixture.                                                                                                                               |
| Downstream suites                                                            | Import the marks from `coola.testing.marks` and load the fixtures with `pytest_plugins`.                                                                           |

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
