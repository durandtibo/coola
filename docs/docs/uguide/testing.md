# Testing Utilities

:book: This page describes the `coola.testing` package, which helps to test code that has optional
dependencies: pytest **marks** that skip a test when a library is (or is not) installed, and a
pytest **fixture** that isolates the default registries.

**Prerequisites:** You'll need to know a bit of Python and pytest.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/) and
[pytest documentation](https://docs.pytest.org/). `pytest` is the `testing` extra:
`pip install coola[testing]`.

## Marks

Import a mark from `coola.testing.marks` and use it as a decorator. Each dependency has a pair:
`<name>_available` runs the test only if the dependency is installed, and `<name>_not_available`
runs it only if it is not (to test the fallback path).

| Dependency | Require it            | Skip if it is available   |
| ---------- | --------------------- | ------------------------- |
| colorlog   | `colorlog_available`  | `colorlog_not_available`  |
| JAX        | `jax_available`       | `jax_not_available`       |
| NumPy      | `numpy_available`     | `numpy_not_available`     |
| packaging  | `packaging_available` | `packaging_not_available` |
| pandas     | `pandas_available`    | `pandas_not_available`    |
| polars     | `polars_available`    | `polars_not_available`    |
| PyArrow    | `pyarrow_available`   | `pyarrow_not_available`   |
| pydantic   | `pydantic_available`  | `pydantic_not_available`  |
| rich       | `rich_available`      | `rich_not_available`      |
| PyTorch    | `torch_available`     | `torch_not_available`     |
| xarray     | `xarray_available`    | `xarray_not_available`    |

Three marks check a feature of PyTorch, not a package: `torch_cuda_*` (a usable CUDA device),
`torch_mps_*` (a usable MPS device) and `torch_numpy_*` (PyTorch and NumPy are both installed and
compatible). They are named like the others, e.g. `torch_cuda_available`.

```python
from coola.testing.marks import numpy_available, numpy_not_available, torch_cuda_available


@numpy_available
def test_with_numpy() -> None:
    import numpy as np

    assert np.array([1, 2, 3]).sum() == 6


@numpy_not_available
def test_fallback_without_numpy() -> None: ...


@torch_cuda_available
def test_on_gpu() -> None: ...
```

A mark is a standard `pytest.mark.skipif`, so it works on a function, on a class, on a module
(`pytestmark = numpy_available`) and in `pytest.param(..., marks=numpy_available)`. A skipped test
is reported with a reason that names the dependency (`Requires NumPy`,
`Skip if NumPy is available`).

### Marks for another dependency

`requires` creates the pair for any dependency. It takes a boolean that says whether the dependency
is available and a singular name for the skip reasons:

```python
from coola.testing.marks import requires
from coola.utils.imports import package_available

dask_available, dask_not_available = requires(package_available("dask"), "Dask")


@dask_available
def test_with_dask() -> None: ...
```

To support a new dependency in coola itself, add one row to `_DEPENDENCIES` in
`src/coola/testing/marks.py`.

## Isolating the Default Registries

The default registries (equality, hashing, summary, iterator, recursive, random) are process-wide: a
test that registers a type in one of them changes it for every other test. The
`isolated_default_registries` fixture of `coola.testing.fixtures` runs a test on copies of them and
restores the originals when the test ends, even if it fails. Load the module as a plugin in the
top-level `conftest.py` (no re-export is needed), then request the fixture:

```python
# tests/conftest.py
pytest_plugins = ["coola.testing.fixtures"]
```

```python
import pytest

from coola.equality.tester import DefaultEqualityTester, get_default_registry


class Point:
    pass


@pytest.mark.usefixtures("isolated_default_registries")
def test_point() -> None:
    get_default_registry().register(Point, DefaultEqualityTester())
    assert get_default_registry().has_equality_tester(Point)
```

The fixture is not autouse: request it from the tests that register something. Outside of pytest,
`coola.registry.default.temporary_default_registries()` is the context manager behind the fixture,
and `coola.registry.default.reset_default_registries()` forgets the default registries so the next
use builds them again with their default entries and plugins. Every registry has a `copy()` method
to build an independent registry with the same entries.

## Properties

| Property                | Statement                                                                                                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Paired marks            | Each dependency has a mark to run a test only if it is installed, and one only if it is not.                                                                                                                |
| Skip, not fail          | A test whose condition is not met is skipped with a message that names the dependency.                                                                                                                      |
| Standard pytest         | The marks are `pytest.mark.skipif` decorators, so they combine with the other pytest features.                                                                                                              |
| Lazy                    | Importing `coola.testing.marks` imports no optional library. The check of a dependency runs once, when its mark is first used, so a suite that uses only `numpy_available` never pays for `torch` or `jax`. |
| Agrees with the library | A mark calls the `is_<name>_available` function of `coola.utils.imports` that the code under test uses, so a test runs the fallback path exactly when the library does.                                     |
| Extensible              | `requires` creates the pair of marks for another dependency, like for the built-in ones.                                                                                                                    |
| Isolated registries     | The `isolated_default_registries` fixture runs a test on copies of the six default registries and restores the originals when the test ends, even if it fails.                                              |
| Opt-in                  | Nothing in `coola` outside `coola.testing` imports it, and the fixture is not autouse.                                                                                                                      |

Not guaranteed:

- **The fixture is not thread-safe.** The default registries are process-wide: do not request the
  fixture from tests that run in threads of the same process (`pytest-xdist` workers are processes
  and are fine).
- **Only the six default registries are isolated** (equality, hashing, iterator, random, recursive,
  summary). The random seeds, `sys.modules`, the environment and the availability caches are not.
- **A package that is installed but cannot be imported** is reported as not available, because
  that is what the library does.
- **A typo in a mark name** raises an `AttributeError` that lists the valid names when the test
  module is imported; the type checker does not catch it.

## See Also

- [pytest documentation](https://docs.pytest.org/): Learn more about pytest
- [`coola.utils.imports`](utils.md): For programmatically checking package availability
- [`coola.equality`](equality.md): For equality testing with multiple backends
