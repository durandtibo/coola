r"""Benchmarks comparing ``objects_are_equal`` and ``objects_are_allclose``
with the built-in comparison of each library.

Each benchmark runs on the same data with coola and with the built-in
function (``==``, ``numpy.array_equal``, ``torch.equal``,
``pandas.Series.equals``, ...), so the overhead of coola is the ratio between
the two times of a group. Run them with:

    inv test.benchmark

and compare the rows of the same group in the output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from coola.equality import objects_are_allclose, objects_are_equal
from coola.testing.marks import (
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    torch_available,
    xarray_available,
)
from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
    is_xarray_available,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_pandas_available():
    import pandas as pd
    import pandas.testing as pdt
else:  # pragma: no cover
    pd = Mock()
    pdt = Mock()

if is_polars_available():
    import polars as pl
    import polars.testing as plt
else:  # pragma: no cover
    pl = Mock()
    plt = Mock()

if is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    pa = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if is_xarray_available():
    import xarray as xr
else:  # pragma: no cover
    xr = Mock()

SIZE = 100_000
IMPLEMENTATIONS = ("coola", "builtin")


def passes(assert_fn: Callable[..., None]) -> Callable[..., Any]:
    r"""Wrap an ``assert_*`` function of a library into a function that
    returns a boolean, so it can be used as a built-in baseline."""

    def wrapper(actual: Any, expected: Any) -> bool:
        try:
            assert_fn(actual, expected)
        except AssertionError:
            return False
        return True

    return wrapper


def run(benchmark: BenchmarkFixture, group: str, fn: Callable[..., Any], *args: Any) -> None:
    r"""Benchmark ``fn(*args)`` in the group ``group`` and check that the
    objects are equal."""
    benchmark.group = group
    assert benchmark(fn, *args)


def equal_fn(impl: str, builtin: Callable[..., Any]) -> Callable[..., Any]:
    r"""Return ``objects_are_equal`` or the built-in function."""
    return objects_are_equal if impl == "coola" else builtin


def allclose_fn(impl: str, builtin: Callable[..., Any]) -> Callable[..., Any]:
    r"""Return ``objects_are_allclose`` or the built-in function."""
    return objects_are_allclose if impl == "coola" else builtin


def _values() -> Any:
    return np.random.default_rng(0).standard_normal(SIZE)


#########################
#     Python (==)       #
#########################


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_python_list_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data, other = list(range(SIZE)), list(range(SIZE))
    run(benchmark, "python list equal", equal_fn(impl, lambda a, b: a == b), data, other)


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_python_dict_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = {f"key{i}": float(i) for i in range(SIZE)}
    other = dict(data)
    run(benchmark, "python dict equal", equal_fn(impl, lambda a, b: a == b), data, other)


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_python_float_list_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = [float(i) for i in range(SIZE)]
    other = [float(i) for i in range(SIZE)]
    run(benchmark, "python list allclose", allclose_fn(impl, lambda a, b: a == b), data, other)


#########################
#     NumPy             #
#########################


@numpy_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_numpy_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = _values()
    run(benchmark, "numpy equal", equal_fn(impl, np.array_equal), data, data.copy())


@numpy_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_numpy_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = _values()
    run(benchmark, "numpy allclose", allclose_fn(impl, np.allclose), data, data.copy())


#########################
#     PyTorch           #
#########################


@torch_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_torch_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = torch.from_numpy(_values())
    run(benchmark, "torch equal", equal_fn(impl, torch.equal), data, data.clone())


@torch_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_torch_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = torch.from_numpy(_values())
    run(benchmark, "torch allclose", allclose_fn(impl, torch.allclose), data, data.clone())


#########################
#     JAX               #
#########################


@jax_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_jax_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = jnp.arange(SIZE, dtype=jnp.float32)
    run(
        benchmark,
        "jax equal",
        equal_fn(impl, lambda a, b: bool(jnp.array_equal(a, b))),
        data,
        data + 0,
    )


@jax_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_jax_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = jnp.arange(SIZE, dtype=jnp.float32)
    run(
        benchmark,
        "jax allclose",
        allclose_fn(impl, lambda a, b: bool(jnp.allclose(a, b))),
        data,
        data + 0,
    )


#########################
#     pandas            #
#########################


@numpy_available
@pandas_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_pandas_series_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pd.Series(_values())
    run(benchmark, "pandas series equal", equal_fn(impl, pd.Series.equals), data, data.copy())


@numpy_available
@pandas_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_pandas_series_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pd.Series(_values())
    fn = allclose_fn(impl, passes(lambda a, b: pdt.assert_series_equal(a, b, check_exact=False)))
    run(benchmark, "pandas series allclose", fn, data, data.copy())


@numpy_available
@pandas_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_pandas_frame_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pd.DataFrame(_values().reshape(-1, 10))
    run(benchmark, "pandas frame equal", equal_fn(impl, pd.DataFrame.equals), data, data.copy())


@numpy_available
@pandas_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_pandas_frame_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pd.DataFrame(_values().reshape(-1, 10))
    fn = allclose_fn(impl, passes(lambda a, b: pdt.assert_frame_equal(a, b, check_exact=False)))
    run(benchmark, "pandas frame allclose", fn, data, data.copy())


#########################
#     polars            #
#########################


@numpy_available
@polars_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_polars_series_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pl.Series(_values())
    run(benchmark, "polars series equal", equal_fn(impl, pl.Series.equals), data, data.clone())


@numpy_available
@polars_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_polars_series_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pl.Series(_values())
    fn = allclose_fn(impl, passes(lambda a, b: plt.assert_series_equal(a, b, check_exact=False)))
    run(benchmark, "polars series allclose", fn, data, data.clone())


@numpy_available
@polars_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_polars_frame_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pl.DataFrame(_values().reshape(-1, 10))
    run(benchmark, "polars frame equal", equal_fn(impl, pl.DataFrame.equals), data, data.clone())


@numpy_available
@polars_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_polars_frame_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pl.DataFrame(_values().reshape(-1, 10))
    fn = allclose_fn(impl, passes(lambda a, b: plt.assert_frame_equal(a, b, check_exact=False)))
    run(benchmark, "polars frame allclose", fn, data, data.clone())


#########################
#     PyArrow           #
#########################


@numpy_available
@pyarrow_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_pyarrow_array_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data, other = pa.array(_values()), pa.array(_values())
    run(benchmark, "pyarrow array equal", equal_fn(impl, pa.Array.equals), data, other)


@numpy_available
@pyarrow_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_pyarrow_table_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = pa.table({"a": _values(), "b": _values()})
    other = pa.table({"a": _values(), "b": _values()})
    run(benchmark, "pyarrow table equal", equal_fn(impl, pa.Table.equals), data, other)


#########################
#     xarray            #
#########################


@numpy_available
@xarray_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_xarray_data_array_equal_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = xr.DataArray(_values(), dims=["x"])
    run(benchmark, "xarray equal", equal_fn(impl, xr.DataArray.equals), data, data.copy())


@numpy_available
@xarray_available
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_xarray_data_array_allclose_benchmark(benchmark: BenchmarkFixture, impl: str) -> None:
    data = xr.DataArray(_values(), dims=["x"])
    fn = allclose_fn(impl, passes(xr.testing.assert_allclose))
    run(benchmark, "xarray allclose", fn, data, data.copy())
