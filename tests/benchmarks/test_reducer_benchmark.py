r"""Benchmarks for the reducers.

Run them explicitly with:

    inv test.benchmark

They document the numbers that motivate ``auto_reducer`` (NumPy before
PyTorch) and ``NativeReducer.mean``: the cost of the statistics on a
list of 100,000 numbers, and of a call on a short list for each backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.reducer import BaseReducer, NativeReducer, NumpyReducer, TorchReducer, auto_reducer
from coola.testing.marks import numpy_available, torch_available

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

LARGE = list(range(100_000))
SHORT = [1, 2, 3]


@pytest.fixture(
    params=[
        pytest.param(NativeReducer, id="native"),
        pytest.param(NumpyReducer, id="numpy", marks=numpy_available),
        pytest.param(TorchReducer, id="torch", marks=torch_available),
        pytest.param(auto_reducer, id="auto"),
    ]
)
def reducer(request: pytest.FixtureRequest) -> BaseReducer:
    return request.param()


@pytest.mark.parametrize("method", ["mean", "median", "max"])
def test_reducer_large_list(benchmark: BenchmarkFixture, reducer: BaseReducer, method: str) -> None:
    benchmark(getattr(reducer, method), LARGE)


def test_reducer_sort_large_list(benchmark: BenchmarkFixture, reducer: BaseReducer) -> None:
    benchmark(reducer.sort, LARGE)


def test_reducer_mean_short_list(benchmark: BenchmarkFixture, reducer: BaseReducer) -> None:
    benchmark(reducer.mean, SHORT)
