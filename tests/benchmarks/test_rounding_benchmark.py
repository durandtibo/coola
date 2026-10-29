r"""Benchmarks for ``round_values``.

Run them explicitly with:

    inv test.benchmark

They exist to compare the performance of the implementations of
``coola.rounding`` over time (``review/rounding.md``, section 8). They
only use ``round_values(data, mode=..., decimals=...)`` with the default
registry, so that they run unchanged on any design of the package. Do
not use the internals of the package (rounders, registry methods) in
this file.

The gate against ``main`` is: ``floor`` and ``ceil`` at most 1.5 times
slower, ``round`` at most 5 times slower, and a Python float and the
recursive path not slower. A ratio that the prototype of the review
already explains does not reject a change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.rounding import round_values
from coola.testing.marks import (
    numpy_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    torch_available,
)
from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_pyarrow_available():
    import pyarrow as pa
if is_torch_available():
    import torch

SIZE = 1_000_000
NESTED_SIZE = 100_000
MODES = ["round", "floor", "ceil"]


def make_values(size: int) -> list[float]:
    r"""Create ``size`` floats with a few decimals, in [-1000, 1000]."""
    return [(i * 7919 % 2_000_001 - 1_000_000) / 1000 + 0.0005 for i in range(size)]


###########################
#     Python float        #
###########################


@pytest.mark.parametrize("mode", MODES)
def test_round_values_float_benchmark(benchmark: BenchmarkFixture, mode: str) -> None:
    assert benchmark(round_values, 2.675, mode=mode, decimals=2) in {2.67, 2.68}


def test_round_values_int_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(round_values, 3, decimals=2) == 3


###########################
#     Arrays and tables   #
###########################


@numpy_available
@pytest.mark.parametrize("mode", MODES)
def test_round_values_numpy_benchmark(benchmark: BenchmarkFixture, mode: str) -> None:
    data = np.array(make_values(SIZE))
    assert benchmark(round_values, data, mode=mode, decimals=2).shape == (SIZE,)


@torch_available
@pytest.mark.parametrize("mode", MODES)
def test_round_values_torch_benchmark(benchmark: BenchmarkFixture, mode: str) -> None:
    data = torch.tensor(make_values(SIZE), dtype=torch.float64)
    assert benchmark(round_values, data, mode=mode, decimals=2).shape == (SIZE,)


@pandas_available
@pytest.mark.parametrize("mode", MODES)
def test_round_values_pandas_series_benchmark(benchmark: BenchmarkFixture, mode: str) -> None:
    data = pd.Series(make_values(SIZE))
    assert len(benchmark(round_values, data, mode=mode, decimals=2)) == SIZE


@polars_available
@pytest.mark.parametrize("mode", MODES)
def test_round_values_polars_series_benchmark(benchmark: BenchmarkFixture, mode: str) -> None:
    data = pl.Series(make_values(SIZE))
    assert len(benchmark(round_values, data, mode=mode, decimals=2)) == SIZE


@pyarrow_available
@pytest.mark.parametrize("mode", MODES)
def test_round_values_pyarrow_array_benchmark(benchmark: BenchmarkFixture, mode: str) -> None:
    data = pa.array(make_values(SIZE))
    assert len(benchmark(round_values, data, mode=mode, decimals=2)) == SIZE


###########################
#     Recursive path      #
###########################


def test_round_values_recursive_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = make_values(NESTED_SIZE)
    assert len(benchmark(round_values, data, decimals=2, recursive=True)) == NESTED_SIZE


def test_round_values_recursive_nested_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    values = make_values(NESTED_SIZE)
    data = {"a": values[: NESTED_SIZE // 2], "b": {"c": values[NESTED_SIZE // 2 :], "d": "x"}}
    assert len(benchmark(round_values, data, decimals=2, recursive=True)) == 2
