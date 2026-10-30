r"""Benchmarks for ``objects_are_equal`` and ``objects_are_allclose``.

These benchmarks are not part of the regular unit/integration test suite
(``inv unit-test`` and ``inv integration-test`` target ``tests/unit`` and
``tests/integration`` explicitly, so this directory is skipped). Run them
explicitly with:

    inv test.benchmark

They exist to catch performance regressions in the equality comparison
machinery (tester/handler dispatch) as the codebase evolves, especially for
large or deeply nested objects.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from coola.equality import (
    assert_objects_allclose,
    assert_objects_equal,
    compare,
    objects_are_allclose,
    objects_are_equal,
)
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
else:  # pragma: no cover
    np = Mock()

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    pd = Mock()

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    pl = Mock()

if is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    pa = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


def make_wide_nested_object(width: int, depth: int) -> dict:
    r"""Create a dict with ``width`` keys at each of ``depth`` nested
    levels."""
    obj: dict[str, Any] = {f"key{i}": float(i) for i in range(width)}
    for _ in range(depth):
        obj = {f"level{i}": dict(obj) for i in range(width)}
    return obj


def make_deep_nested_object(depth: int) -> dict | int:
    r"""Create a dict nested ``depth`` levels deep."""
    obj: dict | int = 0
    for _ in range(depth):
        obj = {"child": obj}
    return obj


def make_records(size: int) -> dict[str, dict]:
    r"""Create a dict of ``size`` records ``{"x": float, "y": [int,
    int]}``."""
    return {f"k{i}": {"x": float(i), "y": [i, i + 1]} for i in range(size)}


@dataclass
class Point:
    x: float
    y: float


def test_objects_are_equal_flat_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = list(range(10_000))
    other = list(range(10_000))
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_wide_nested_benchmark(benchmark: BenchmarkFixture) -> None:
    data = make_wide_nested_object(width=10, depth=3)
    other = make_wide_nested_object(width=10, depth=3)
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_deep_nested_benchmark(benchmark: BenchmarkFixture) -> None:
    data = make_deep_nested_object(depth=50)
    other = make_deep_nested_object(depth=50)
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_allclose_flat_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [float(i) for i in range(10_000)]
    other = [float(i) for i in range(10_000)]
    assert benchmark(objects_are_allclose, data, other)


def test_objects_are_equal_large_float_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [float(i) for i in range(100_000)]
    other = [float(i) for i in range(100_000)]
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_records_benchmark(benchmark: BenchmarkFixture) -> None:
    data, other = make_records(5_000), make_records(5_000)
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_records_different_benchmark(benchmark: BenchmarkFixture) -> None:
    data, other = make_records(5_000), make_records(5_000)
    other["k0"]["y"][1] = -1
    assert not benchmark(objects_are_equal, data, other)


def test_compare_large_float_list_different_at_first_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [float(i) for i in range(1_000_000)]
    other = [-1.0, *data[1:]]
    assert not benchmark(compare, data, other)


def test_objects_are_equal_large_float_list_different_at_first_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    data = [float(i) for i in range(1_000_000)]
    other = [-1.0, *data[1:]]
    assert not benchmark(objects_are_equal, data, other)


@numpy_available
@pandas_available
def test_objects_are_equal_pandas_series_different_at_first_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    values = np.random.default_rng(0).standard_normal(2_000_000)
    other = values.copy()
    other[0] += 1.0
    assert not benchmark(objects_are_equal, pd.Series(values), pd.Series(other))


@numpy_available
@pandas_available
def test_objects_are_equal_pandas_series_equal_benchmark(benchmark: BenchmarkFixture) -> None:
    values = np.random.default_rng(0).standard_normal(2_000_000)
    assert benchmark(objects_are_equal, pd.Series(values), pd.Series(values.copy()))


def _frame_values() -> tuple[Any, Any]:
    values = np.random.default_rng(0).standard_normal((200_000, 10))
    other = values.copy()
    other[0, 0] += 1.0
    return values, other


@numpy_available
@pandas_available
def test_objects_are_equal_pandas_frame_different_in_one_cell_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    values, other = _frame_values()
    assert not benchmark(objects_are_equal, pd.DataFrame(values), pd.DataFrame(other))


@numpy_available
@polars_available
def test_objects_are_equal_polars_frame_different_in_one_cell_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    values, other = _frame_values()
    assert not benchmark(objects_are_equal, pl.DataFrame(values), pl.DataFrame(other))


@numpy_available
@pyarrow_available
def test_objects_are_equal_pyarrow_table_different_in_one_cell_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    values, other = _frame_values()
    table = pa.table({str(i): values[:, i] for i in range(10)})
    table_other = pa.table({str(i): other[:, i] for i in range(10)})
    assert not benchmark(objects_are_equal, table, table_other)


@numpy_available
def test_objects_are_allclose_numpy_array_benchmark(benchmark: BenchmarkFixture) -> None:
    values = np.random.default_rng(0).standard_normal(1_000_000)
    assert benchmark(objects_are_allclose, values, values.copy())


def test_objects_are_allclose_nested_records_benchmark(benchmark: BenchmarkFixture) -> None:
    data, other = make_records(5_000), make_records(5_000)
    assert benchmark(objects_are_allclose, data, other, rtol=1e-3, atol=1e-6)


def test_objects_are_allclose_float_list_different_at_last_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    data = [float(i) for i in range(10_000)]
    other = [*data[:-1], -1.0]
    assert not benchmark(objects_are_allclose, data, other)


def test_objects_are_equal_different_types_benchmark(benchmark: BenchmarkFixture) -> None:
    # The types are compared first, so this should not traverse the lists.
    data = list(range(100_000))
    assert not benchmark(objects_are_equal, data, tuple(data))


def test_objects_are_equal_different_length_benchmark(benchmark: BenchmarkFixture) -> None:
    data = list(range(100_000))
    assert not benchmark(objects_are_equal, data, data[:-1])


def test_objects_are_equal_set_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(objects_are_equal, set(range(10_000)), set(range(10_000)))


def test_objects_are_equal_str_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [f"value{i}" for i in range(10_000)]
    assert benchmark(objects_are_equal, data, list(data))


def test_objects_are_equal_dataclass_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [Point(float(i), float(i + 1)) for i in range(2_000)]
    other = [Point(float(i), float(i + 1)) for i in range(2_000)]
    assert benchmark(objects_are_equal, data, other)


def test_compare_records_benchmark(benchmark: BenchmarkFixture) -> None:
    data, other = make_records(5_000), make_records(5_000)
    assert benchmark(compare, data, other)


def test_compare_records_different_at_last_benchmark(benchmark: BenchmarkFixture) -> None:
    # Building the difference report is the slow path of ``compare``.
    data, other = make_records(5_000), make_records(5_000)
    other["k4999"]["y"][1] = -1
    assert not benchmark(compare, data, other)


@numpy_available
def test_objects_are_equal_numpy_array_benchmark(benchmark: BenchmarkFixture) -> None:
    values = np.random.default_rng(0).standard_normal(1_000_000)
    assert benchmark(objects_are_equal, values, values.copy())


@numpy_available
def test_objects_are_equal_numpy_array_different_at_first_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    values = np.random.default_rng(0).standard_normal(1_000_000)
    other = values.copy()
    other[0] += 1.0
    assert not benchmark(objects_are_equal, values, other)


@torch_available
def test_objects_are_equal_torch_tensor_benchmark(benchmark: BenchmarkFixture) -> None:
    values = torch.randn(1_000_000, generator=torch.Generator().manual_seed(0))
    assert benchmark(objects_are_equal, values, values.clone())


def test_assert_objects_equal_wide_nested_benchmark(benchmark: BenchmarkFixture) -> None:
    data = {f"key{i}": {"values": list(range(10)), "name": f"n{i}"} for i in range(1_000)}
    assert benchmark(assert_objects_equal, data, copy.deepcopy(data)) is None


def test_assert_objects_allclose_flat_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [float(i) for i in range(10_000)]
    assert benchmark(assert_objects_allclose, data, list(data)) is None


def test_assert_objects_equal_different_at_last_benchmark(benchmark: BenchmarkFixture) -> None:
    actual = list(range(10_000))
    expected = [*actual[:-1], -1]

    def run() -> None:
        with pytest.raises(AssertionError, match=r"differ"):
            assert_objects_equal(actual, expected)

    benchmark(run)
