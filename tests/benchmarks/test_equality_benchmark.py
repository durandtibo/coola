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

from typing import Any

import pytest

from coola.equality import compare, objects_are_allclose, objects_are_equal


def make_wide_nested_object(width: int, depth: int) -> dict:
    r"""Create a dict with ``width`` keys at each of ``depth`` nested
    levels."""
    obj = {f"key{i}": float(i) for i in range(width)}
    for _ in range(depth):
        obj = {f"level{i}": dict(obj) for i in range(width)}
    return obj


def make_deep_nested_object(depth: int) -> dict | int:
    r"""Create a dict nested ``depth`` levels deep."""
    obj: dict | int = 0
    for _ in range(depth):
        obj = {"child": obj}
    return obj


def test_objects_are_equal_flat_list_benchmark(benchmark) -> None:  # noqa: ANN001
    data = list(range(10_000))
    other = list(range(10_000))
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_wide_nested_benchmark(benchmark) -> None:  # noqa: ANN001
    data = make_wide_nested_object(width=10, depth=3)
    other = make_wide_nested_object(width=10, depth=3)
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_deep_nested_benchmark(benchmark) -> None:  # noqa: ANN001
    data = make_deep_nested_object(depth=50)
    other = make_deep_nested_object(depth=50)
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_allclose_flat_list_benchmark(benchmark) -> None:  # noqa: ANN001
    data = [float(i) for i in range(10_000)]
    other = [float(i) for i in range(10_000)]
    assert benchmark(objects_are_allclose, data, other)


def test_objects_are_equal_large_float_list_benchmark(benchmark) -> None:  # noqa: ANN001
    data = [float(i) for i in range(100_000)]
    other = [float(i) for i in range(100_000)]
    assert benchmark(objects_are_equal, data, other)


def make_records(size: int) -> dict[str, dict]:
    r"""Create a dict of ``size`` records ``{"x": float, "y": [int,
    int]}``."""
    return {f"k{i}": {"x": float(i), "y": [i, i + 1]} for i in range(size)}


def test_objects_are_equal_records_benchmark(benchmark) -> None:  # noqa: ANN001
    data, other = make_records(5_000), make_records(5_000)
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_records_different_benchmark(benchmark) -> None:  # noqa: ANN001
    data, other = make_records(5_000), make_records(5_000)
    other["k0"]["y"][1] = -1
    assert not benchmark(objects_are_equal, data, other)


def test_compare_large_float_list_different_at_first_benchmark(benchmark) -> None:  # noqa: ANN001
    data = [float(i) for i in range(1_000_000)]
    other = [-1.0, *data[1:]]
    assert not benchmark(compare, data, other)


def test_objects_are_equal_large_float_list_different_at_first_benchmark(benchmark) -> None:  # noqa: ANN001
    data = [float(i) for i in range(1_000_000)]
    other = [-1.0, *data[1:]]
    assert not benchmark(objects_are_equal, data, other)


def test_objects_are_equal_pandas_series_different_at_first_benchmark(benchmark) -> None:  # noqa: ANN001
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    values = np.random.default_rng(0).standard_normal(2_000_000)
    other = values.copy()
    other[0] += 1.0
    assert not benchmark(objects_are_equal, pd.Series(values), pd.Series(other))


def test_objects_are_equal_pandas_series_equal_benchmark(benchmark) -> None:  # noqa: ANN001
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    values = np.random.default_rng(0).standard_normal(2_000_000)
    assert benchmark(objects_are_equal, pd.Series(values), pd.Series(values.copy()))


def _frame_values() -> tuple[Any, Any]:
    np = pytest.importorskip("numpy")
    values = np.random.default_rng(0).standard_normal((200_000, 10))
    other = values.copy()
    other[0, 0] += 1.0
    return values, other


def test_objects_are_equal_pandas_frame_different_in_one_cell_benchmark(benchmark) -> None:  # noqa: ANN001
    pd = pytest.importorskip("pandas")
    values, other = _frame_values()
    assert not benchmark(objects_are_equal, pd.DataFrame(values), pd.DataFrame(other))


def test_objects_are_equal_polars_frame_different_in_one_cell_benchmark(benchmark) -> None:  # noqa: ANN001
    pl = pytest.importorskip("polars")
    values, other = _frame_values()
    assert not benchmark(objects_are_equal, pl.DataFrame(values), pl.DataFrame(other))


def test_objects_are_equal_pyarrow_table_different_in_one_cell_benchmark(benchmark) -> None:  # noqa: ANN001
    pa = pytest.importorskip("pyarrow")
    values, other = _frame_values()
    table = pa.table({str(i): values[:, i] for i in range(10)})
    table_other = pa.table({str(i): other[:, i] for i in range(10)})
    assert not benchmark(objects_are_equal, table, table_other)


def test_objects_are_allclose_numpy_array_benchmark(benchmark) -> None:  # noqa: ANN001
    np = pytest.importorskip("numpy")
    values = np.random.default_rng(0).standard_normal(1_000_000)
    assert benchmark(objects_are_allclose, values, values.copy())
