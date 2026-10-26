from __future__ import annotations

import math

import pytest

from coola.rounding.pyarrow import round_pyarrow
from coola.testing.fixtures import pyarrow_available
from coola.utils.imports import is_pyarrow_available

if is_pyarrow_available():
    import pyarrow as pa

######################################
#     Tests for round_pyarrow        #
######################################


@pyarrow_available
def test_round_pyarrow_array_floor() -> None:
    result = round_pyarrow(pa.array([2.567, -2.567, 0.29]), "floor", 2)
    assert result.to_pylist() == [2.56, -2.57, 0.29]


@pyarrow_available
def test_round_pyarrow_array_ceil() -> None:
    result = round_pyarrow(pa.array([2.561, -2.561, 0.07]), "ceil", 2)
    assert result.to_pylist() == [2.57, -2.56, 0.07]


@pyarrow_available
def test_round_pyarrow_array_round() -> None:
    assert round_pyarrow(pa.array([0.5, 1.5, 2.5]), "round", 0).to_pylist() == [0.0, 2.0, 2.0]


@pyarrow_available
def test_round_pyarrow_array_null_kept() -> None:
    assert round_pyarrow(pa.array([1.26, None]), "floor", 1).to_pylist() == [1.2, None]


@pyarrow_available
def test_round_pyarrow_array_special_values_kept() -> None:
    result = round_pyarrow(pa.array([math.inf, math.nan, 1e308]), "floor", 2).to_pylist()
    assert result[0] == math.inf
    assert math.isnan(result[1])
    assert result[2] == 1e308


@pyarrow_available
def test_round_pyarrow_array_type_kept() -> None:
    assert round_pyarrow(pa.array([1.26], pa.float32()), "floor", 1).type == pa.float32()


@pyarrow_available
def test_round_pyarrow_array_float32_value_on_the_grid() -> None:
    array = pa.array([0.29, 0.07], pa.float32())
    assert round_pyarrow(array, "floor", 2).equals(array)


@pyarrow_available
def test_round_pyarrow_array_integer_unchanged() -> None:
    array = pa.array([1, 2])
    assert round_pyarrow(array, "floor", 1).equals(array)


@pyarrow_available
def test_round_pyarrow_chunked_array() -> None:
    array = pa.chunked_array([[1.26], [2.34]])
    assert round_pyarrow(array, "floor", 1).to_pylist() == [1.2, 2.3]


@pyarrow_available
def test_round_pyarrow_empty_array() -> None:
    assert len(round_pyarrow(pa.array([], pa.float64()), "floor", 1)) == 0


@pyarrow_available
def test_round_pyarrow_array_too_many_decimals_unchanged() -> None:
    assert round_pyarrow(pa.array([0.1]), "floor", 400).to_pylist() == [0.1]


@pyarrow_available
@pytest.mark.parametrize("array", [pa.array(["a"]), pa.array([True])])
def test_round_pyarrow_array_unsupported_type(array: pa.Array) -> None:
    with pytest.raises(TypeError, match=r"Unsupported type"):
        round_pyarrow(array, "round", 0)


@pyarrow_available
def test_round_pyarrow_table() -> None:
    table = pa.table({"a": [1.26, None], "b": [1, 2], "c": ["x", "y"]})
    assert round_pyarrow(table, "floor", 1).equals(
        pa.table({"a": [1.2, None], "b": [1, 2], "c": ["x", "y"]})
    )


@pyarrow_available
def test_round_pyarrow_table_keeps_field_metadata() -> None:
    schema = pa.schema([pa.field("a", pa.float64(), metadata={"unit": "m"})])
    table = pa.table({"a": [1.26]}, schema=schema)
    assert round_pyarrow(table, "floor", 1).schema.field("a").metadata == {b"unit": b"m"}


@pyarrow_available
def test_round_pyarrow_table_no_numerical_column() -> None:
    table = pa.table({"a": ["x"]})
    assert round_pyarrow(table, "floor", 1).equals(table)


@pyarrow_available
def test_round_pyarrow_unsupported_type() -> None:
    with pytest.raises(TypeError, match=r"Unsupported type"):
        round_pyarrow([1.5], "round", 0)
