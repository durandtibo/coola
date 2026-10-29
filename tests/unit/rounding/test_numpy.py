from __future__ import annotations

import math

import pytest

from coola.rounding.numpy import get_default_rounders, round_numpy
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np

####################################
#     Tests for round_numpy        #
####################################


@numpy_available
def test_round_numpy_round_half_to_even() -> None:
    assert round_numpy(np.array([0.5, 1.5, 2.5, -0.5, -1.5]), "round", 0).tolist() == [
        0.0,
        2.0,
        2.0,
        -0.0,
        -2.0,
    ]


@numpy_available
def test_round_numpy_floor() -> None:
    assert round_numpy(np.array([2.567, -2.567, 2.0]), "floor", 0).tolist() == [2.0, -3.0, 2.0]


@numpy_available
def test_round_numpy_ceil() -> None:
    assert round_numpy(np.array([2.567, -2.567, 2.0]), "ceil", 0).tolist() == [3.0, -2.0, 2.0]


@numpy_available
def test_round_numpy_floor_decimals() -> None:
    assert round_numpy(np.array([2.567, 0.29, -2.567]), "floor", 2).tolist() == [2.56, 0.29, -2.57]


@numpy_available
def test_round_numpy_ceil_decimals() -> None:
    assert round_numpy(np.array([2.561, 0.07, -2.561]), "ceil", 2).tolist() == [2.57, 0.07, -2.56]


@numpy_available
def test_round_numpy_round_decimals() -> None:
    assert round_numpy(np.array([1.234, 5.678]), "round", 1).tolist() == [1.2, 5.7]


@numpy_available
def test_round_numpy_special_values_kept() -> None:
    result = round_numpy(np.array([np.inf, -np.inf, np.nan, 1e308]), "floor", 2)
    assert result.tolist()[:2] == [math.inf, -math.inf]
    assert math.isnan(result[2])
    assert result[3] == 1e308


@numpy_available
@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
@pytest.mark.parametrize("dtype", ["int8", "int64", "uint8", "bool"])
def test_round_numpy_integer_and_bool_unchanged(mode: str, dtype: str) -> None:
    data = np.array([0, 1, 1], dtype=dtype)
    assert np.array_equal(round_numpy(data, mode, 2), data)  # ty: ignore[invalid-argument-type]


@numpy_available
def test_round_numpy_integer_dtype_kept() -> None:
    assert round_numpy(np.array([1, 2], dtype=np.int32), "floor", 1).dtype == np.int32


@numpy_available
def test_round_numpy_integer_does_not_share_memory() -> None:
    data = np.array([1, 2])
    assert not np.shares_memory(round_numpy(data, "floor", 1), data)


@numpy_available
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_round_numpy_float_dtype_kept(dtype: str) -> None:
    assert round_numpy(np.array([1.25], dtype=dtype), "floor", 1).dtype == np.dtype(dtype)


@numpy_available
def test_round_numpy_float32_value_on_the_grid() -> None:
    data = np.array([0.29, 0.07], dtype=np.float32)
    assert np.array_equal(round_numpy(data, "floor", 2), data)


@numpy_available
def test_round_numpy_float32_ceil_value_on_the_grid() -> None:
    data = np.array([0.29, 0.07], dtype=np.float32)
    assert np.array_equal(round_numpy(data, "ceil", 2), data)


@numpy_available
def test_round_numpy_does_not_modify_input() -> None:
    data = np.array([1.26, 2.5])
    round_numpy(data, "floor", 1)
    assert data.tolist() == [1.26, 2.5]


@numpy_available
def test_round_numpy_multidimensional_shape() -> None:
    assert round_numpy(np.ones((2, 3, 4)) * 1.26, "floor", 1).shape == (2, 3, 4)


@numpy_available
def test_round_numpy_multidimensional_values() -> None:
    assert round_numpy(np.array([[1.26, 2.34], [3.45, 4.56]]), "floor", 1).tolist() == [
        [1.2, 2.3],
        [3.4, 4.5],
    ]


@numpy_available
def test_round_numpy_empty() -> None:
    assert round_numpy(np.array([], dtype=float), "floor", 1).shape == (0,)


@numpy_available
def test_round_numpy_non_contiguous() -> None:
    assert round_numpy(np.array([[1.26, 2.34], [3.45, 4.56]])[:, 0], "floor", 1).tolist() == [
        1.2,
        3.4,
    ]


@numpy_available
def test_round_numpy_too_many_decimals_unchanged() -> None:
    assert round_numpy(np.array([0.1]), "floor", 400).tolist() == [0.1]


@numpy_available
def test_round_numpy_scalar_value() -> None:
    assert round_numpy(np.float64(2.567), "floor", 2) == 2.56


@numpy_available
def test_round_numpy_scalar_type() -> None:
    assert type(round_numpy(np.float32(2.567), "floor", 2)) is np.float32


@numpy_available
def test_round_numpy_integer_scalar() -> None:
    assert round_numpy(np.int64(3), "floor", 2) == 3


@numpy_available
def test_round_numpy_masked_array_values() -> None:
    data = np.ma.array([1.26, 2.34, 3.45], mask=[False, True, False])
    assert round_numpy(data, "floor", 1).tolist() == [1.2, None, 3.4]


@numpy_available
def test_round_numpy_masked_array_type() -> None:
    data = np.ma.array([1.26, 2.34], mask=[False, True])
    assert isinstance(round_numpy(data, "floor", 1), np.ma.MaskedArray)


@numpy_available
@pytest.mark.parametrize(
    "data", [np.array(["a"]), np.array([1 + 2j]), np.array([None], dtype=object)]
)
def test_round_numpy_unsupported_dtype(data: np.ndarray) -> None:
    with pytest.raises(TypeError, match=r"Unsupported dtype"):
        round_numpy(data, "round", 0)


@numpy_available
def test_round_numpy_unsupported_type() -> None:
    with pytest.raises(TypeError, match=r"Unsupported type"):
        round_numpy([1.5], "round", 0)


##########################################
#     Tests for the one decimal rule     #
##########################################


@numpy_available
def test_round_numpy_ceil_one_ulp_above_grid_point() -> None:
    assert round_numpy(np.array([5.5200000000000005]), "ceil", 2).tolist() == [5.53]


@numpy_available
def test_round_numpy_round_halfway_decimal_goes_to_even() -> None:
    assert round_numpy(np.array([2.675, 0.125, 8.995]), "round", 2).tolist() == [2.68, 0.12, 9.0]


@numpy_available
def test_round_numpy_negative_zero_result_keeps_sign() -> None:
    assert math.copysign(1.0, round_numpy(np.array([-0.4]), "ceil", 0).item()) == -1.0


@numpy_available
def test_round_numpy_float16_idempotent() -> None:
    data = np.array([1.2345], dtype="float16")
    once = round_numpy(data, "round", 2)
    assert round_numpy(once, "round", 2).tolist() == once.tolist()


@numpy_available
def test_round_numpy_float32_round_halfway_decimal() -> None:
    assert round_numpy(np.array([2.675], dtype="float32"), "round", 2).tolist() == [
        np.float32(2.68).item()
    ]


@numpy_available
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_round_numpy_matrix_type_kept() -> None:
    assert isinstance(round_numpy(np.matrix([[1.26, 2.34]]), "floor", 1), np.matrix)


@numpy_available
def test_round_numpy_ndarray_subclass_type_kept() -> None:
    class MyArray(np.ndarray):
        pass

    assert isinstance(round_numpy(np.array([1.26]).view(MyArray), "floor", 1), MyArray)


@numpy_available
def test_round_numpy_masked_array_fill_value_kept() -> None:
    data = np.ma.array([1.26, 2.34], mask=[False, True], fill_value=-1.0)
    assert round_numpy(data, "floor", 1).fill_value == -1.0


@numpy_available
def test_round_numpy_empty_float_keeps_dtype() -> None:
    assert round_numpy(np.array([], dtype="float32"), "round", 1).dtype == np.float32


@numpy_available
def test_round_numpy_floor_value_just_below_grid_point() -> None:
    assert round_numpy(np.array([0.8999999999999999]), "floor", 1).tolist() == [0.8]


#############################################
#     Tests for get_default_rounders        #
#############################################


@numpy_available
def test_get_default_rounders_numpy() -> None:
    assert get_default_rounders() == {np.ndarray: round_numpy, np.generic: round_numpy}


@numpy_available
def test_get_default_rounders_numpy_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("coola.rounding.numpy.is_numpy_available", lambda: False)
    assert get_default_rounders() == {}
