from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.reducer import EmptySequenceError, NativeReducer
from coola.testing.marks import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover

if is_torch_available():
    import torch
else:
    torch = Mock()  # pragma: no cover

if TYPE_CHECKING:
    from collections.abc import Sequence


EMPTY_SEQUENCES = [[], ()]


##################################
#     Tests for BasicReducer     #
##################################


def test_native_reducer_str() -> None:
    assert str(NativeReducer()).startswith("NativeReducer(")


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3]])
def test_native_reducer_max_int_type(values: Sequence[int | float]) -> None:
    val = NativeReducer().max(values)
    assert isinstance(val, int)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3]])
def test_native_reducer_max_int_value(values: Sequence[int | float]) -> None:
    val = NativeReducer().max(values)
    assert val == 2


@pytest.mark.parametrize(
    "values", [[-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [2.5], [2.5, -1.5, -2, -3]]
)
def test_native_reducer_max_float_type(values: Sequence[int | float]) -> None:
    val = NativeReducer().max(values)
    assert isinstance(val, float)


@pytest.mark.parametrize(
    "values", [[-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [2.5], [2.5, -1.5, -2, -3]]
)
def test_native_reducer_max_float_value(values: Sequence[int | float]) -> None:
    val = NativeReducer().max(values)
    assert val == 2.5


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_max_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the maximum because the sequence is empty"
    ):
        NativeReducer().max(values)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]])
def test_native_reducer_mean_int_type(values: Sequence[int | float]) -> None:
    val = NativeReducer().mean(values)
    assert isinstance(val, float)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]])
def test_native_reducer_mean_int_value(values: Sequence[int | float]) -> None:
    val = NativeReducer().mean(values)
    assert val == 0.0


@pytest.mark.parametrize(
    "values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]]
)
def test_native_reducer_mean_float_type(values: Sequence[int | float]) -> None:
    val = NativeReducer().mean(values)
    assert isinstance(val, float)


@pytest.mark.parametrize(
    "values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]]
)
def test_native_reducer_mean_float_value(values: Sequence[int | float]) -> None:
    val = NativeReducer().mean(values)
    assert val == 0.5


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_mean_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the mean because the sequence is empty"
    ):
        NativeReducer().mean(values)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]])
def test_native_reducer_median_int_type(values: Sequence[int | float]) -> None:
    val = NativeReducer().median(values)
    assert isinstance(val, int)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]])
def test_native_reducer_median_int_value(values: Sequence[int | float]) -> None:
    val = NativeReducer().median(values)
    assert val == 0


@pytest.mark.parametrize(
    "values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]]
)
def test_native_reducer_median_float_type(values: Sequence[int | float]) -> None:
    val = NativeReducer().median(values)
    assert isinstance(val, float)


@pytest.mark.parametrize(
    "values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]]
)
def test_native_reducer_median_float_value(values: Sequence[int | float]) -> None:
    val = NativeReducer().median(values)
    assert val == 0.5


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_median_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the median because the sequence is empty"
    ):
        NativeReducer().median(values)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3]])
def test_native_reducer_min_int_type(values: Sequence[int | float]) -> None:
    val = NativeReducer().min(values)
    assert isinstance(val, int)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3]])
def test_native_reducer_min_int_value(values: Sequence[int | float]) -> None:
    val = NativeReducer().min(values)
    assert val == -2


@pytest.mark.parametrize(
    "values", [[-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [-2.5], [-2.5, 1.5, 2, 3]]
)
def test_native_reducer_min_float_type(values: Sequence[int | float]) -> None:
    val = NativeReducer().min(values)
    assert isinstance(val, float)


@pytest.mark.parametrize(
    "values", [[-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [-2.5], [-2.5, 1.5, 2, 3]]
)
def test_native_reducer_min_float_value(values: Sequence[int | float]) -> None:
    val = NativeReducer().min(values)
    assert val == -2.5


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_min_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the minimum because the sequence is empty"
    ):
        NativeReducer().min(values)


@pytest.mark.parametrize(
    "values", [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
)
def test_native_reducer_quantile_int(values: Sequence[int | float]) -> None:
    assert NativeReducer().quantile(values, (0.2, 0.5, 0.9)) == [2, 5, 9]


@pytest.mark.parametrize(
    "values",
    [
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5),
    ],
)
def test_native_reducer_quantile_float(values: Sequence[int | float]) -> None:
    assert NativeReducer().quantile(values, (0.0, 0.1, 0.4, 0.9)) == [0.5, 1.5, 4.5, 9.5]


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_quantile_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the quantiles because the sequence is empty"
    ):
        NativeReducer().quantile(values, [0.5])


@pytest.mark.parametrize("values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0)])
def test_native_reducer_sort_int(values: Sequence[int | float]) -> None:
    assert NativeReducer().sort(values) == [-2, 0, 1, 2, 3]


@pytest.mark.parametrize("values", [[2.5, 1.5, -2.5, 3.5, 0.5], (2.5, 1.5, -2.5, 3.5, 0.5)])
def test_native_reducer_sort_float(values: Sequence[int | float]) -> None:
    assert NativeReducer().sort(values) == [-2.5, 0.5, 1.5, 2.5, 3.5]


@pytest.mark.parametrize("values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0)])
def test_native_reducer_sort_descending(values: Sequence[int | float]) -> None:
    assert NativeReducer().sort(values, descending=True) == [3, 2, 1, 0, -2]


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_sort_empty(values: Sequence[int | float]) -> None:
    assert NativeReducer().sort(values) == []


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2)])
def test_native_reducer_std_int(values: Sequence[int | float]) -> None:
    assert math.isclose(NativeReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@pytest.mark.parametrize("values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5)])
def test_native_reducer_std_float(values: Sequence[int | float]) -> None:
    assert math.isclose(NativeReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@pytest.mark.parametrize("values", [[1], (1,), [1.0], (1.0,)])
def test_native_reducer_std_one(values: Sequence[int | float]) -> None:
    assert math.isnan(NativeReducer().std(values))


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_std_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError,
        match=r"Cannot compute the standard deviation because the sequence is empty",
    ):
        NativeReducer().std(values)


def test_native_reducer_max_nan_does_not_depend_on_the_order_nan_first() -> None:
    nan = float("nan")
    assert math.isnan(NativeReducer().max([1.0, nan, 2.0]))


def test_native_reducer_max_nan_does_not_depend_on_the_order_nan_second() -> None:
    nan = float("nan")
    assert math.isnan(NativeReducer().max([nan, 1.0, 2.0]))


def test_native_reducer_min_nan_does_not_depend_on_the_order_nan_first() -> None:
    nan = float("nan")
    assert math.isnan(NativeReducer().min([1.0, nan, 2.0]))


def test_native_reducer_min_nan_does_not_depend_on_the_order_nan_second() -> None:
    nan = float("nan")
    assert math.isnan(NativeReducer().min([nan, 1.0, 2.0]))


def test_native_reducer_sort_ints_unchanged_ascending() -> None:
    assert NativeReducer().sort([3, 1, 2]) == [1, 2, 3]


def test_native_reducer_sort_ints_unchanged_descending() -> None:
    assert NativeReducer().sort([3, 1, 2], descending=True) == [3, 2, 1]


def test_native_reducer_std_infinite_value_is_nan() -> None:
    assert math.isnan(NativeReducer().std([1.0, float("-inf"), 2.0]))


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([float("nan"), 1.0, 2.0], id="nan first"),
        pytest.param([1.0, float("nan"), 2.0], id="nan second"),
        pytest.param([1.0, 2.0, float("nan")], id="nan last"),
    ],
)
def test_native_reducer_median_nan_does_not_depend_on_the_order(values: list[float]) -> None:
    assert math.isnan(NativeReducer().median(values))


def test_native_reducer_quantile_nan_length() -> None:
    result = NativeReducer().quantile([1.0, float("nan"), 2.0], [0.0, 0.5, 1.0])
    assert len(result) == 3


def test_native_reducer_quantile_nan_all_nan() -> None:
    result = NativeReducer().quantile([1.0, float("nan"), 2.0], [0.0, 0.5, 1.0])
    assert all(math.isnan(q) for q in result)


def test_native_reducer_median_does_not_overflow() -> None:
    assert NativeReducer().median([1e308, 1e308]) == 1e308


def test_native_reducer_median_infinite_values() -> None:
    assert NativeReducer().median([float("inf"), float("inf")]) == float("inf")


def test_native_reducer_max_generator_raises() -> None:
    with pytest.raises(TypeError, match="iterator"):
        NativeReducer().max(i for i in range(3))  # ty: ignore[invalid-argument-type]


@numpy_available
def test_native_reducer_array_raises() -> None:
    with pytest.raises(
        TypeError,
        match=r"NativeReducer does not support an input of type ndarray; use NumpyReducer",
    ):
        NativeReducer().max(np.array([1, 2, 3]))  # ty: ignore[invalid-argument-type]


@torch_available
def test_native_reducer_tensor_raises() -> None:
    with pytest.raises(
        TypeError, match=r"NativeReducer does not support an input of type Tensor; use TorchReducer"
    ):
        NativeReducer().max(torch.tensor([1, 2, 3]))  # ty: ignore[invalid-argument-type]


def test_native_reducer_dict_raises() -> None:
    with pytest.raises(TypeError, match=r"NativeReducer does not support an input of type dict$"):
        NativeReducer().max({1: 2})  # ty: ignore[invalid-argument-type]


def test_native_reducer_max_range() -> None:
    assert NativeReducer().max(range(5)) == 4


def test_native_reducer_big_int() -> None:
    assert NativeReducer().max([2**70, 1]) == 1180591620717411303424


def test_native_reducer_mean_sum_exceeds_float_range() -> None:
    assert NativeReducer().mean([1e308, 1e308]) == 1e308


def test_native_reducer_mean_int_returns_float() -> None:
    assert type(NativeReducer().mean([1, 2, 3])) is float


def test_native_reducer_mean_big_int() -> None:
    assert NativeReducer().mean([2**70, 2**70]) == 1.1805916207174113e21


def test_native_reducer_mean_cancellation() -> None:
    assert NativeReducer().mean([1e16, 1.0, -1e16, 3.0]) == 1.0
