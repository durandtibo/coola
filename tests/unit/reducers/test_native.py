from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from coola.reducers import EmptySequenceError, NativeReducer, ReducerRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence


EMPTY_SEQUENCES = [[], ()]


def test_reducer_registry_available_reducers() -> None:
    assert isinstance(ReducerRegistry.registry["native"], NativeReducer)


##################################
#     Tests for BasicReducer     #
##################################


def test_native_reducer_str() -> None:
    assert str(NativeReducer()).startswith("NativeReducer(")


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3]])
def test_native_reducer_max_int(values: Sequence[int | float]) -> None:
    val = NativeReducer().max(values)
    assert isinstance(val, int)
    assert val == 2


@pytest.mark.parametrize(
    "values", [[-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [2.5], [2.5, -1.5, -2, -3]]
)
def test_native_reducer_max_float(values: Sequence[int | float]) -> None:
    val = NativeReducer().max(values)
    assert isinstance(val, float)
    assert val == 2.5


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_max_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the maximum because the summary is empty"
    ):
        NativeReducer().max(values)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]])
def test_native_reducer_mean_int(values: Sequence[int | float]) -> None:
    val = NativeReducer().mean(values)
    assert isinstance(val, float)
    assert val == 0.0


@pytest.mark.parametrize(
    "values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]]
)
def test_native_reducer_mean_float(values: Sequence[int | float]) -> None:
    val = NativeReducer().mean(values)
    assert isinstance(val, float)
    assert val == 0.5


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_mean_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the mean because the summary is empty"
    ):
        NativeReducer().mean(values)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]])
def test_native_reducer_median_int(values: Sequence[int | float]) -> None:
    val = NativeReducer().median(values)
    assert isinstance(val, int)
    assert val == 0


@pytest.mark.parametrize(
    "values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]]
)
def test_native_reducer_median_float(values: Sequence[int | float]) -> None:
    val = NativeReducer().median(values)
    assert isinstance(val, float)
    assert val == 0.5


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_median_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the median because the summary is empty"
    ):
        NativeReducer().median(values)


@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3]])
def test_native_reducer_min_int(values: Sequence[int | float]) -> None:
    val = NativeReducer().min(values)
    assert isinstance(val, int)
    assert val == -2


@pytest.mark.parametrize(
    "values", [[-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [-2.5], [-2.5, 1.5, 2, 3]]
)
def test_native_reducer_min_float(values: Sequence[int | float]) -> None:
    val = NativeReducer().min(values)
    assert isinstance(val, float)
    assert val == -2.5


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_min_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the minimum because the summary is empty"
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
        EmptySequenceError, match=r"Cannot compute the quantiles because the summary is empty"
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


@pytest.mark.parametrize("values", [[1], [1.0]])
def test_native_reducer_std_one(values: Sequence[int | float]) -> None:
    assert math.isnan(NativeReducer().std(values))


@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_native_reducer_std_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError,
        match=r"Cannot compute the standard deviation because the summary is empty",
    ):
        NativeReducer().std(values)
