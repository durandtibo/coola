r"""Property-based tests for the guarantees of the reducers."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, strategies as st

from coola.reducer import BaseReducer, NativeReducer, NumpyReducer, TorchReducer
from coola.testing.fixtures import numpy_available, torch_available
from tests.properties.strategies import float_lists, quantile_lists

if TYPE_CHECKING:
    from collections.abc import Callable


def native() -> BaseReducer:
    return NativeReducer()


def numpy() -> BaseReducer:
    return NumpyReducer()


def torch() -> BaseReducer:
    return TorchReducer()


REDUCERS = [
    pytest.param(native, id="native"),
    pytest.param(numpy, id="numpy", marks=numpy_available),
    pytest.param(torch, id="torch", marks=torch_available),
]
OTHERS = [
    pytest.param(numpy, id="numpy", marks=numpy_available),
    pytest.param(torch, id="torch", marks=torch_available),
]


def close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-6, abs_tol=1e-6)


###########################################
#     Agreement with the native reducer   #
###########################################


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_max_min_and_sort_agree_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer, reference = make_reducer(), NativeReducer()
    assert reducer.max(values) == reference.max(values)
    assert reducer.min(values) == reference.min(values)
    assert reducer.sort(values) == reference.sort(values)
    assert reducer.sort(values, descending=True) == reference.sort(values, descending=True)


@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(values=float_lists, data=st.data())
def test_a_nan_propagates_whatever_its_position(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    index = data.draw(st.integers(0, len(values)))
    with_nan = [*values[:index], float("nan"), *values[index:]]
    reducer = make_reducer()
    for method in ("max", "min", "mean", "median", "std"):
        assert math.isnan(getattr(reducer, method)(with_nan)), method
    assert all(math.isnan(q) for q in reducer.quantile(with_nan, [0.0, 0.5, 1.0]))


@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(values=float_lists, data=st.data())
def test_sort_puts_the_nan_values_last(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    index = data.draw(st.integers(0, len(values)))
    with_nan = [*values[:index], float("nan"), *values[index:]]
    reducer = make_reducer()
    ascending = reducer.sort(with_nan)
    assert ascending[:-1] == sorted(values)
    assert math.isnan(ascending[-1])
    descending = reducer.sort(with_nan, descending=True)
    assert descending[1:] == sorted(values, reverse=True)
    assert math.isnan(descending[0])


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_mean_and_median_agree_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer, reference = make_reducer(), NativeReducer()
    assert close(reducer.mean(values), reference.mean(values))
    assert close(reducer.median(values), reference.median(values))


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_std_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    result, expected = make_reducer().std(values), NativeReducer().std(values)
    if len(values) == 1:
        assert math.isnan(result)
        assert math.isnan(expected)
    else:
        assert close(result, expected)


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists, quantiles=quantile_lists)
def test_quantile_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float], quantiles: list[float]
) -> None:
    result = make_reducer().quantile(values, quantiles)
    expected = NativeReducer().quantile(values, quantiles)
    assert all(close(r, e) for r, e in zip(result, expected, strict=True))


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=30))
def test_integers_agree_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[int]
) -> None:
    reducer, reference = make_reducer(), NativeReducer()
    assert reducer.max(values) == reference.max(values)
    assert reducer.min(values) == reference.min(values)
    assert reducer.sort(values) == reference.sort(values)
    assert close(reducer.mean(values), reference.mean(values))
    assert close(reducer.median(values), reference.median(values))


#######################
#     Bounds/order    #
#######################


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_mean_and_median_are_between_the_min_and_the_max(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    low, high = reducer.min(values), reducer.max(values)
    assert low - 1e-6 <= reducer.mean(values) <= high + 1e-6
    assert low <= reducer.median(values) <= high


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_sort_returns_the_sorted_values(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    ascending = reducer.sort(values)
    assert ascending == sorted(values)
    assert reducer.sort(values, descending=True) == ascending[::-1]
    assert reducer.min(values) == ascending[0]
    assert reducer.max(values) == ascending[-1]


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists, quantiles=quantile_lists)
def test_quantile_is_bounded_and_monotonic(
    make_reducer: Callable[[], BaseReducer], values: list[float], quantiles: list[float]
) -> None:
    reducer = make_reducer()
    result = reducer.quantile(values, sorted(quantiles))
    assert len(result) == len(quantiles)
    assert all(min(values) - 1e-6 <= q <= max(values) + 1e-6 for q in result)
    assert all(a <= b + 1e-6 for a, b in itertools.pairwise(result))


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_quantile_extremes_are_the_min_and_the_max(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    low, high = reducer.quantile(values, [0.0, 1.0])
    assert close(low, reducer.min(values))
    assert close(high, reducer.max(values))


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_quantile_of_one_half_is_the_median(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert close(reducer.quantile(values, [0.5])[0], reducer.median(values))


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_std_is_not_negative(make_reducer: Callable[[], BaseReducer], values: list[float]) -> None:
    result = make_reducer().std(values)
    assert math.isnan(result) if len(values) == 1 else result >= 0


#########################
#     Invariances       #
#########################


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists, data=st.data())
def test_the_statistics_do_not_depend_on_the_order(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    reducer = make_reducer()
    shuffled = data.draw(st.permutations(values))
    assert reducer.max(shuffled) == reducer.max(values)
    assert reducer.min(shuffled) == reducer.min(values)
    assert reducer.median(shuffled) == reducer.median(values)
    assert close(reducer.mean(shuffled), reducer.mean(values))
    assert reducer.sort(shuffled) == reducer.sort(values)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_the_reducers_do_not_modify_the_input_and_return_python_numbers(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer, before = make_reducer(), list(values)
    results = [
        reducer.max(values),
        reducer.min(values),
        reducer.mean(values),
        reducer.median(values),
        reducer.std(values),
    ]
    reducer.sort(values)
    reducer.quantile(values, [0.5])
    assert values == before
    assert all(type(result) in {int, float} for result in results)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(value=st.floats(min_value=-1e6, max_value=1e6), size=st.integers(1, 20))
def test_a_constant_sequence_has_that_value_as_statistic(
    make_reducer: Callable[[], BaseReducer], value: float, size: int
) -> None:
    reducer, values = make_reducer(), [value] * size
    assert reducer.max(values) == value
    assert reducer.min(values) == value
    assert close(reducer.mean(values), value)
    assert close(reducer.median(values), value)
    if size > 1:
        assert close(reducer.std(values), 0.0)
