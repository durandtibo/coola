r"""Property-based tests for the guarantees of the reducers."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import pytest
from hypothesis import assume, given, strategies as st

from coola.reducer import BaseReducer, NativeReducer, NumpyReducer, TorchReducer
from coola.testing.marks import numpy_available, torch_available
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
def test_max_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assert make_reducer().max(values) == NativeReducer().max(values)


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_min_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assert make_reducer().min(values) == NativeReducer().min(values)


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_sort_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assert make_reducer().sort(values) == NativeReducer().sort(values)


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_sort_descending_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assert make_reducer().sort(values, descending=True) == NativeReducer().sort(
        values, descending=True
    )


def insert_nan(values: list[float], index: int) -> list[float]:
    r"""Insert a NaN in ``values`` at ``index``."""
    return [*values[:index], float("nan"), *values[index:]]


@pytest.mark.parametrize("method", ["max", "min", "mean", "median", "std"])
@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(values=float_lists, data=st.data())
def test_a_nan_propagates_whatever_its_position(
    make_reducer: Callable[[], BaseReducer], method: str, values: list[float], data: st.DataObject
) -> None:
    with_nan = insert_nan(values, data.draw(st.integers(0, len(values))))
    assert math.isnan(getattr(make_reducer(), method)(with_nan))


@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(values=float_lists, data=st.data())
def test_a_nan_propagates_to_the_quantiles_whatever_its_position(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    with_nan = insert_nan(values, data.draw(st.integers(0, len(values))))
    assert all(math.isnan(q) for q in make_reducer().quantile(with_nan, [0.0, 0.5, 1.0]))


@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(values=float_lists, data=st.data())
def test_sort_puts_the_nan_values_last(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    with_nan = insert_nan(values, data.draw(st.integers(0, len(values))))
    assert make_reducer().sort(with_nan)[:-1] == sorted(values)


@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(values=float_lists, data=st.data())
def test_sort_ends_with_a_nan(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    with_nan = insert_nan(values, data.draw(st.integers(0, len(values))))
    assert math.isnan(make_reducer().sort(with_nan)[-1])


@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(values=float_lists, data=st.data())
def test_sort_descending_puts_the_nan_values_first(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    with_nan = insert_nan(values, data.draw(st.integers(0, len(values))))
    assert make_reducer().sort(with_nan, descending=True)[1:] == sorted(values, reverse=True)


@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(values=float_lists, data=st.data())
def test_sort_descending_starts_with_a_nan(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    with_nan = insert_nan(values, data.draw(st.integers(0, len(values))))
    assert math.isnan(make_reducer().sort(with_nan, descending=True)[0])


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_mean_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assert close(make_reducer().mean(values), NativeReducer().mean(values))


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_median_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assert close(make_reducer().median(values), NativeReducer().median(values))


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists)
def test_std_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assume(len(values) > 1)
    assert close(make_reducer().std(values), NativeReducer().std(values))


@pytest.mark.parametrize("make_reducer", [NativeReducer, *OTHERS])
@given(value=st.floats(min_value=-1e6, max_value=1e6))
def test_std_of_a_single_value_is_nan(
    make_reducer: Callable[[], BaseReducer], value: float
) -> None:
    assert math.isnan(make_reducer().std([value]))


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=float_lists, quantiles=quantile_lists)
def test_quantile_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[float], quantiles: list[float]
) -> None:
    result = make_reducer().quantile(values, quantiles)
    expected = NativeReducer().quantile(values, quantiles)
    assert all(close(r, e) for r, e in zip(result, expected, strict=True))


integer_lists = st.lists(st.integers(-1000, 1000), min_size=1, max_size=30)


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=integer_lists)
def test_integers_max_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[int]
) -> None:
    assert make_reducer().max(values) == NativeReducer().max(values)


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=integer_lists)
def test_integers_min_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[int]
) -> None:
    assert make_reducer().min(values) == NativeReducer().min(values)


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=integer_lists)
def test_integers_sort_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[int]
) -> None:
    assert make_reducer().sort(values) == NativeReducer().sort(values)


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=integer_lists)
def test_integers_mean_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[int]
) -> None:
    assert close(make_reducer().mean(values), NativeReducer().mean(values))


@pytest.mark.parametrize("make_reducer", OTHERS)
@given(values=integer_lists)
def test_integers_median_agrees_with_the_native_reducer(
    make_reducer: Callable[[], BaseReducer], values: list[int]
) -> None:
    assert close(make_reducer().median(values), NativeReducer().median(values))


#######################
#     Bounds/order    #
#######################


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_mean_is_not_below_the_min(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert reducer.min(values) - 1e-6 <= reducer.mean(values)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_mean_is_not_above_the_max(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert reducer.mean(values) <= reducer.max(values) + 1e-6


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_median_is_not_below_the_min(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert reducer.min(values) <= reducer.median(values)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_median_is_not_above_the_max(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert reducer.median(values) <= reducer.max(values)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_sort_returns_the_sorted_values(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assert make_reducer().sort(values) == sorted(values)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_sort_descending_returns_the_reversed_sorted_values(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    assert make_reducer().sort(values, descending=True) == sorted(values)[::-1]


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_min_is_the_first_sorted_value(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert reducer.min(values) == reducer.sort(values)[0]


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_max_is_the_last_sorted_value(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert reducer.max(values) == reducer.sort(values)[-1]


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists, quantiles=quantile_lists)
def test_quantile_returns_one_value_for_each_quantile(
    make_reducer: Callable[[], BaseReducer], values: list[float], quantiles: list[float]
) -> None:
    assert len(make_reducer().quantile(values, sorted(quantiles))) == len(quantiles)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists, quantiles=quantile_lists)
def test_quantile_is_bounded(
    make_reducer: Callable[[], BaseReducer], values: list[float], quantiles: list[float]
) -> None:
    result = make_reducer().quantile(values, sorted(quantiles))
    assert all(min(values) - 1e-6 <= q <= max(values) + 1e-6 for q in result)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists, quantiles=quantile_lists)
def test_quantile_is_monotonic(
    make_reducer: Callable[[], BaseReducer], values: list[float], quantiles: list[float]
) -> None:
    result = make_reducer().quantile(values, sorted(quantiles))
    assert all(a <= b + 1e-6 for a, b in itertools.pairwise(result))


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_quantile_of_zero_is_the_min(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert close(reducer.quantile(values, [0.0])[0], reducer.min(values))


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_quantile_of_one_is_the_max(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert close(reducer.quantile(values, [1.0])[0], reducer.max(values))


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
    assume(len(values) > 1)
    assert make_reducer().std(values) >= 0


#########################
#     Invariances       #
#########################


@pytest.mark.parametrize("method", ["max", "min", "median"])
@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists, data=st.data())
def test_the_exact_statistics_do_not_depend_on_the_order(
    make_reducer: Callable[[], BaseReducer], method: str, values: list[float], data: st.DataObject
) -> None:
    reducer = make_reducer()
    shuffled = data.draw(st.permutations(values))
    assert getattr(reducer, method)(shuffled) == getattr(reducer, method)(values)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists, data=st.data())
def test_the_mean_does_not_depend_on_the_order(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    reducer = make_reducer()
    shuffled = data.draw(st.permutations(values))
    assert close(reducer.mean(shuffled), reducer.mean(values))


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists, data=st.data())
def test_the_sort_does_not_depend_on_the_order(
    make_reducer: Callable[[], BaseReducer], values: list[float], data: st.DataObject
) -> None:
    reducer = make_reducer()
    shuffled = data.draw(st.permutations(values))
    assert reducer.sort(shuffled) == reducer.sort(values)


@pytest.mark.parametrize("method", ["max", "min", "mean", "median", "std", "sort"])
@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_the_reducers_do_not_modify_the_input(
    make_reducer: Callable[[], BaseReducer], method: str, values: list[float]
) -> None:
    before = list(values)
    getattr(make_reducer(), method)(values)
    assert values == before


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_quantile_does_not_modify_the_input(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    before = list(values)
    make_reducer().quantile(values, [0.5])
    assert values == before


@pytest.mark.parametrize("method", ["max", "min", "mean", "median", "std"])
@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_the_reducers_return_python_numbers(
    make_reducer: Callable[[], BaseReducer], method: str, values: list[float]
) -> None:
    assert type(getattr(make_reducer(), method)(values)) in {int, float}


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(value=st.floats(min_value=-1e6, max_value=1e6), size=st.integers(1, 20))
def test_a_constant_sequence_has_that_value_as_max(
    make_reducer: Callable[[], BaseReducer], value: float, size: int
) -> None:
    assert make_reducer().max([value] * size) == value


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(value=st.floats(min_value=-1e6, max_value=1e6), size=st.integers(1, 20))
def test_a_constant_sequence_has_that_value_as_min(
    make_reducer: Callable[[], BaseReducer], value: float, size: int
) -> None:
    assert make_reducer().min([value] * size) == value


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(value=st.floats(min_value=-1e6, max_value=1e6), size=st.integers(1, 20))
def test_a_constant_sequence_has_that_value_as_mean(
    make_reducer: Callable[[], BaseReducer], value: float, size: int
) -> None:
    assert close(make_reducer().mean([value] * size), value)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(value=st.floats(min_value=-1e6, max_value=1e6), size=st.integers(1, 20))
def test_a_constant_sequence_has_that_value_as_median(
    make_reducer: Callable[[], BaseReducer], value: float, size: int
) -> None:
    assert close(make_reducer().median([value] * size), value)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(value=st.floats(min_value=-1e6, max_value=1e6), size=st.integers(2, 20))
def test_a_constant_sequence_has_a_zero_std(
    make_reducer: Callable[[], BaseReducer], value: float, size: int
) -> None:
    assert close(make_reducer().std([value] * size), 0.0)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_the_mean_is_between_the_min_and_the_max(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    assert reducer.min(values) - 1e-6 <= reducer.mean(values) <= reducer.max(values) + 1e-6


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=float_lists)
def test_the_results_are_python_numbers(
    make_reducer: Callable[[], BaseReducer], values: list[float]
) -> None:
    reducer = make_reducer()
    results = [
        reducer.max(values),
        reducer.min(values),
        reducer.median(values),
        reducer.mean(values),
    ]
    assert all(type(result) is float for result in results)


@pytest.mark.parametrize("make_reducer", REDUCERS)
@given(values=st.lists(st.integers(-1000, 1000), min_size=1, max_size=30))
def test_max_and_min_of_ints_are_ints(
    make_reducer: Callable[[], BaseReducer], values: list[int]
) -> None:
    reducer = make_reducer()
    assert [type(reducer.max(values)), type(reducer.min(values))] == [int, int]
