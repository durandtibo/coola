from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from coola.reducer import EmptySequenceError, NativeReducer, NumpyReducer, TorchReducer
from coola.testing.marks import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.reducer import BaseReducer


def _reducer_factories() -> list[Any]:
    r"""Return one ``pytest.param`` per ``BaseReducer`` implementation.

    Each param wraps a zero-argument factory that creates a fresh
    reducer instance. Reducers whose optional dependency is not
    installed (NumPy, PyTorch) are marked as skipped rather than
    omitted, so they are still visible (as skips) in `pytest
    -k`/reports.
    """
    return [
        pytest.param(NativeReducer, id="native"),
        pytest.param(NumpyReducer, id="numpy", marks=numpy_available),
        pytest.param(TorchReducer, id="torch", marks=torch_available),
    ]


@pytest.fixture(params=_reducer_factories())
def reducer(request: pytest.FixtureRequest) -> BaseReducer:
    return request.param()


def test_reducers_max_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the maximum
    value."""
    assert reducer.max([-2, -1, 0, 1, 2]) == 2


@pytest.mark.parametrize("values", [[], ()])
def test_reducers_max_empty_raises(reducer: BaseReducer, values: Sequence[int | float]) -> None:
    r"""All ``BaseReducer`` implementations must raise
    ``EmptySequenceError`` for an empty sequence."""
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        reducer.max(values)


def test_reducers_mean_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the mean
    value."""
    assert math.isclose(reducer.mean([-2, -1, 0, 1, 2]), 0.0, abs_tol=1e-6)


@pytest.mark.parametrize("values", [[], ()])
def test_reducers_mean_empty_raises(reducer: BaseReducer, values: Sequence[int | float]) -> None:
    r"""All ``BaseReducer`` implementations must raise
    ``EmptySequenceError`` for an empty sequence."""
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        reducer.mean(values)


def test_reducers_median_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the median
    value."""
    assert math.isclose(reducer.median([-2, -1, 0, 1, 2]), 0.0, abs_tol=1e-6)


@pytest.mark.parametrize("values", [[], ()])
def test_reducers_median_empty_raises(reducer: BaseReducer, values: Sequence[int | float]) -> None:
    r"""All ``BaseReducer`` implementations must raise
    ``EmptySequenceError`` for an empty sequence."""
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        reducer.median(values)


def test_reducers_min_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the minimum
    value."""
    assert reducer.min([-2, -1, 0, 1, 2]) == -2


@pytest.mark.parametrize("values", [[], ()])
def test_reducers_min_empty_raises(reducer: BaseReducer, values: Sequence[int | float]) -> None:
    r"""All ``BaseReducer`` implementations must raise
    ``EmptySequenceError`` for an empty sequence."""
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        reducer.min(values)


def test_reducers_quantile_consistent_length(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must return one value for
    each quantile."""
    assert len(reducer.quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0.2, 0.5, 0.9))) == 3


@pytest.mark.parametrize(("index", "expected"), [(0, 2.0), (1, 5.0), (2, 9.0)])
def test_reducers_quantile_consistent(reducer: BaseReducer, index: int, expected: float) -> None:
    r"""All ``BaseReducer`` implementations must agree on the
    quantiles."""
    quantiles = reducer.quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0.2, 0.5, 0.9))
    assert math.isclose(quantiles[index], expected, abs_tol=1e-6)


@pytest.mark.parametrize("values", [[], ()])
def test_reducers_quantile_empty_raises(
    reducer: BaseReducer, values: Sequence[int | float]
) -> None:
    r"""All ``BaseReducer`` implementations must raise
    ``EmptySequenceError`` for an empty sequence."""
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        reducer.quantile(values, (0.2, 0.5, 0.9))


def test_reducers_sort_ascending_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the sorted
    values in ascending order."""
    assert reducer.sort([2, 1, -2, 3, 0]) == [-2, 0, 1, 2, 3]


def test_reducers_sort_descending_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the sorted
    values in descending order."""
    assert reducer.sort([2, 1, -2, 3, 0], descending=True) == [3, 2, 1, 0, -2]


@pytest.mark.parametrize("values", [[], ()])
def test_reducers_sort_empty_returns_empty_list(
    reducer: BaseReducer, values: Sequence[int | float]
) -> None:
    r"""Unlike the other reduction methods, sorting an empty sequence
    must return an empty list instead of raising
    ``EmptySequenceError``, and all ``BaseReducer`` implementations
    must agree on this."""
    assert reducer.sort(values) == []


def test_reducers_std_multiple_values_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the standard
    deviation for a sequence with more than one value."""
    assert math.isclose(reducer.std([-2, -1, 0, 1, 2]), 1.5811388300841898, abs_tol=1e-6)


@pytest.mark.parametrize("values", [[1], [1.0], (1,), (1.0,)])
def test_reducers_std_one_returns_nan(reducer: BaseReducer, values: Sequence[int | float]) -> None:
    r"""All ``BaseReducer`` implementations must agree that the standard
    deviation of a single-element sequence is ``nan``, not an
    exception."""
    assert math.isnan(reducer.std(values))


@pytest.mark.parametrize("values", [[], ()])
def test_reducers_std_empty_raises(reducer: BaseReducer, values: Sequence[int | float]) -> None:
    r"""All ``BaseReducer`` implementations must raise
    ``EmptySequenceError`` for an empty sequence."""
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        reducer.std(values)


#####################################################
#     Agreement with the reference (native) one     #
#####################################################

# Datasets that exercise the cases where implementations tend to differ:
# even/odd lengths (median), unsorted input, duplicates, negative values,
# floats that are not exactly representable (precision) and large magnitudes.
DATASETS = [
    pytest.param([1, 2, 3, 4], id="even-length-ints"),
    pytest.param([4, 1, 3, 2], id="even-length-unsorted"),
    pytest.param([5, 1, 3], id="odd-length-unsorted"),
    pytest.param([1, 2], id="two-values"),
    pytest.param([7, 7, 7, 7], id="constant"),
    pytest.param([3, 1, 3, 1, 2, 2], id="duplicates"),
    pytest.param([-5, -1, -3, -2], id="negative-even"),
    pytest.param([0.1, 0.7, 0.3, 2.5, 1.1, 9.9], id="floats-even"),
    pytest.param([0.1, 0.7, 0.3, 2.5, 1.1], id="floats-odd"),
    pytest.param([1e-9, 2e-9, 3e-9, 4e-9], id="small-magnitude"),
    pytest.param([1e9, 2e9, 3e9, 4e9], id="large-magnitude"),
    pytest.param((1, 2, 3, 4), id="tuple"),
    pytest.param(
        [(i * 7919) % 101 / 10 for i in range(200)],
        id="pseudo-random-200",
    ),
]

QUANTILES = [
    (0.0, 0.25, 0.5, 0.75, 1.0),
    (0.1, 0.33, 0.9),
    (0.5,),
]


@pytest.fixture
def native() -> NativeReducer:
    return NativeReducer()


@pytest.mark.parametrize("values", DATASETS)
def test_reducers_max_agrees_with_native(
    reducer: BaseReducer, native: NativeReducer, values: Sequence[float]
) -> None:
    assert reducer.max(values) == native.max(values)


@pytest.mark.parametrize("values", DATASETS)
def test_reducers_min_agrees_with_native(
    reducer: BaseReducer, native: NativeReducer, values: Sequence[float]
) -> None:
    assert reducer.min(values) == native.min(values)


@pytest.mark.parametrize("values", DATASETS)
def test_reducers_mean_agrees_with_native(
    reducer: BaseReducer, native: NativeReducer, values: Sequence[float]
) -> None:
    assert reducer.mean(values) == pytest.approx(native.mean(values), rel=1e-12)


@pytest.mark.parametrize("values", DATASETS)
def test_reducers_median_agrees_with_native(
    reducer: BaseReducer, native: NativeReducer, values: Sequence[float]
) -> None:
    assert reducer.median(values) == pytest.approx(native.median(values), rel=1e-12)


@pytest.mark.parametrize("values", DATASETS)
def test_reducers_std_agrees_with_native(
    reducer: BaseReducer, native: NativeReducer, values: Sequence[float]
) -> None:
    assert reducer.std(values) == pytest.approx(native.std(values), rel=1e-9)


@pytest.mark.parametrize("values", DATASETS)
@pytest.mark.parametrize("quantiles", QUANTILES)
def test_reducers_quantile_agrees_with_native(
    reducer: BaseReducer,
    native: NativeReducer,
    values: Sequence[float],
    quantiles: Sequence[float],
) -> None:
    assert reducer.quantile(values, quantiles) == pytest.approx(
        native.quantile(values, quantiles), rel=1e-9, abs=1e-12
    )


@pytest.mark.parametrize("values", DATASETS)
@pytest.mark.parametrize("descending", [False, True])
def test_reducers_sort_agrees_with_native(
    reducer: BaseReducer,
    native: NativeReducer,
    values: Sequence[float],
    descending: bool,
) -> None:
    assert reducer.sort(values, descending=descending) == native.sort(values, descending=descending)


@pytest.mark.parametrize("values", [[1, 2, 3, 4], [4, 1, 3, 2], [-4, -1, -3, -2]])
def test_reducers_median_even_length_is_mean_of_middle_values(
    reducer: BaseReducer, values: Sequence[int]
) -> None:
    r"""The median of an even number of values must be the mean of the
    two middle values, not the lower (or upper) one."""
    middle = sorted(values)[len(values) // 2 - 1 : len(values) // 2 + 1]
    assert reducer.median(values) == sum(middle) / 2


def test_reducers_median_odd_length_is_a_value_of_the_sequence(reducer: BaseReducer) -> None:
    assert reducer.median([9, 1, 5, 3, 7]) == 5


@pytest.mark.parametrize("method", ["max", "min", "mean", "median", "std"])
@pytest.mark.parametrize("values", [[1, 2, 3, 4], [0.5, 1.5, 2.5]])
def test_reducers_return_python_numbers(
    reducer: BaseReducer, method: str, values: Sequence[float]
) -> None:
    r"""All reducers must return plain Python numbers (not NumPy or
    torch scalars) so the results are interchangeable."""
    assert type(getattr(reducer, method)(values)) in {int, float}


@pytest.mark.parametrize("values", [[1, 2, 3, 4], [0.5, 1.5, 2.5]])
def test_reducers_quantile_return_python_floats(
    reducer: BaseReducer, values: Sequence[float]
) -> None:
    assert all(type(q) is float for q in reducer.quantile(values, (0.25, 0.75)))


@pytest.mark.parametrize("values", [[1, 2, 3, 4], [0.5, 1.5, 2.5]])
def test_reducers_sort_return_python_numbers(reducer: BaseReducer, values: Sequence[float]) -> None:
    assert all(type(v) in {int, float} for v in reducer.sort(values))


def test_reducers_do_not_modify_the_input(reducer: BaseReducer) -> None:
    values = [3, 1, 2, 5, 4]
    reducer.sort(values)
    reducer.median(values)
    reducer.quantile(values, (0.5,))
    assert values == [3, 1, 2, 5, 4]


def test_reducers_quantile_extremes_are_min_and_max(reducer: BaseReducer) -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert reducer.quantile(values, (0.0, 1.0)) == [1.0, 4.0]


def test_reducers_quantile_median_matches_median(reducer: BaseReducer) -> None:
    values = [4.0, 1.0, 3.0, 2.0, 10.0, 6.0]
    assert reducer.quantile(values, (0.5,))[0] == pytest.approx(reducer.median(values))


def _array_and_tensor_reducers(dtype: str) -> list[tuple[BaseReducer, Any]]:
    data = [4, 1, 3, 2, 8, 6]
    array = np.array(data, dtype=dtype)
    tensor = torch.tensor(data, dtype=getattr(torch, dtype))
    return [(NativeReducer(), data), (NumpyReducer(), array), (TorchReducer(), tensor)]


@numpy_available
@torch_available
@pytest.mark.parametrize("method", ["max", "min", "mean", "median", "std"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_reducers_agree_on_array_inputs(dtype: str, method: str) -> None:
    r"""The NumPy reducer must agree with the native reducer on array
    inputs of various dtypes."""
    results = [getattr(r, method)(v) for r, v in _array_and_tensor_reducers(dtype)]
    assert results[1] == pytest.approx(results[0], rel=1e-6)


@numpy_available
@torch_available
@pytest.mark.parametrize("method", ["max", "min", "mean", "median", "std"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_reducers_agree_on_tensor_inputs(dtype: str, method: str) -> None:
    r"""The torch reducer must agree with the native reducer on tensor
    inputs of various dtypes."""
    results = [getattr(r, method)(v) for r, v in _array_and_tensor_reducers(dtype)]
    assert results[2] == pytest.approx(results[0], rel=1e-6)


@numpy_available
@torch_available
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_reducers_quantile_agree_on_array_inputs(dtype: str) -> None:
    quantiles = [r.quantile(v, (0.1, 0.5, 0.9)) for r, v in _array_and_tensor_reducers(dtype)]
    assert quantiles[1] == pytest.approx(quantiles[0], rel=1e-6)


@numpy_available
@torch_available
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_reducers_quantile_agree_on_tensor_inputs(dtype: str) -> None:
    quantiles = [r.quantile(v, (0.1, 0.5, 0.9)) for r, v in _array_and_tensor_reducers(dtype)]
    assert quantiles[2] == pytest.approx(quantiles[0], rel=1e-6)


@numpy_available
@torch_available
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_reducers_sort_agree_on_array_inputs(dtype: str) -> None:
    sorts = [r.sort(v) for r, v in _array_and_tensor_reducers(dtype)]
    assert sorts[0] == sorts[1]


@numpy_available
@torch_available
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_reducers_sort_agree_on_tensor_inputs(dtype: str) -> None:
    sorts = [r.sort(v) for r, v in _array_and_tensor_reducers(dtype)]
    assert sorts[0] == sorts[2]


NAN = float("nan")
NAN_VALUES = [
    pytest.param([NAN, 1.0, 2.0], id="first"),
    pytest.param([1.0, NAN, 2.0], id="middle"),
    pytest.param([1.0, 2.0, NAN], id="last"),
    pytest.param([NAN], id="only"),
    pytest.param([NAN, NAN, 1.0], id="several"),
]


@pytest.mark.parametrize("values", NAN_VALUES)
@pytest.mark.parametrize("method", ["max", "min", "mean", "median"])
def test_reducers_nan_propagates(reducer: BaseReducer, values: list[float], method: str) -> None:
    r"""A NaN gives NaN, whatever its position, for all the reducers."""
    assert math.isnan(getattr(reducer, method)(values))


@pytest.mark.parametrize("values", NAN_VALUES)
def test_reducers_nan_std(reducer: BaseReducer, values: list[float]) -> None:
    assert math.isnan(reducer.std(values))


@pytest.mark.parametrize("values", NAN_VALUES[:3])
def test_reducers_nan_quantile(reducer: BaseReducer, values: list[float]) -> None:
    assert all(math.isnan(q) for q in reducer.quantile(values, [0.1, 0.5, 0.9]))


@pytest.mark.parametrize("values", NAN_VALUES[:3])
def test_reducers_nan_sort_ascending_puts_nan_last_sorted_values(
    reducer: BaseReducer, values: list[float]
) -> None:
    result = reducer.sort(values)
    assert result[:2] == [1.0, 2.0]


@pytest.mark.parametrize("values", NAN_VALUES[:3])
def test_reducers_nan_sort_ascending_puts_nan_last_nan_last(
    reducer: BaseReducer, values: list[float]
) -> None:
    result = reducer.sort(values)
    assert math.isnan(result[2])


@pytest.mark.parametrize("values", NAN_VALUES[:3])
def test_reducers_nan_sort_descending_puts_nan_first_nan_first(
    reducer: BaseReducer, values: list[float]
) -> None:
    result = reducer.sort(values, descending=True)
    assert math.isnan(result[0])


@pytest.mark.parametrize("values", NAN_VALUES[:3])
def test_reducers_nan_sort_descending_puts_nan_first_sorted_values(
    reducer: BaseReducer, values: list[float]
) -> None:
    result = reducer.sort(values, descending=True)
    assert result[1:] == [2.0, 1.0]


@pytest.mark.parametrize("method", ["max", "min", "mean", "median"])
def test_reducers_inf_is_not_nan(reducer: BaseReducer, method: str) -> None:
    assert not math.isnan(getattr(reducer, method)([1.0, float("inf"), 2.0]))


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_reducers_std_infinite_value_is_nan(reducer: BaseReducer) -> None:
    assert math.isnan(reducer.std([1.0, 2.0, float("inf")]))


def test_reducers_empty_sequence_error_is_a_value_error(reducer: BaseReducer) -> None:
    with pytest.raises(ValueError, match=r"the sequence is empty"):
        reducer.max([])


@pytest.mark.parametrize("quantiles", [[-0.1], [1.1], [0.5, 2], [float("nan")]])
def test_reducers_quantile_invalid_quantiles_raise(
    reducer: BaseReducer, quantiles: list[float]
) -> None:
    with pytest.raises(ValueError, match=r"quantiles must be in \[0, 1\]"):
        reducer.quantile([1, 2, 3], quantiles)


def test_reducers_quantile_invalid_quantiles_are_checked_before_the_values(
    reducer: BaseReducer,
) -> None:
    with pytest.raises(ValueError, match=r"quantiles must be in \[0, 1\], got 2"):
        reducer.quantile([], [2])


def test_reducers_quantile_empty_quantiles_returns_empty_list(reducer: BaseReducer) -> None:
    assert reducer.quantile([1, 2, 3], []) == []


def test_reducers_sort_descending_is_keyword_only(reducer: BaseReducer) -> None:
    with pytest.raises(TypeError, match=r"positional argument"):
        reducer.sort([1, 2, 3], True)  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize("method", ["max", "mean", "median", "min", "std"])
def test_reducers_statistic_set_raises(reducer: BaseReducer, method: str) -> None:
    with pytest.raises(TypeError, match=r"of a set, pass a sequence of numbers"):
        getattr(reducer, method)({1, 2, 3})


def test_reducers_quantile_set_raises(reducer: BaseReducer) -> None:
    with pytest.raises(TypeError, match=r"of a set, pass a sequence of numbers"):
        reducer.quantile({1, 2, 3}, [0.5])


def test_reducers_sort_set_raises(reducer: BaseReducer) -> None:
    with pytest.raises(TypeError, match=r"of a set, pass a sequence of numbers"):
        reducer.sort({1, 2, 3})


@pytest.mark.parametrize("values", ["abc", b"abc"])
def test_reducers_max_str_or_bytes_raises(reducer: BaseReducer, values: str | bytes) -> None:
    with pytest.raises(TypeError, match=r"pass a sequence of numbers"):
        reducer.max(values)


def test_reducers_sort_generator_raises(reducer: BaseReducer) -> None:
    with pytest.raises(TypeError, match=r"of an iterator, convert it to a list first"):
        reducer.sort(x for x in [3, 1, 2])


def test_reducers_quantile_bool_values(reducer: BaseReducer) -> None:
    assert reducer.quantile([True, False], [0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]


def test_reducers_input_is_not_modified(reducer: BaseReducer) -> None:
    values = [3, 1, 2]
    reducer.sort(values)
    assert values == [3, 1, 2]
