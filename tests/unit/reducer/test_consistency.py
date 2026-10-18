from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.reducer import EmptySequenceError, NativeReducer, NumpyReducer, TorchReducer
from coola.testing.fixtures import numpy_available, torch_available
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


def _reducer_factories() -> list[pytest.mark.ParameterSet]:
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


def test_reducers_quantile_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the
    quantiles."""
    quantiles = reducer.quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0.2, 0.5, 0.9))
    assert len(quantiles) == 3
    for value, expected in zip(quantiles, [2.0, 5.0, 9.0], strict=True):
        assert math.isclose(value, expected, abs_tol=1e-6)


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


####################################################
#     Agreement with the reference (native) one    #
####################################################

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


@pytest.mark.parametrize("values", [[1, 2, 3, 4], [0.5, 1.5, 2.5]])
def test_reducers_return_python_numbers(reducer: BaseReducer, values: Sequence[float]) -> None:
    r"""All reducers must return plain Python numbers (not NumPy or
    torch scalars) so the results are interchangeable."""
    for result in (
        reducer.max(values),
        reducer.min(values),
        reducer.mean(values),
        reducer.median(values),
        reducer.std(values),
    ):
        assert type(result) in {int, float}
    assert all(type(q) is float for q in reducer.quantile(values, (0.25, 0.75)))
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


@numpy_available
@torch_available
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_reducers_agree_on_array_and_tensor_inputs(dtype: str) -> None:
    r"""The NumPy and torch reducers must agree with each other, and
    with the native reducer, on array/tensor inputs of various
    dtypes."""
    data = [4, 1, 3, 2, 8, 6]
    array = np.array(data, dtype=dtype)
    tensor = torch.tensor(data, dtype=getattr(torch, dtype))
    reducers = [(NativeReducer(), data), (NumpyReducer(), array), (TorchReducer(), tensor)]
    tol = {"rel": 1e-6}
    for method in ("max", "min", "mean", "median", "std"):
        results = [getattr(reducer, method)(values) for reducer, values in reducers]
        assert results[1] == pytest.approx(results[0], **tol), method
        assert results[2] == pytest.approx(results[0], **tol), method
    quantiles = [reducer.quantile(values, (0.1, 0.5, 0.9)) for reducer, values in reducers]
    assert quantiles[1] == pytest.approx(quantiles[0], **tol)
    assert quantiles[2] == pytest.approx(quantiles[0], **tol)
    sorts = [reducer.sort(values) for reducer, values in reducers]
    assert sorts[0] == sorts[1] == sorts[2]
