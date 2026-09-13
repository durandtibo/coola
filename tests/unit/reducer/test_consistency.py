from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from coola.reducer import EmptySequenceError, NativeReducer, NumpyReducer, TorchReducer
from coola.testing.fixtures import numpy_available, torch_available

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
