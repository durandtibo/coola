from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.reducer import BaseReducer, EmptySequenceError

if TYPE_CHECKING:
    from collections.abc import Sequence


class RecordingReducer(BaseReducer[list]):
    r"""Implement a minimal backend that records the calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def _prepare(self, values: list) -> list:
        self.calls.append(("prepare", values))
        return values

    def _max(self, values: list) -> int | float:
        self.calls.append(("max", values))
        return 1

    def _mean(self, values: list) -> float:
        self.calls.append(("mean", values))
        return 1.0

    def _median(self, values: list) -> int | float:
        self.calls.append(("median", values))
        return 1

    def _min(self, values: list) -> int | float:
        self.calls.append(("min", values))
        return 1

    def _quantile(self, values: list, quantiles: Sequence[float]) -> list[float]:
        self.calls.append(("quantile", values))
        return [float(q) for q in quantiles]

    def _sort(self, values: list, *, descending: bool) -> list[int | float]:
        self.calls.append(("sort", descending))
        return list(values)

    def _std(self, values: list) -> float:
        self.calls.append(("std", values))
        return 1.0


@pytest.fixture
def reducer() -> RecordingReducer:
    return RecordingReducer()


METHODS = ["max", "mean", "median", "min", "std"]


@pytest.mark.parametrize("method", METHODS)
def test_base_reducer_statistic_calls_prepare_then_the_computation(
    reducer: RecordingReducer, method: str
) -> None:
    getattr(reducer, method)([1, 2])
    assert reducer.calls == [("prepare", [1, 2]), (method, [1, 2])]


@pytest.mark.parametrize("method", METHODS)
def test_base_reducer_statistic_empty_input_does_not_call_the_computation(
    reducer: RecordingReducer, method: str
) -> None:
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        getattr(reducer, method)([])
    assert reducer.calls == [("prepare", [])]


@pytest.mark.parametrize("method", METHODS)
def test_base_reducer_statistic_iterator_is_rejected_before_prepare(
    reducer: RecordingReducer, method: str
) -> None:
    with pytest.raises(TypeError, match=r"convert it to a list first"):
        getattr(reducer, method)(iter([1, 2]))
    assert reducer.calls == []


def test_base_reducer_quantile_calls_prepare_then_the_computation(
    reducer: RecordingReducer,
) -> None:
    reducer.quantile([1, 2], [0.5])
    assert reducer.calls == [("prepare", [1, 2]), ("quantile", [1, 2])]


def test_base_reducer_quantile_validates_the_quantiles_before_prepare(
    reducer: RecordingReducer,
) -> None:
    with pytest.raises(ValueError, match=r"quantiles must be in \[0, 1\], got 2"):
        reducer.quantile([1, 2], [2])
    assert reducer.calls == []


def test_base_reducer_quantile_empty_input_does_not_call_the_computation(
    reducer: RecordingReducer,
) -> None:
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        reducer.quantile([], [0.5])
    assert reducer.calls == [("prepare", [])]


def test_base_reducer_sort_empty_input_calls_the_computation(reducer: RecordingReducer) -> None:
    assert reducer.sort([]) == []
    assert reducer.calls == [("prepare", []), ("sort", False)]


def test_base_reducer_sort_descending(reducer: RecordingReducer) -> None:
    reducer.sort([1, 2], descending=True)
    assert reducer.calls == [("prepare", [1, 2]), ("sort", True)]


def test_base_reducer_prepare_default_returns_the_values(reducer: RecordingReducer) -> None:
    values = [1, 2]
    assert BaseReducer._prepare(reducer, values) is values


def test_base_reducer_quantile_accepts_the_bounds(reducer: RecordingReducer) -> None:
    assert reducer.quantile([1, 2], [0, 1]) == [0.0, 1.0]


def test_base_reducer_sort_iterator_is_rejected_before_prepare(reducer: RecordingReducer) -> None:
    with pytest.raises(TypeError, match=r"of an iterator, convert it to a list first"):
        reducer.sort(iter([1, 2]))  # ty: ignore[invalid-argument-type]
    assert reducer.calls == []


def test_base_reducer_set_is_rejected_before_prepare(reducer: RecordingReducer) -> None:
    with pytest.raises(TypeError, match=r"of a set, pass a sequence of numbers"):
        reducer.max({1, 2})  # ty: ignore[invalid-argument-type]
    assert reducer.calls == []


def test_base_reducer_raise_unsupported_names_the_backend_and_the_hint(
    reducer: RecordingReducer,
) -> None:
    with pytest.raises(TypeError, match=r"^RecordingReducer does not support bytes; use another$"):
        reducer._raise_unsupported("bytes", "use another")


def test_base_reducer_raise_unsupported_without_hint(reducer: RecordingReducer) -> None:
    with pytest.raises(TypeError, match=r"^RecordingReducer does not support bytes$"):
        reducer._raise_unsupported("bytes")


def test_base_reducer_raise_ndim_names_the_backend(reducer: RecordingReducer) -> None:
    with pytest.raises(ValueError, match=r"^RecordingReducer takes a 1-D input, got 2 dimension"):
        reducer._raise_ndim(2)
