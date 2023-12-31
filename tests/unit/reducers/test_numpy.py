from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.reducers import EmptySequenceError, NumpyReducer, ReducerRegistry
from coola.testing import numpy_available

if TYPE_CHECKING:
    from collections.abc import Sequence


@numpy_available
def test_reducer_registry_available_reducers() -> None:
    assert isinstance(ReducerRegistry.registry["numpy"], NumpyReducer)


##################################
#     Tests for NumpyReducer     #
##################################


@numpy_available
def test_numpy_reducer_str() -> None:
    assert str(NumpyReducer()).startswith("NumpyReducer(")


@numpy_available
@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3]])
def test_numpy_reducer_max_int(values: Sequence[int | float]) -> None:
    val = NumpyReducer().max(values)
    assert isinstance(val, int)
    assert val == 2


@numpy_available
@pytest.mark.parametrize(
    "values", [[-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [2.5], [2.5, -1.5, -2, -3]]
)
def test_numpy_reducer_max_float(values: Sequence[int | float]) -> None:
    val = NumpyReducer().max(values)
    assert isinstance(val, float)
    assert val == 2.5


@numpy_available
@pytest.mark.parametrize("values", [[], ()])
def test_numpy_reducer_max_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match="Cannot compute the maximum because the summary is empty"
    ):
        NumpyReducer().max([])


@numpy_available
@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]])
def test_numpy_reducer_mean_int(values: Sequence[int | float]) -> None:
    val = NumpyReducer().mean(values)
    assert isinstance(val, float)
    assert val == 0.0


@numpy_available
@pytest.mark.parametrize(
    "values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]]
)
def test_numpy_reducer_mean_float(values: Sequence[int | float]) -> None:
    val = NumpyReducer().mean(values)
    assert isinstance(val, float)
    assert val == 0.5


@numpy_available
@pytest.mark.parametrize("values", [[], ()])
def test_numpy_reducer_mean_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match="Cannot compute the mean because the summary is empty"
    ):
        NumpyReducer().mean([])


@numpy_available
@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0]])
def test_numpy_reducer_median_int(values: Sequence[int | float]) -> None:
    assert NumpyReducer().median(values) == 0


@numpy_available
@pytest.mark.parametrize(
    "values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5), [0.5]]
)
def test_numpy_reducer_median_float(values: Sequence[int | float]) -> None:
    assert NumpyReducer().median(values) == 0.5


@numpy_available
@pytest.mark.parametrize("values", [[], ()])
def test_numpy_reducer_median_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match="Cannot compute the median because the summary is empty"
    ):
        NumpyReducer().median([])


@numpy_available
@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3]])
def test_numpy_reducer_min_int(values: Sequence[int | float]) -> None:
    val = NumpyReducer().min(values)
    assert isinstance(val, int)
    assert val == -2


@numpy_available
@pytest.mark.parametrize(
    "values", [[-2.5, -1.5, 0.5, 1.5, 2.5], (-2.5, -1.5, 0.5, 1.5, 2.5), [-2.5], [-2.5, 1.5, 2, 3]]
)
def test_numpy_reducer_min_float(values: Sequence[int | float]) -> None:
    val = NumpyReducer().min(values)
    assert isinstance(val, float)
    assert val == -2.5


@numpy_available
@pytest.mark.parametrize("values", [[], ()])
def test_numpy_reducer_min_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match="Cannot compute the minimum because the summary is empty"
    ):
        NumpyReducer().min([])


@numpy_available
@pytest.mark.parametrize(
    "values", [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
)
def test_numpy_reducer_quantile_int(values: Sequence[int | float]) -> None:
    assert NumpyReducer().quantile(values, (0.2, 0.5, 0.9)) == [2, 5, 9]


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5),
    ],
)
def test_numpy_reducer_quantile_float(values: Sequence[int | float]) -> None:
    assert NumpyReducer().quantile(values, (0.0, 0.1, 0.4, 0.9)) == [0.5, 1.5, 4.5, 9.5]


@numpy_available
@pytest.mark.parametrize("values", [[], ()])
def test_numpy_reducer_quantile_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match="Cannot compute the quantiles because the summary is empty"
    ):
        NumpyReducer().quantile([], [0.5])


@numpy_available
@pytest.mark.parametrize("values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0)])
def test_numpy_reducer_sort_int(values: Sequence[int | float]) -> None:
    assert NumpyReducer().sort(values) == [-2, 0, 1, 2, 3]


@numpy_available
@pytest.mark.parametrize("values", [[2.5, 1.5, -2.5, 3.5, 0.5], (2.5, 1.5, -2.5, 3.5, 0.5)])
def test_numpy_reducer_sort_float(values: Sequence[int | float]) -> None:
    assert NumpyReducer().sort(values) == [-2.5, 0.5, 1.5, 2.5, 3.5]


@numpy_available
@pytest.mark.parametrize("values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0)])
def test_numpy_reducer_sort_descending(values: Sequence[int | float]) -> None:
    assert NumpyReducer().sort(values, descending=True) == [3, 2, 1, 0, -2]


@numpy_available
@pytest.mark.parametrize("values", [[], ()])
def test_numpy_reducer_sort_empty(values: Sequence[int | float]) -> None:
    assert NumpyReducer().sort([]) == []


@numpy_available
@pytest.mark.parametrize("values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2)])
def test_numpy_reducer_std_int(values: Sequence[int | float]) -> None:
    assert math.isclose(NumpyReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@numpy_available
@pytest.mark.parametrize("values", [[-1.5, -0.5, 0.5, 1.5, 2.5], (-1.5, -0.5, 0.5, 1.5, 2.5)])
def test_numpy_reducer_std_float(values: Sequence[int | float]) -> None:
    assert math.isclose(NumpyReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@numpy_available
@pytest.mark.parametrize("values", [[1], [1.0]])
def test_numpy_reducer_std_one(values: Sequence[int | float]) -> None:
    assert math.isnan(NumpyReducer().std(values))


@numpy_available
@pytest.mark.parametrize("values", [[], ()])
def test_numpy_reducer_std_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError,
        match="Cannot compute the standard deviation because the summary is empty",
    ):
        NumpyReducer().std([])


@numpy_available
def test_numpy_reducer_no_numpy() -> None:
    with patch(
        "coola.utils.imports.is_numpy_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`numpy` package is required but not installed."):
        NumpyReducer()
