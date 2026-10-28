from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.reducer import EmptySequenceError, NumpyReducer
from coola.testing.marks import numpy_available
from coola.utils.imports import MissingPackageError, is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover

if TYPE_CHECKING:
    from collections.abc import Sequence


EMPTY_SEQUENCES = [[], (), np.array([])]


##################################
#     Tests for NumpyReducer     #
##################################


@numpy_available
def test_numpy_reducer_str() -> None:
    assert str(NumpyReducer()).startswith("NumpyReducer(")


@numpy_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3], np.array([-2, -1, 0, 1, 2])],
)
def test_numpy_reducer_max_int_type(values: Sequence[int | float]) -> None:
    val = NumpyReducer().max(values)
    assert isinstance(val, int)


@numpy_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3], np.array([-2, -1, 0, 1, 2])],
)
def test_numpy_reducer_max_int_value(values: Sequence[int | float]) -> None:
    val = NumpyReducer().max(values)
    assert val == 2


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [2.5],
        [2.5, -1.5, -2, -3],
        np.array([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_numpy_reducer_max_float_type(values: Sequence[int | float]) -> None:
    val = NumpyReducer().max(values)
    assert isinstance(val, float)


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [2.5],
        [2.5, -1.5, -2, -3],
        np.array([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_numpy_reducer_max_float_value(values: Sequence[int | float]) -> None:
    val = NumpyReducer().max(values)
    assert val == 2.5


@numpy_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_numpy_reducer_max_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the maximum because the sequence is empty"
    ):
        NumpyReducer().max(values)


@numpy_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], np.array([-2, -1, 0, 1, 2])]
)
def test_numpy_reducer_mean_int_type(values: Sequence[int | float]) -> None:
    val = NumpyReducer().mean(values)
    assert isinstance(val, float)


@numpy_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], np.array([-2, -1, 0, 1, 2])]
)
def test_numpy_reducer_mean_int_value(values: Sequence[int | float]) -> None:
    val = NumpyReducer().mean(values)
    assert val == 0.0


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        np.array([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_numpy_reducer_mean_float_type(values: Sequence[int | float]) -> None:
    val = NumpyReducer().mean(values)
    assert isinstance(val, float)


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        np.array([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_numpy_reducer_mean_float_value(values: Sequence[int | float]) -> None:
    val = NumpyReducer().mean(values)
    assert val == 0.5


@numpy_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_numpy_reducer_mean_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the mean because the sequence is empty"
    ):
        NumpyReducer().mean(values)


@numpy_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], np.array([-2, -1, 0, 1, 2])]
)
def test_numpy_reducer_median_int(values: Sequence[int | float]) -> None:
    assert NumpyReducer().median(values) == 0


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        np.array([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_numpy_reducer_median_float(values: Sequence[int | float]) -> None:
    assert NumpyReducer().median(values) == 0.5


@numpy_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_numpy_reducer_median_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the median because the sequence is empty"
    ):
        NumpyReducer().median(values)


@numpy_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3], np.array([-2, -1, 0, 1, 2])],
)
def test_numpy_reducer_min_int_type(values: Sequence[int | float]) -> None:
    val = NumpyReducer().min(values)
    assert isinstance(val, int)


@numpy_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3], np.array([-2, -1, 0, 1, 2])],
)
def test_numpy_reducer_min_int_value(values: Sequence[int | float]) -> None:
    val = NumpyReducer().min(values)
    assert val == -2


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [-2.5],
        [-2.5, 1.5, 2, 3],
        np.array([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_numpy_reducer_min_float_type(values: Sequence[int | float]) -> None:
    val = NumpyReducer().min(values)
    assert isinstance(val, float)


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [-2.5],
        [-2.5, 1.5, 2, 3],
        np.array([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_numpy_reducer_min_float_value(values: Sequence[int | float]) -> None:
    val = NumpyReducer().min(values)
    assert val == -2.5


@numpy_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_numpy_reducer_min_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the minimum because the sequence is empty"
    ):
        NumpyReducer().min(values)


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ],
)
def test_numpy_reducer_quantile_int(values: Sequence[int | float]) -> None:
    assert NumpyReducer().quantile(values, (0.2, 0.5, 0.9)) == [2, 5, 9]


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5),
        np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
    ],
)
def test_numpy_reducer_quantile_float(values: Sequence[int | float]) -> None:
    assert NumpyReducer().quantile(values, (0.0, 0.1, 0.4, 0.9)) == [0.5, 1.5, 4.5, 9.5]


@numpy_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_numpy_reducer_quantile_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the quantiles because the sequence is empty"
    ):
        NumpyReducer().quantile(values, [0.5])


@numpy_available
@pytest.mark.parametrize("values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0), np.array([2, 1, -2, 3, 0])])
def test_numpy_reducer_sort_int(values: Sequence[int | float]) -> None:
    assert NumpyReducer().sort(values) == [-2, 0, 1, 2, 3]


@numpy_available
@pytest.mark.parametrize(
    "values",
    [[2.5, 1.5, -2.5, 3.5, 0.5], (2.5, 1.5, -2.5, 3.5, 0.5), np.array([2.5, 1.5, -2.5, 3.5, 0.5])],
)
def test_numpy_reducer_sort_float(values: Sequence[int | float]) -> None:
    assert NumpyReducer().sort(values) == [-2.5, 0.5, 1.5, 2.5, 3.5]


@numpy_available
@pytest.mark.parametrize("values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0), np.array([2, 1, -2, 3, 0])])
def test_numpy_reducer_sort_descending(values: Sequence[int | float]) -> None:
    assert NumpyReducer().sort(values, descending=True) == [3, 2, 1, 0, -2]


@numpy_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_numpy_reducer_sort_empty(values: Sequence[int | float]) -> None:
    assert NumpyReducer().sort(values) == []


@numpy_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), np.array([-2, -1, 0, 1, 2])]
)
def test_numpy_reducer_std_int(values: Sequence[int | float]) -> None:
    assert math.isclose(NumpyReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        np.array([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_numpy_reducer_std_float(values: Sequence[int | float]) -> None:
    assert math.isclose(NumpyReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@numpy_available
@pytest.mark.parametrize("values", [[1], [1.0], np.array([1]), np.array([1.0])])
def test_numpy_reducer_std_one(values: Sequence[int | float]) -> None:
    assert math.isnan(NumpyReducer().std(values))


@numpy_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_numpy_reducer_std_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError,
        match=r"Cannot compute the standard deviation because the sequence is empty",
    ):
        NumpyReducer().std(values)


@numpy_available
def test_numpy_reducer_std_2d_array_raises() -> None:
    with pytest.raises(ValueError, match=r"NumpyReducer takes a 1-D input, got 2 dimension"):
        NumpyReducer().std(np.ones((1, 3)))


@numpy_available
def test_numpy_reducer_std_2d_array_one_element_raises() -> None:
    with pytest.raises(ValueError, match=r"NumpyReducer takes a 1-D input, got 2 dimension"):
        NumpyReducer().std(np.ones((1, 1)))


@numpy_available
def test_numpy_reducer_no_numpy() -> None:
    with (
        patch("coola.utils.imports.optional.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        NumpyReducer()


@numpy_available
def test_numpy_reducer_median_odd_integers_is_an_integer() -> None:
    assert isinstance(NumpyReducer().median([3, 1, 2]), int)


@numpy_available
def test_numpy_reducer_median_even_integers_is_a_float() -> None:
    assert NumpyReducer().median([1, 2, 3, 4]) == 2.5


@numpy_available
def test_numpy_reducer_median_does_not_overflow() -> None:
    assert NumpyReducer().median([1e308, 1e308]) == 1e308


@numpy_available
def test_numpy_reducer_median_infinite_values() -> None:
    assert NumpyReducer().median([float("inf"), float("inf")]) == float("inf")


@numpy_available
def test_numpy_reducer_max_generator_raises() -> None:
    with pytest.raises(TypeError, match="iterator"):
        NumpyReducer().max(i for i in range(3))


@numpy_available
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([2**70, 1], id="big-int"),
        pytest.param([1 + 2j, 3], id="complex"),
        pytest.param([None, 1], id="none"),
        pytest.param(["a", "b"], id="str"),
        pytest.param([Decimal(1)], id="decimal"),
    ],
)
def test_numpy_reducer_unsupported_values_raise(values: list) -> None:
    with pytest.raises(TypeError, match=r"NumpyReducer does not support .*use NativeReducer"):
        NumpyReducer().max(values)


@numpy_available
def test_numpy_reducer_ragged_sequences_raise() -> None:
    with pytest.raises(TypeError, match=r"NumpyReducer does not support nested sequences"):
        NumpyReducer().max([[1, 2], [3]])


@numpy_available
@pytest.mark.parametrize("values", [[[1, 2], [3, 4]], np.array(1), np.ones((2, 0))])
def test_numpy_reducer_not_1d_raises(values: object) -> None:
    with pytest.raises(ValueError, match=r"NumpyReducer takes a 1-D input"):
        NumpyReducer().max(values)


@numpy_available
def test_numpy_reducer_sort_2d_raises() -> None:
    with pytest.raises(ValueError, match=r"NumpyReducer takes a 1-D input, got 2 dimension"):
        NumpyReducer().sort([[2, 1], [4, 3]])


@numpy_available
def test_numpy_reducer_array_does_not_copy() -> None:
    array = np.array([3, 1, 2])
    assert NumpyReducer()._prepare(array) is array


@numpy_available
def test_numpy_reducer_result_is_python_scalar() -> None:
    assert type(NumpyReducer().max(np.array([1, 2]))) is int
