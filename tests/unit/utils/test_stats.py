from __future__ import annotations

import math
from unittest.mock import Mock

import pytest

from coola.equality import objects_are_allclose, objects_are_equal
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available
from coola.utils.stats import quantile

if is_numpy_available():
    import numpy as np
    from numpy.random import default_rng
else:
    np = Mock()
    default_rng = Mock()

##############################
#     Tests for quantile     #
##############################


def test_quantile_empty() -> None:
    assert objects_are_equal(quantile([], []), [])


def test_quantile_11_decile() -> None:
    assert objects_are_equal(
        quantile(list(range(11)), (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )


def test_quantile_11_quantile() -> None:
    assert objects_are_allclose(
        quantile(list(range(11)), (0.1, 0.25, 0.5, 0.75, 0.9)),
        [1.0, 2.5, 5.0, 7.5, 9.0],
    )


def test_quantile_21_decile() -> None:
    assert objects_are_equal(
        quantile(list(range(21)), (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    )


@numpy_available
@pytest.mark.parametrize(
    "array",
    [
        default_rng(0).standard_normal(5),
        default_rng(1).standard_normal(100),
        default_rng(2).standard_normal(1000),
        default_rng(3).random(1000),
    ],
)
@pytest.mark.parametrize("num_quantiles", [0, 1, 2, 4, 8])
def test_quantile_numpy(array: np.ndarray, num_quantiles: int) -> None:
    q: list[float] = default_rng(4).random(num_quantiles).tolist()
    assert objects_are_allclose(
        quantile(array.tolist(), q),
        np.quantile(array, q=q).tolist(),
        atol=1e-6,
    )


def test_quantile_single_value() -> None:
    """Test quantile with single-value sequence."""
    assert objects_are_equal(quantile([5], [0.0, 0.5, 1.0]), [5.0, 5.0, 5.0])


def test_quantile_negative_values() -> None:
    """Test quantile with negative values."""
    assert objects_are_allclose(
        quantile([-5, -3, -1], [0.0, 0.5, 1.0]),
        [-5.0, -3.0, -1.0],
    )


def test_quantile_mixed_positive_negative() -> None:
    """Test quantile with mixed positive and negative values."""
    assert objects_are_allclose(
        quantile([-10, -5, 0, 5, 10], [0.0, 0.25, 0.5, 0.75, 1.0]),
        [-10.0, -5.0, 0.0, 5.0, 10.0],
    )


def test_quantile_identical_values() -> None:
    """Test quantile with all identical values."""
    assert objects_are_equal(quantile([3, 3, 3, 3, 3], [0.0, 0.5, 1.0]), [3.0, 3.0, 3.0])


def test_quantile_single_quantile() -> None:
    """Test quantile with single quantile value."""
    assert objects_are_allclose(quantile([1, 2, 3, 4, 5], [0.5]), [3.0])


def test_quantile_empty_values_raises_error() -> None:
    """Test that empty values with non-empty quantiles raises
    ValueError."""
    with pytest.raises(ValueError, match="values must not be empty"):
        quantile([], [0.5])


@pytest.mark.parametrize("q", [-0.1, 1.1, float("nan"), float("inf")])
def test_quantile_invalid_quantile_raises(q: float) -> None:
    with pytest.raises(ValueError, match=r"quantiles must be in the range \[0.0, 1.0\]"):
        quantile([1, 2, 3], [0.5, q])


def test_quantile_boundaries() -> None:
    assert quantile([1, 2, 3], [0.0, 1.0]) == [1, 3]


NAN_VALUES = [
    [float("nan"), 1.0, 2.0],
    [1.0, float("nan"), 2.0],
    [1.0, 2.0, float("nan")],
    [float("nan")],
]


@pytest.mark.parametrize("values", NAN_VALUES)
def test_quantile_nan_gives_a_value_for_each_quantile(values: list[float]) -> None:
    assert len(quantile(values, (0.0, 0.5, 1.0))) == 3


@pytest.mark.parametrize("values", NAN_VALUES)
def test_quantile_nan_gives_nan_whatever_the_position(values: list[float]) -> None:
    assert all(math.isnan(value) for value in quantile(values, (0.0, 0.5, 1.0)))


def test_quantile_nan_with_no_quantile() -> None:
    assert quantile([1.0, float("nan")], ()) == []


def test_quantile_nan_invalid_quantile_still_raises() -> None:
    with pytest.raises(ValueError, match="quantiles must be in the range"):
        quantile([float("nan")], (2.0,))


def test_quantile_infinite_values_are_not_nan() -> None:
    assert quantile([1.0, float("inf")], (0.0,)) == [1.0]
