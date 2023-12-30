from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola import objects_are_allclose, objects_are_equal
from coola.testing import numpy_available
from coola.utils.imports import is_numpy_available
from coola.utils.stats import quantile

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

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
        show_difference=True,
    )


def test_quantile_21_decile() -> None:
    assert objects_are_equal(
        quantile(list(range(21)), (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    )


@numpy_available
@pytest.mark.parametrize(
    "array",
    [np.random.randn(5), np.random.randn(100), np.random.randn(1000), np.random.rand(1000)],
)
@pytest.mark.parametrize("num_quantiles", [0, 1, 2, 4, 8])
def test_quantile_numpy(array: np.ndarray, num_quantiles: int) -> None:
    q: list[float] = np.random.rand(num_quantiles).tolist()
    assert objects_are_allclose(
        quantile(array.tolist(), q),
        np.quantile(array, q=q).tolist(),
        show_difference=True,
        atol=1e-6,
    )
