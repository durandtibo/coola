from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available
from coola.utils.array import to_array

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


if TYPE_CHECKING:
    from collections.abc import Sequence


##############################
#     Tests for to_array     #
##############################


@numpy_available
@pytest.mark.parametrize(
    "data",
    [
        np.array([3, 1, 2, 0, 1]),
        [3, 1, 2, 0, 1],
        (3, 1, 2, 0, 1),
    ],
)
def test_to_array_int(data: Sequence | np.ndarray) -> None:
    assert np.array_equal(to_array(data), np.array([3, 1, 2, 0, 1], dtype=int))


@numpy_available
@pytest.mark.parametrize(
    "data",
    [
        np.array([3.0, 1.0, 2.0, 0.0, 1.0]),
        [3.0, 1.0, 2.0, 0.0, 1.0],
        (3.0, 1.0, 2.0, 0.0, 1.0),
    ],
)
def test_to_array_float(data: Sequence | np.ndarray) -> None:
    assert np.array_equal(to_array(data), np.array([3.0, 1.0, 2.0, 0.0, 1.0], dtype=float))


@torch_available
def test_to_array_numpy() -> None:
    assert np.array_equal(
        to_array(torch.tensor([3, 1, 2, 0, 1])), np.array([3, 1, 2, 0, 1], dtype=int)
    )
