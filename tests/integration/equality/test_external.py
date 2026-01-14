from __future__ import annotations

import copy
from collections import deque
from typing import Any
from unittest.mock import Mock

import pytest

from coola.equality import objects_are_equal
from coola.testing.fixtures import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch
else:
    torch = Mock()


#############################
#     External packages     #
#############################

# Verify the behavior is consistent with existing libraries.


@numpy_available
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(np.ones((2, 3)), id="numpy ndarray"),
        pytest.param(
            {"key1": np.ones((2, 3)), "key2": np.ones((2, 3)), "key3": 42},
            id="dict with array like objects",
        ),
        pytest.param(
            {"key1": np.ones((2, 3)), "key2": np.ones((2, 3)), "key3": 42, "key4": "abc"},
            id="dict with string",
        ),
        pytest.param([np.ones((2, 3)), np.zeros((2, 3)), 42], id="list with array like objects"),
        pytest.param(
            [np.ones((2, 3)), np.zeros((2, 3)), 42, "abc"],
            id="list with string",
        ),
        pytest.param(
            deque([np.ones((2, 3)), np.zeros((2, 3)), 42]),
            marks=pytest.mark.xfail,
            id="deque with array like objects",
        ),
    ],
)
def test_numpy_assert_equal(data: Any) -> None:
    np.testing.assert_equal(data, copy.deepcopy(data))
    assert objects_are_equal(data, copy.deepcopy(data))


@torch_available
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(torch.ones(2, 3), id="torch Tensor"),
        pytest.param(
            {"key1": torch.ones(2, 3), "key2": torch.ones(2, 3), "key3": 42},
            id="dict with tensor like objects",
        ),
        pytest.param(
            {"key1": torch.ones(2, 3), "key2": torch.ones(2, 3), "key3": 42, "key4": "abc"},
            marks=pytest.mark.xfail,
            id="dict with string",
        ),
        pytest.param([torch.ones(2, 3), torch.zeros(2, 3), 42], id="list with tensor like objects"),
        pytest.param(
            [torch.ones(2, 3), torch.zeros(2, 3), 42, "abc"],
            marks=pytest.mark.xfail,
            id="list with string",
        ),
    ],
)
def test_torch_assert_close(data: Any) -> None:
    torch.testing.assert_close(data, copy.deepcopy(data))
    assert objects_are_equal(data, copy.deepcopy(data))
