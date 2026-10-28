from __future__ import annotations

import copy
from collections import deque
from typing import Any
from unittest.mock import Mock

import pytest

from coola.equality import objects_are_equal
from coola.testing.marks import numpy_available, torch_available
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


NUMPY_DATA = [
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
    pytest.param([np.ones((2, 3)), np.zeros((2, 3)), 42, "abc"], id="list with string"),
]
TORCH_DATA = [
    pytest.param(torch.ones(2, 3), id="torch Tensor"),
    pytest.param(
        {"key1": torch.ones(2, 3), "key2": torch.ones(2, 3), "key3": 42},
        id="dict with tensor like objects",
    ),
    pytest.param([torch.ones(2, 3), torch.zeros(2, 3), 42], id="list with tensor like objects"),
]
# numpy and torch fail on these data, but ``objects_are_equal`` does not.
NUMPY_UNSUPPORTED_DATA = [
    pytest.param(
        deque([np.ones((2, 3)), np.zeros((2, 3)), 42]), id="deque with array like objects"
    ),
]
TORCH_UNSUPPORTED_DATA = [
    pytest.param(
        {"key1": torch.ones(2, 3), "key2": torch.ones(2, 3), "key3": 42, "key4": "abc"},
        id="dict with string",
    ),
    pytest.param([torch.ones(2, 3), torch.zeros(2, 3), 42, "abc"], id="list with string"),
]


@numpy_available
@pytest.mark.parametrize("data", NUMPY_DATA)
def test_numpy_assert_equal(data: Any) -> None:
    np.testing.assert_equal(data, copy.deepcopy(data))


@numpy_available
@pytest.mark.parametrize("data", NUMPY_DATA + NUMPY_UNSUPPORTED_DATA)
def test_objects_are_equal_numpy_data(data: Any) -> None:
    assert objects_are_equal(data, copy.deepcopy(data))


@numpy_available
@pytest.mark.xfail(strict=True, raises=ValueError, reason="numpy cannot compare a deque of arrays")
@pytest.mark.parametrize("data", NUMPY_UNSUPPORTED_DATA)
def test_numpy_assert_equal_unsupported(data: Any) -> None:
    np.testing.assert_equal(data, copy.deepcopy(data))


@torch_available
@pytest.mark.parametrize("data", TORCH_DATA)
def test_torch_assert_close(data: Any) -> None:
    torch.testing.assert_close(data, copy.deepcopy(data))


@torch_available
@pytest.mark.parametrize("data", TORCH_DATA + TORCH_UNSUPPORTED_DATA)
def test_objects_are_equal_torch_data(data: Any) -> None:
    assert objects_are_equal(data, copy.deepcopy(data))


@torch_available
@pytest.mark.xfail(strict=True, raises=TypeError, reason="torch cannot compare strings")
@pytest.mark.parametrize("data", TORCH_UNSUPPORTED_DATA)
def test_torch_assert_close_unsupported(data: Any) -> None:
    torch.testing.assert_close(data, copy.deepcopy(data))
