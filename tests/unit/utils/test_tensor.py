from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available
from coola.utils.tensor import (
    get_available_devices,
    is_cuda_available,
    is_mps_available,
    to_tensor,
)

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


@pytest.fixture(autouse=True)
def _reset() -> None:
    get_available_devices.cache_clear()
    is_cuda_available.cache_clear()
    is_mps_available.cache_clear()


###########################################
#     Tests for get_available_devices     #
###########################################


@torch_available
@patch("torch.cuda.is_available", lambda: False)
@patch("coola.utils.tensor.is_mps_available", lambda: False)
def test_get_available_devices_cpu() -> None:
    assert get_available_devices() == ("cpu",)


@torch_available
@patch("torch.cuda.is_available", lambda: True)
@patch("torch.cuda.device_count", lambda: 1)
@patch("coola.utils.tensor.is_mps_available", lambda: False)
def test_get_available_devices_cpu_and_gpu() -> None:
    assert get_available_devices() == ("cpu", "cuda:0")


@torch_available
@patch("torch.cuda.is_available", lambda: False)
@patch("coola.utils.tensor.is_mps_available", lambda: True)
def test_get_available_devices_cpu_and_mps() -> None:
    assert get_available_devices() == ("cpu", "mps:0")


@torch_available
@patch("torch.cuda.is_available", lambda: True)
@patch("torch.cuda.device_count", lambda: 1)
@patch("coola.utils.tensor.is_mps_available", lambda: True)
def test_get_available_devices_cpu_and_gpu_and_mps() -> None:
    assert get_available_devices() == ("cpu", "cuda:0", "mps:0")


#######################################
#     Tests for is_cuda_available     #
#######################################


@torch_available
def test_is_cuda_available() -> None:
    assert isinstance(is_cuda_available(), bool)


@torch_available
@patch("torch.cuda.is_available", lambda: True)
def test_is_cuda_available_true() -> None:
    assert is_cuda_available()


@torch_available
@patch("torch.cuda.is_available", lambda: False)
def test_is_cuda_available_false() -> None:
    assert not is_cuda_available()


@patch("coola.utils.tensor.is_torch_available", lambda: False)
def test_is_cuda_available_no_torch() -> None:
    assert not is_cuda_available()


######################################
#     Tests for is_mps_available     #
######################################


@torch_available
def test_is_mps_available() -> None:
    assert isinstance(is_mps_available(), bool)


@torch_available
@patch("coola.utils.tensor.is_torch_available", lambda: True)
def test_is_mps_available_with_mps() -> None:
    with patch("coola.utils.tensor.torch.ones", Mock(return_value=torch.ones(1))):
        assert is_mps_available()


@torch_available
@patch("coola.utils.tensor.is_torch_available", lambda: True)
def test_is_mps_available_without_mps() -> None:
    with patch("coola.utils.tensor.torch.ones", Mock(side_effect=RuntimeError)):
        assert not is_mps_available()


@patch("coola.utils.tensor.is_torch_available", lambda: False)
def test_is_mps_available_no_torch() -> None:
    assert not is_mps_available()


###############################
#     Tests for to_tensor     #
###############################


@torch_available
@pytest.mark.parametrize(
    "data",
    [
        torch.tensor([3, 1, 2, 0, 1]),
        [3, 1, 2, 0, 1],
        (3, 1, 2, 0, 1),
    ],
)
def test_to_tensor_long(data: Sequence | torch.Tensor) -> None:
    assert to_tensor(data).equal(torch.tensor([3, 1, 2, 0, 1], dtype=torch.long))


@torch_available
@pytest.mark.parametrize(
    "data",
    [
        torch.tensor([3.0, 1.0, 2.0, 0.0, 1.0]),
        [3.0, 1.0, 2.0, 0.0, 1.0],
        (3.0, 1.0, 2.0, 0.0, 1.0),
    ],
)
def test_to_tensor_float(data: Sequence | torch.Tensor) -> None:
    # A RuntimeError can be raised if torch and numpy are not compatible
    with suppress(RuntimeError):
        assert to_tensor(data).equal(torch.tensor([3.0, 1.0, 2.0, 0.0, 1.0], dtype=torch.float))


@numpy_available
@torch_available
def test_to_tensor_numpy() -> None:
    # A RuntimeError can be raised if torch and numpy are not compatible
    with suppress(RuntimeError):
        assert to_tensor(np.array([3, 1, 2, 0, 1])).equal(
            torch.tensor([3, 1, 2, 0, 1], dtype=torch.long)
        )
