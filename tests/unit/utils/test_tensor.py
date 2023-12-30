from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.testing import torch_available
from coola.utils.tensor import (
    get_available_devices,
    is_cuda_available,
    is_mps_available,
    torch,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    get_available_devices.cache_clear()
    is_cuda_available.cache_clear()
    is_mps_available.cache_clear()


###########################################
#     Tests for get_available_devices     #
###########################################


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: False)
@patch("coola.utils.tensor.is_mps_available", lambda *args, **kwargs: False)
def test_get_available_devices_cpu() -> None:
    assert get_available_devices() == ("cpu",)


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
@patch("torch.cuda.device_count", lambda *args, **kwargs: 1)
@patch("coola.utils.tensor.is_mps_available", lambda *args, **kwargs: False)
def test_get_available_devices_cpu_and_gpu() -> None:
    assert get_available_devices() == ("cpu", "cuda:0")


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: False)
@patch("coola.utils.tensor.is_mps_available", lambda *args, **kwargs: True)
def test_get_available_devices_cpu_and_mps() -> None:
    assert get_available_devices() == ("cpu", "mps:0")


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
@patch("torch.cuda.device_count", lambda *args, **kwargs: 1)
@patch("coola.utils.tensor.is_mps_available", lambda *args, **kwargs: True)
def test_get_available_devices_cpu_and_gpu_and_mps() -> None:
    assert get_available_devices() == ("cpu", "cuda:0", "mps:0")


#######################################
#     Tests for is_cuda_available     #
#######################################


@torch_available
def test_is_cuda_available() -> None:
    assert isinstance(is_cuda_available(), bool)


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
def test_is_cuda_available_true() -> None:
    assert is_cuda_available()


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: False)
def test_is_cuda_available_false() -> None:
    assert not is_cuda_available()


@patch("coola.utils.tensor.is_torch_available", lambda *args, **kwargs: False)
def test_is_cuda_available_no_torch() -> None:
    assert not is_cuda_available()


######################################
#     Tests for is_mps_available     #
######################################


@torch_available
def test_is_mps_available() -> None:
    assert isinstance(is_mps_available(), bool)


@torch_available
@patch("coola.utils.tensor.is_torch_available", lambda *args, **kwargs: True)
def test_is_mps_available_with_mps() -> None:
    with patch("coola.utils.tensor.torch.ones", Mock(return_value=torch.ones(1))):
        assert is_mps_available()


@torch_available
@patch("coola.utils.tensor.is_torch_available", lambda *args, **kwargs: True)
def test_is_mps_available_without_mps() -> None:
    with patch("coola.utils.tensor.torch.ones", Mock(side_effect=RuntimeError)):
        assert not is_mps_available()


@patch("coola.utils.tensor.is_torch_available", lambda *args, **kwargs: False)
def test_is_mps_available_no_torch() -> None:
    assert not is_mps_available()
