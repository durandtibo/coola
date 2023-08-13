from unittest.mock import patch

from coola.testing import torch_available
from coola.utils.tensor import (
    get_available_devices,
    is_cuda_available,
    is_mps_available,
)

###########################################
#     Tests for get_available_devices     #
###########################################


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: False)
@patch("torch.backends.mps.is_available", lambda *args, **kwargs: False)
def test_get_available_devices_cpu() -> None:
    assert get_available_devices() == ("cpu",)


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
@patch("torch.cuda.device_count", lambda *args, **kwargs: 1)
@patch("torch.backends.mps.is_available", lambda *args, **kwargs: False)
def test_get_available_devices_cpu_and_gpu() -> None:
    assert get_available_devices() == ("cpu", "cuda:0")


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: False)
@patch("torch.backends.mps.is_available", lambda *args, **kwargs: True)
def test_get_available_devices_cpu_and_mps() -> None:
    assert get_available_devices() == ("cpu", "mps:0")


@torch_available
@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
@patch("torch.cuda.device_count", lambda *args, **kwargs: 1)
@patch("torch.backends.mps.is_available", lambda *args, **kwargs: True)
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


######################################
#     Tests for is_mpa_available     #
######################################


@torch_available
def test_is_mps_available() -> None:
    assert isinstance(is_mps_available(), bool)
