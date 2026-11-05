from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.testing.marks import numpy_available, torch_available
from coola.utils.imports import (
    MissingPackageError,
    clear_availability_caches,
    is_numpy_available,
    is_torch_available,
)
from coola.utils.tensor import (
    get_available_devices,
    is_cuda_available,
    is_mps_available,
    sparse_components,
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
    from collections.abc import Iterator, Sequence


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    clear_availability_caches()
    yield
    clear_availability_caches()


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
@patch("torch.cuda.is_available", lambda: True)
@patch("torch.cuda.device_count", lambda: 2)
@patch("coola.utils.tensor.is_mps_available", lambda: False)
def test_get_available_devices_one_entry_per_cuda_device() -> None:
    assert get_available_devices() == ("cpu", "cuda:0", "cuda:1")


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
@patch("torch.backends.mps.is_available", lambda: True)
def test_is_mps_available_with_mps() -> None:
    assert is_mps_available()


@torch_available
@patch("coola.utils.tensor.is_torch_available", lambda: True)
@patch("torch.backends.mps.is_available", lambda: False)
def test_is_mps_available_without_mps() -> None:
    assert not is_mps_available()


@torch_available
@patch("coola.utils.tensor.is_torch_available", lambda: True)
def test_is_mps_available_does_not_allocate_a_tensor() -> None:
    with patch("coola.utils.tensor.torch.ones", Mock(side_effect=RuntimeError("allocated"))):
        assert isinstance(is_mps_available(), bool)


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
    assert to_tensor(data).equal(torch.tensor([3.0, 1.0, 2.0, 0.0, 1.0], dtype=torch.float))


@numpy_available
@torch_available
def test_to_tensor_numpy() -> None:
    # A RuntimeError can be raised if torch and numpy are not compatible
    # RuntimeError: Numpy is not available
    with suppress(RuntimeError):
        assert to_tensor(np.array([3, 1, 2, 0, 1])).equal(
            torch.tensor([3, 1, 2, 0, 1], dtype=torch.long)
        )


@numpy_available
@torch_available
def test_to_tensor_numpy_negative_strides() -> None:
    out = to_tensor(np.array([1, 2, 3])[::-1])
    assert out.tolist() == [3, 2, 1]


@patch("coola.utils.imports.optional.is_torch_available", lambda: False)
def test_to_tensor_no_torch() -> None:
    with pytest.raises(MissingPackageError, match=r"'torch' package is required"):
        to_tensor([1, 2, 3])


###############################################
#     Tests for clear_availability_caches     #
###############################################


@torch_available
def test_clear_availability_caches_resets_is_cuda_available() -> None:
    with patch("torch.cuda.is_available", lambda: True):
        assert is_cuda_available()
    clear_availability_caches()
    with patch("torch.cuda.is_available", lambda: False):
        assert not is_cuda_available()


@torch_available
def test_clear_availability_caches_resets_is_mps_available() -> None:
    with patch("torch.backends.mps.is_available", lambda: True):
        assert is_mps_available()
    clear_availability_caches()
    with patch("torch.backends.mps.is_available", lambda: False):
        assert not is_mps_available()


@torch_available
def test_clear_availability_caches_resets_get_available_devices() -> None:
    with (
        patch("torch.cuda.is_available", lambda: False),
        patch("torch.backends.mps.is_available", lambda: False),
    ):
        assert get_available_devices() == ("cpu",)
    clear_availability_caches()
    with (
        patch("torch.cuda.is_available", lambda: False),
        patch("torch.backends.mps.is_available", lambda: True),
    ):
        assert get_available_devices() == ("cpu", "mps:0")


#######################################
#     Tests for sparse_components     #
#######################################


@torch_available
def test_sparse_components_coo() -> None:
    indices, values = sparse_components(torch.eye(2).to_sparse())
    assert indices.tolist() == [[0, 1], [0, 1]]
    assert values.tolist() == [1.0, 1.0]


@torch_available
def test_sparse_components_coo_is_coalesced() -> None:
    indices, values = sparse_components(
        torch.sparse_coo_tensor(torch.tensor([[1, 1]]), torch.tensor([1.0, 2.0]), (2,))
    )
    assert indices.tolist() == [[1]]
    assert values.tolist() == [3.0]


@torch_available
def test_sparse_components_csr() -> None:
    crow, col, values = sparse_components(torch.eye(2).to_sparse_csr())
    assert crow.tolist() == [0, 1, 2]
    assert col.tolist() == [0, 1]
    assert values.tolist() == [1.0, 1.0]


@torch_available
def test_sparse_components_csc() -> None:
    ccol, row, values = sparse_components(torch.eye(2).to_sparse_csc())
    assert ccol.tolist() == [0, 1, 2]
    assert row.tolist() == [0, 1]
    assert values.tolist() == [1.0, 1.0]


@torch_available
def test_sparse_components_bsr() -> None:
    crow, col, values = sparse_components(torch.eye(4).to_sparse_bsr((2, 2)))
    assert crow.tolist() == [0, 1, 2]
    assert col.tolist() == [0, 1]
    assert values.shape == (2, 2, 2)


@torch_available
def test_sparse_components_bsc() -> None:
    ccol, row, values = sparse_components(torch.eye(4).to_sparse_bsc((2, 2)))
    assert ccol.tolist() == [0, 1, 2]
    assert row.tolist() == [0, 1]
    assert values.shape == (2, 2, 2)


@torch_available
def test_sparse_components_strided_tensor_is_not_supported() -> None:
    with pytest.raises(TypeError, match=r"Unsupported tensor layout: torch\.strided"):
        sparse_components(torch.eye(2))
