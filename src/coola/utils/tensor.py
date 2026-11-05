r"""Implement some utility functions for ``torch.Tensor``s."""

from __future__ import annotations

__all__ = [
    "get_available_devices",
    "is_cuda_available",
    "is_mps_available",
    "sparse_components",
    "to_tensor",
]

from typing import TYPE_CHECKING, Any, cast

from coola.utils.imports import (
    availability_cache,
    check_torch,
    is_numpy_available,
    is_torch_available,
)

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


@availability_cache
def get_available_devices() -> tuple[str, ...]:
    r"""Get the available PyTorch devices on the machine.

    The result is cached until ``clear_availability_caches`` is called.

    Returns:
        The available devices: ``"cpu"``, ``"cuda:<i>"`` for each CUDA
            device and ``"mps:0"`` if MPS is available.

    Example:
        ```pycon
        >>> from coola.utils.tensor import get_available_devices
        >>> get_available_devices()
        ('cpu'...)

        ```
    """
    devices = ["cpu"]
    if is_cuda_available():
        devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    if is_mps_available():
        devices.append("mps:0")
    return tuple(devices)


@availability_cache
def is_cuda_available() -> bool:
    r"""Indicate if CUDA is currently available.

    Returns:
        A boolean indicating if CUDA is currently available.

    Example:
        ```pycon
        >>> from coola.utils.tensor import is_cuda_available
        >>> is_cuda_available()

        ```
    """
    return is_torch_available() and torch.cuda.is_available()


@availability_cache
def is_mps_available() -> bool:
    r"""Indicate if MPS is currently available.

    It does not allocate a tensor on the device.

    Returns:
        A boolean indicating if MPS is currently available.

    Example:
        ```pycon
        >>> from coola.utils.tensor import is_mps_available
        >>> is_mps_available()

        ```
    """
    return is_torch_available() and torch.backends.mps.is_available()


def to_tensor(data: Sequence[int | float] | torch.Tensor | np.ndarray) -> torch.Tensor:
    r"""Convert the input to a ``torch.Tensor``.

    Args:
        data: The data to convert to a tensor.

    Returns:
        A tensor. A tensor is returned as is, and the tensor of a
            NumPy array may share its memory with the array; a list is
            copied.

    Raises:
        MissingPackageError: if ``torch`` is not installed.

    Example:
        ```pycon
        >>> from coola.utils.tensor import to_tensor
        >>> x = to_tensor([1, 2, 3, 4, 5])
        >>> x
        tensor([1, 2, 3, 4, 5])
        >>> import numpy as np
        >>> x = to_tensor(np.array([1, 2, 3, 4, 5]))
        >>> x
        tensor([1, 2, 3, 4, 5])

        ```
    """
    check_torch()
    if is_numpy_available() and isinstance(data, np.ndarray):
        if any(stride < 0 for stride in data.strides):
            data = data.copy()  # ``torch.from_numpy`` does not support negative strides
        return cast("Any", torch).from_numpy(data)
    if not torch.is_tensor(data):
        return torch.as_tensor(data)
    return data


def sparse_components(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    r"""Return the strided tensors that store a sparse tensor.

    The positions are returned first, then the values. A COO tensor is
    coalesced first, so two COO tensors that hold the same values have
    the same components. No dense copy is made.

    Args:
        tensor: The sparse tensor (COO, CSR, CSC, BSR or BSC layout).

    Returns:
        The positions (one or two tensors) followed by the values.

    Raises:
        TypeError: If the layout of the tensor is not supported.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.utils.tensor import sparse_components
        >>> indices, values = sparse_components(torch.eye(2).to_sparse())
        >>> indices
        tensor([[0, 1],
                [0, 1]])
        >>> values
        tensor([1., 1.])

        ```
    """
    layout = tensor.layout
    if layout == torch.sparse_coo:
        tensor = tensor.coalesce()
        return tensor.indices(), tensor.values()
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        return tensor.crow_indices(), tensor.col_indices(), tensor.values()
    if layout in (torch.sparse_csc, torch.sparse_bsc):
        return tensor.ccol_indices(), tensor.row_indices(), tensor.values()
    msg = f"Unsupported tensor layout: {layout}"
    raise TypeError(msg)
