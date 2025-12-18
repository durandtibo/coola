r"""Implement some utility functions for ``torch.Tensor``s."""

from __future__ import annotations

__all__ = ["get_available_devices", "is_cuda_available", "is_mps_available", "to_tensor"]

from functools import lru_cache
from typing import TYPE_CHECKING

from coola.utils.imports import is_numpy_available, is_torch_available

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


@lru_cache(1)
def get_available_devices() -> tuple[str, ...]:
    r"""Get the available PyTorch devices on the machine.

    Returns:
        The available devices.

    Example usage:

    ```pycon

    >>> from coola.utils.tensor import get_available_devices
    >>> get_available_devices()
    ('cpu'...)

    ```
    """
    devices = ["cpu"]
    if is_cuda_available():
        devices.append("cuda:0")
    if is_mps_available():
        devices.append("mps:0")
    return tuple(devices)


@lru_cache(1)
def is_cuda_available() -> bool:
    r"""Indicate if CUDA is currently available.

    Returns:
        A boolean indicating if CUDA is currently available.

    Example usage:

    ```pycon

    >>> from coola.utils.tensor import is_cuda_available
    >>> is_cuda_available()

    ```
    """
    return is_torch_available() and torch.cuda.is_available()


@lru_cache(1)
def is_mps_available() -> bool:
    r"""Indicate if MPS is currently available.

    Returns:
        A boolean indicating if MPS is currently available.

    Example usage:

    ```pycon

    >>> from coola.utils.tensor import is_mps_available
    >>> is_mps_available()

    ```
    """
    if not is_torch_available():
        return False
    try:
        torch.ones(1, device="mps")
    except RuntimeError:
        return False
    return True


def to_tensor(data: Sequence[int | float] | torch.Tensor | np.ndarray) -> torch.Tensor:
    r"""Convert the input to a ``torch.Tensor``.

    Args:
        data: The data to convert to a tensor.

    Returns:
        A tensor.

    Example usage:

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
    if is_numpy_available() and isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if not torch.is_tensor(data):
        return torch.as_tensor(data)
    return data
