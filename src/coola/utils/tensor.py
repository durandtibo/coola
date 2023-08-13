from __future__ import annotations

__all__ = ["get_available_devices", "is_cuda_available", "is_mps_available"]

from unittest.mock import Mock

from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


def get_available_devices() -> tuple[str, ...]:
    r"""Gets the available PyTorch devices on the machine.

    Returns
    -------
        tuple: The available devices.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.tensor import get_available_devices
        >>> get_available_devices()  # doctest:+ELLIPSIS
        ('cpu'...)
    """
    devices = ["cpu"]
    if is_cuda_available():
        devices.append("cuda:0")
    if is_mps_available():
        devices.append("mps:0")
    return tuple(devices)


def is_cuda_available() -> bool:
    r"""Indicates if CUDA is currently available.

    Returns:
    -------
        bool: A boolean indicating if CUDA is currently available.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.tensor import is_cuda_available
        >>> is_cuda_available()
    """
    return is_torch_available() and torch.cuda.is_available()


def is_mps_available() -> bool:
    r"""Indicates if MPS is currently available.

    Returns:
    -------
        bool: A boolean indicating if MPS is currently available.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.tensor import is_mps_available
        >>> is_mps_available()
    """
    return (
        is_torch_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
