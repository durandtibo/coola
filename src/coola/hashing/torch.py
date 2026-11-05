r"""Provide deterministic hashing for PyTorch tensors."""

from __future__ import annotations

__all__ = ["TorchTensorHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, digest
from coola.hashing.numpy import hash_array_content
from coola.utils.imports import is_torch_available
from coola.utils.tensor import sparse_components

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if TYPE_CHECKING:
    from coola.hashing.base import Hasher


class TorchTensorHasher(InlineDisplayMixin, BaseHasher["torch.Tensor"]):
    r"""Hasher for ``torch.Tensor`` objects.

    The hash depends on the dtype, the shape and the values. The device
    and ``requires_grad`` are not part of the hash, so a tensor and its
    copy on another device hash the same. ``-0.0`` hashes like ``0.0``.
    A sparse tensor is hashed from its layout, its positions and its
    stored values, without a dense copy, so it does not hash like the
    dense tensor with the same values. A ``meta`` tensor has no values
    and raises a ``TypeError``.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.hashing import HasherRegistry, TorchTensorHasher
        >>> hasher = TorchTensorHasher()
        >>> hasher
        TorchTensorHasher()
        >>> hasher.encode(torch.arange(3)).startswith(b"torch.Tensor")
        True

        ```
    """

    def encode(self, data: torch.Tensor) -> Encoding:
        r"""Describe a PyTorch tensor.

        The dense values on the CPU are described: the device, the
        layout and ``requires_grad`` are not part of the encoding.

        Args:
            data: The tensor to describe.

        Returns:
            The leaf that describes ``data``.

        Raises:
            TypeError: If ``data`` is on the ``meta`` device.
        """
        tensor = data.detach()
        if tensor.device.type == "meta":
            msg = "Cannot hash a tensor on the meta device: it has no data"
            raise TypeError(msg)
        if tensor.layout != torch.strided:
            # The stored components are hashed: a dense copy would need the
            # memory of the product of the shape.
            raw = b"".join(
                digest(_raw_bytes(component)) for component in sparse_components(tensor.cpu())
            )
            return hash_array_content(
                "torch.Tensor",
                f"{tensor.dtype}\0{tensor.layout}",
                tuple(tensor.shape),
                digest(raw),
            )
        return hash_array_content(
            "torch.Tensor",
            str(tensor.dtype),
            tuple(tensor.shape),
            digest(_raw_bytes(tensor)),
        )


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    r"""Return the canonical bytes of the values of a strided tensor.

    ``-0.0`` becomes ``0.0`` and every NaN becomes the default NaN. The
    device is not part of the bytes.

    Args:
        tensor: The strided tensor.

    Returns:
        The bytes of the values in row-major order.
    """
    tensor = tensor.cpu()
    values = tensor
    if tensor.is_complex():
        values = torch.view_as_real(tensor.contiguous())  # the real and imaginary parts
    if values.is_floating_point():
        values = torch.where(torch.isnan(values), torch.full_like(values, float("nan")), values + 0)
    return values.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the PyTorch tensors.

    Returns:
        The hashers of the PyTorch tensors and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the PyTorch tensors.
    """
    if not is_torch_available():
        return {}
    return {torch.Tensor: TorchTensorHasher()}
