r"""Provide deterministic hashing for PyTorch tensors."""

from __future__ import annotations

__all__ = ["TorchTensorHasher"]

from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.bytes import hash_bytes
from coola.hashing.numpy import hash_array_content
from coola.utils.imports import is_torch_available

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class TorchTensorHasher(InlineDisplayMixin, BaseHasher["torch.Tensor"]):
    r"""Hasher for ``torch.Tensor`` objects.

    The hash depends on the dtype, the shape and the values. The device
    and ``requires_grad`` are not part of the hash, so a tensor and its
    copy on another device hash the same. ``-0.0`` hashes like ``0.0``.
    A sparse tensor is hashed in its dense form. A ``meta`` tensor has
    no values and raises an error.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.hashing import HasherRegistry, TorchTensorHasher
        >>> hasher = TorchTensorHasher()
        >>> hasher
        TorchTensorHasher()
        >>> len(hasher.hash(torch.arange(3), registry=HasherRegistry()))
        64

        ```
    """

    def hash(
        self,
        data: torch.Tensor,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        tensor = data.detach()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        tensor = tensor.cpu()
        if tensor.is_floating_point() or tensor.is_complex():
            tensor = tensor + 0  # -0.0 becomes 0.0
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return hash_array_content(
            "torch.Tensor",
            str(tensor.dtype),
            tuple(tensor.shape),
            hash_bytes(raw, length=length),
            length=length,
        )
