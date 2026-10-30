r"""Implement handlers for ``torch.Tensor``s."""

from __future__ import annotations

__all__ = ["tensor_close", "tensor_equal"]

from typing import TYPE_CHECKING

from coola.equality._helpers.elementwise import elementwise_close
from coola.utils.imports import is_torch_available

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


def tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, config: EqualityConfig) -> bool:
    r"""Indicate if the two tensors are equal within a tolerance.

    Args:
        tensor1: The first tensor to compare.
        tensor2: The second tensor to compare.
        config: The equality configuration.

    Returns:
        ``True`` if the two tensors are equal within a tolerance,
            otherwise ``False``.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.torch import tensor_equal
        >>> config = EqualityConfig()
        >>> tensor_equal(torch.ones(2, 3), torch.ones(2, 3), config)
        True
        >>> tensor_equal(torch.ones(2, 3), torch.zeros(2, 3), config)
        False

        ```
    """
    if tensor1.device.type == "meta" or tensor2.device.type == "meta":
        # Meta tensors have no data: only the metadata can be compared.
        return tensor1.shape == tensor2.shape and tensor1.dtype == tensor2.dtype
    if tensor1.layout != tensor2.layout:
        return False
    if tensor1.layout != torch.strided:
        # ``equal`` and ``allclose`` are not implemented for sparse layouts.
        tensor1, tensor2 = tensor1.to_dense(), tensor2.to_dense()
    if torch.bool in (tensor1.dtype, tensor2.dtype):
        # A boolean is a label, not a quantity: no tolerance (and no NaN).
        return tensor1.equal(tensor2)
    if config.equal_nan or config.atol > 0 or config.rtol > 0:
        return _tensors_close(tensor1, tensor2, config)
    return tensor1.equal(tensor2)


def tensor_close(
    tensor1: torch.Tensor, tensor2: torch.Tensor, config: EqualityConfig
) -> torch.Tensor:
    r"""Indicate for each element if two tensors are equal within a
    tolerance.

    It is the rule of ``tensor_equal`` for each pair of elements, so it
    gives the location of the first difference. The two tensors must
    have the same shape, dtype and layout, and must not be on the
    ``meta`` device.

    Args:
        tensor1: The first tensor to compare.
        tensor2: The second tensor to compare.
        config: The equality configuration.

    Returns:
        A boolean tensor with the shape of the inputs.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.torch import tensor_close
        >>> tensor_close(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]), EqualityConfig())
        tensor([ True, False])

        ```
    """
    if tensor1.layout != torch.strided:
        tensor1, tensor2 = tensor1.to_dense(), tensor2.to_dense()
    if torch.bool not in (tensor1.dtype, tensor2.dtype) and (
        config.equal_nan or config.atol > 0 or config.rtol > 0
    ):
        if tensor1.dtype == torch.uint64:
            tensor1, tensor2 = tensor1.to(torch.float64), tensor2.to(torch.float64)
        return elementwise_close(
            tensor1,
            tensor2,
            atol=config.atol,
            rtol=config.rtol,
            equal_nan=config.equal_nan,
            xp=torch,
        )
    return tensor1 == tensor2


def _tensors_close(tensor1: torch.Tensor, tensor2: torch.Tensor, config: EqualityConfig) -> bool:
    r"""Indicate if two tensors are equal within the tolerance.

    Args:
        tensor1: The first tensor.
        tensor2: The second tensor.
        config: The configuration of the comparison.

    Returns:
        ``True`` if the tensors have the same shape and dtype and all
            the elements are close, otherwise ``False``.
    """
    if tensor1.shape != tensor2.shape or tensor1.dtype != tensor2.dtype:
        return False
    if tensor1.dtype == torch.uint64:
        # ``torch.maximum`` is not implemented for ``uint64``: converted, like before.
        tensor1, tensor2 = tensor1.to(torch.float64), tensor2.to(torch.float64)
    close = elementwise_close(
        tensor1,
        tensor2,
        atol=config.atol,
        rtol=config.rtol,
        equal_nan=config.equal_nan,
        xp=torch,
    )
    return bool(close.all().item())
