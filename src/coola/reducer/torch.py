r"""Implement a reducer using PyTorch functions.

The reducer is registered only if ``torch`` is available.
"""

from __future__ import annotations

__all__ = ["TorchReducer"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.reducer.base import BaseBasicReducer
from coola.utils.imports import check_torch, is_torch_available
from coola.utils.tensor import to_tensor

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

T = TypeVar("T", Sequence[float], torch.Tensor)


class TorchReducer(InlineDisplayMixin, NoArgsDisplayMixin, BaseBasicReducer[T]):
    r"""Implement a reducer based on torch functions.

    Raises:
        MissingPackageError: if ``torch`` is not installed.

    Example:
        ```pycon
        >>> from coola.reducer import TorchReducer
        >>> reducer = TorchReducer()
        >>> reducer.max([-2, -1, 0, 1, 2])
        2
        >>> reducer.median([-2, -1, 0, 1, 2])
        0
        >>> reducer.sort([2, 1, -2, 3, 0])
        [-2, 0, 1, 2, 3]

        ```
    """

    def __init__(self) -> None:
        check_torch()

    def _is_empty(self, values: T) -> bool:
        if isinstance(values, torch.Tensor):
            return values.numel() == 0
        return not values

    def _max(self, values: T) -> int | float:
        return _to_tensor(values).max().item()

    def _mean(self, values: T) -> float:
        return _to_float_tensor(values).mean().item()

    def _median(self, values: T) -> int | float:
        # ``torch.median`` returns the lower of the two middle values;
        # average them to match ``statistics.median`` and ``numpy.median``.
        tensor = _to_tensor(values).flatten()
        n = tensor.numel()
        ordered = tensor.sort()[0]
        if n % 2 == 1:
            return ordered[n // 2].item()
        middle = _to_float_tensor(ordered[n // 2 - 1 : n // 2 + 1])
        return middle.mean().item()

    def _min(self, values: T) -> int | float:
        return _to_tensor(values).min().item()

    def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        tensor = _to_float_tensor(values)
        if tensor.dtype not in {torch.float32, torch.float64}:
            tensor = tensor.float()  # ``torch.quantile`` only supports float32/float64
        result: Any = torch.quantile(tensor, torch.as_tensor(quantiles, dtype=tensor.dtype))
        return result.tolist()

    def sort(self, values: T, descending: bool = False) -> list[int | float]:
        result: Any = torch.sort(_to_tensor(values), descending=descending)[0]
        return result.tolist()

    def _std(self, values: T) -> float:
        tensor = _to_float_tensor(values)
        if tensor.numel() == 1:
            return float("nan")
        return tensor.std().item()


def _to_float_tensor(values: Any) -> torch.Tensor:
    """Convert the values to a floating-point tensor of similar
    precision.

    Python numbers are double precision, so lists and tuples are
    converted to ``float64`` (instead of the ``float32`` default of
    torch). Tensors and arrays that are already floating-point keep
    their dtype; integer and boolean ones are promoted to ``float64``.
    """
    if isinstance(values, (list, tuple)):
        return torch.as_tensor(values, dtype=torch.float64)
    tensor = _to_tensor(values)
    return tensor if tensor.is_floating_point() else tensor.double()


def _to_tensor(values: Any) -> torch.Tensor:
    """Convert the values to a tensor without losing precision.

    Python floats are double precision, so a list or tuple that contains
    a float is converted to ``float64`` (instead of the ``float32``
    default of torch). A list of ints stays integer. Tensors and arrays
    keep their dtype.
    """
    if isinstance(values, (list, tuple)):
        has_float = any(isinstance(value, float) for value in values)
        return torch.as_tensor(values, dtype=torch.float64 if has_float else None)
    return to_tensor(values)
