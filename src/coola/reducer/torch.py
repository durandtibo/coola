r"""Implement a reducer using PyTorch functions.

The reducer is registered only if ``torch`` is available.
"""

from __future__ import annotations

__all__ = ["TorchReducer"]

import math
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.reducer.base import BaseReducer
from coola.utils.imports import check_torch, is_torch_available
from coola.utils.tensor import to_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


class TorchReducer(InlineDisplayMixin, BaseReducer["Sequence[float] | torch.Tensor"]):
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

    def _max(self, values: Sequence[float] | torch.Tensor) -> int | float:
        return _to_tensor(values).max().item()

    def _mean(self, values: Sequence[float] | torch.Tensor) -> float:
        return _to_float_tensor(values).mean().item()

    def _median(self, values: Sequence[float] | torch.Tensor) -> int | float:
        tensor = _to_tensor(values).flatten()
        if tensor.isnan().any():
            return float("nan")
        result = _median_of(tensor)
        if math.isinf(result) and tensor.isfinite().all():
            # The mean of the two middle values overflowed: halve the values first.
            return _median_of(_to_float_tensor(tensor) / 2) * 2
        return result

    def _min(self, values: Sequence[float] | torch.Tensor) -> int | float:
        return _to_tensor(values).min().item()

    def _quantile(
        self, values: Sequence[float] | torch.Tensor, quantiles: Sequence[float]
    ) -> list[float]:
        tensor = _to_float_tensor(values)
        if tensor.dtype not in {torch.float32, torch.float64}:
            tensor = tensor.float()  # ``torch.quantile`` only supports float32/float64
        result: Any = torch.quantile(tensor, torch.as_tensor(quantiles, dtype=tensor.dtype))
        return result.tolist()

    def _sort(
        self, values: Sequence[float] | torch.Tensor, *, descending: bool
    ) -> list[int | float]:
        result: Any = torch.sort(_to_tensor(values), descending=descending)[0]
        return result.tolist()

    def _std(self, values: Sequence[float] | torch.Tensor) -> float:
        tensor = _to_float_tensor(values)
        if tensor.numel() == 1:
            return float("nan")
        return tensor.std().item()


def _median_of(tensor: torch.Tensor) -> int | float:
    # ``torch.median`` returns the lower of the two middle values;
    # average them to match ``statistics.median`` and ``numpy.median``.
    n = tensor.numel()
    ordered = tensor.sort()[0]
    if n % 2 == 1:
        return ordered[n // 2].item()
    middle = _to_float_tensor(ordered[n // 2 - 1 : n // 2 + 1])
    return middle.mean().item()


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
