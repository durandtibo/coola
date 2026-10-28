r"""Implement a reducer using PyTorch functions.

The reducer is registered only if ``torch`` is available.
"""

from __future__ import annotations

__all__ = ["TorchReducer"]

import math
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.reducer.base import BaseReducer
from coola.tools.tensor import to_tensor
from coola.utils.imports import check_torch, is_numpy_available, is_torch_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

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

    def _prepare(self, values: Sequence[float] | torch.Tensor) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            tensor = values
        elif is_numpy_available() and isinstance(values, np.ndarray):
            if values.dtype.kind not in "biuf":
                self._raise_unsupported(f"an array of dtype {values.dtype}")
            tensor = to_tensor(values)
        else:
            tensor = self._to_tensor(values)
        if tensor.is_complex():
            self._raise_unsupported("complex numbers")
        if tensor.ndim != 1:
            self._raise_ndim(tensor.ndim)
        return tensor

    def _to_tensor(self, values: Sequence[float]) -> torch.Tensor:
        # Python floats are double precision, so a list that contains a float is converted
        # to ``float64`` (instead of the ``float32`` default of torch).
        has_float = any(isinstance(value, float) for value in values)
        try:
            return torch.as_tensor(values, dtype=torch.float64 if has_float else None)
        except (OverflowError, RuntimeError, TypeError, ValueError):
            self._raise_unsupported(
                "this input (an integer beyond 64 bits, a string, a Decimal, None, nested "
                "sequences of different lengths or another type)",
                "use NativeReducer for arbitrary precision integers",
            )

    def _max(self, values: torch.Tensor) -> int | float:
        return values.max().item()

    def _mean(self, values: torch.Tensor) -> float:
        return _to_float_tensor(values).mean().item()

    def _median(self, values: torch.Tensor) -> int | float:
        if values.isnan().any():
            return float("nan")
        result = _median_of(values)
        if math.isinf(result) and values.isfinite().all():
            # The mean of the two middle values overflowed: halve the values first.
            return _median_of(_to_float_tensor(values) / 2) * 2
        return result

    def _min(self, values: torch.Tensor) -> int | float:
        return values.min().item()

    def _quantile(self, values: torch.Tensor, quantiles: Sequence[float]) -> list[float]:
        tensor = _to_float_tensor(values)
        if tensor.dtype not in {torch.float32, torch.float64}:
            tensor = tensor.float()  # ``torch.quantile`` only supports float32/float64
        result: Any = torch.quantile(tensor, torch.as_tensor(quantiles, dtype=tensor.dtype))
        return result.tolist()

    def _sort(self, values: torch.Tensor, *, descending: bool) -> list[int | float]:
        result: Any = torch.sort(values, descending=descending)[0]
        return result.tolist()

    def _std(self, values: torch.Tensor) -> float:
        if values.numel() == 1:
            return float("nan")
        return _to_float_tensor(values).std().item()


def _median_of(tensor: torch.Tensor) -> int | float:
    # ``torch.median`` returns the lower of the two middle values;
    # average them to match ``statistics.median`` and ``numpy.median``.
    n = tensor.numel()
    ordered = tensor.sort()[0]
    if n % 2 == 1:
        return ordered[n // 2].item()
    middle = _to_float_tensor(ordered[n // 2 - 1 : n // 2 + 1])
    return middle.mean().item()


def _to_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to a floating-point tensor.

    Floating-point tensors keep their dtype; integer and boolean ones
    are promoted to ``float64``.
    """
    return tensor if tensor.is_floating_point() else tensor.double()
