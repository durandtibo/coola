r"""Implement a reducer using PyTorch functions.

The reducer is registered only if ``torch`` is available.
"""

from __future__ import annotations

__all__ = ["TorchReducer"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import InlineDisplayMixin
from coola.reducer.base import BaseBasicReducer
from coola.utils.imports import check_torch, is_torch_available
from coola.utils.tensor import to_tensor

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

T = TypeVar("T", Sequence[float], torch.Tensor)


class TorchReducer(InlineDisplayMixin, BaseBasicReducer[T]):
    r"""Implement a reducer based on torch functions.

    Raises:
        RuntimeError: if ``torch`` is not installed.

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

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}

    def __init__(self) -> None:
        check_torch()

    def _is_empty(self, values: T) -> bool:
        if isinstance(values, torch.Tensor):
            return values.numel() == 0
        return not values

    def _max(self, values: T) -> int | float:
        return to_tensor(values).max().item()

    def _mean(self, values: T) -> float:
        return _to_double(values).mean().item()

    def _median(self, values: T) -> int | float:
        # ``torch.median`` returns the lower of the two middle values;
        # average them to match ``statistics.median`` and ``numpy.median``.
        tensor = to_tensor(values).flatten()
        n = tensor.numel()
        ordered = tensor.sort()[0]
        if n % 2 == 1:
            return ordered[n // 2].item()
        return ((ordered[n // 2 - 1].double() + ordered[n // 2].double()) / 2).item()

    def _min(self, values: T) -> int | float:
        return to_tensor(values).min().item()

    def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        result: Any = torch.quantile(
            _to_double(values),
            _to_double(quantiles),
        )
        return result.tolist()

    def sort(self, values: T, descending: bool = False) -> list[int | float]:
        result: Any = torch.sort(to_tensor(values), descending=descending)[0]
        return result.tolist()

    def _std(self, values: T) -> float:
        tensor = _to_double(values)
        if tensor.numel() == 1:
            return float("nan")
        return tensor.std().item()


def _to_double(values: Any) -> torch.Tensor:
    """Convert the values to a ``float64`` tensor without first rounding
    Python floats to ``float32``."""
    if isinstance(values, torch.Tensor) or not isinstance(values, (list, tuple)):
        return to_tensor(values).double()
    return torch.as_tensor(values, dtype=torch.float64)
