r"""Implement a reducer using PyTorch functions.

The reducer is registered only if ``torch`` is available.
"""

from __future__ import annotations

__all__ = ["TorchReducer"]

from collections.abc import Sequence
from typing import TypeVar, Union
from unittest.mock import Mock

from coola.reducers.base import BaseBasicReducer
from coola.reducers.registry import ReducerRegistry
from coola.utils import check_torch, is_torch_available
from coola.utils.tensor import to_tensor

if is_torch_available():
    import torch
else:
    torch = Mock()  # pragma: no cover

T = TypeVar("T", Sequence[Union[int, float]], torch.Tensor)


class TorchReducer(BaseBasicReducer[T]):
    r"""Implement a reducer based on torch functions.

    Raises:
        RuntimeError: if ``torch`` is not installed.

    Example usage:

    ```pycon

    >>> from coola.reducers import TorchReducer
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

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _is_empty(self, values: T) -> bool:
        if torch.is_tensor(values):
            return values.numel() == 0
        return not values

    def _max(self, values: T) -> int | float:
        return to_tensor(values).max().item()

    def _mean(self, values: T) -> float:
        return to_tensor(values).float().mean().item()

    def _median(self, values: T) -> int | float:
        return to_tensor(values).median().item()

    def _min(self, values: T) -> int | float:
        return to_tensor(values).min().item()

    def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        return torch.quantile(
            to_tensor(values).float(),
            to_tensor(quantiles).float(),
        ).tolist()

    def sort(self, values: T, descending: bool = False) -> list[int | float]:
        return torch.sort(to_tensor(values), descending=descending)[0].tolist()

    def _std(self, values: T) -> float:
        values = to_tensor(values).float()
        if values.numel() == 1:
            return float("nan")
        return values.std().item()


if is_torch_available() and not ReducerRegistry.has_reducer("torch"):  # pragma: no cover
    ReducerRegistry.add_reducer("torch", TorchReducer())
