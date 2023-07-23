from __future__ import annotations

__all__ = ["TorchReducer"]

from collections.abc import Sequence
from unittest.mock import Mock

from coola.reducers.base import BaseBasicReducer
from coola.reducers.registry import ReducerRegistry
from coola.utils import check_torch, is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()  # pragma: no cover


class TorchReducer(BaseBasicReducer):
    r"""Implement a reducer based on torch functions.

    Raises:
        RuntimeError if ``torch`` is not installed.
    """

    def __init__(self) -> None:
        check_torch()

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _max(self, values: Sequence[int | float]) -> int | float:
        return torch.as_tensor(values).max().item()

    def _mean(self, values: Sequence[int | float]) -> float:
        return torch.as_tensor(values, dtype=torch.float).mean().item()

    def _median(self, values: Sequence[int | float]) -> int | float:
        return torch.as_tensor(values).median().item()

    def _min(self, values: Sequence[int | float]) -> int | float:
        return torch.as_tensor(values).min().item()

    def _quantile(self, values: Sequence[int | float], quantiles: Sequence[float]) -> list[float]:
        return torch.quantile(
            torch.as_tensor(values, dtype=torch.float),
            torch.as_tensor(quantiles, dtype=torch.float),
        ).tolist()

    def sort(self, values: Sequence[int | float], descending: bool = False) -> list[int | float]:
        return torch.sort(torch.as_tensor(values), descending=descending)[0].tolist()

    def _std(self, values: Sequence[int | float]) -> float:
        return torch.as_tensor(values, dtype=torch.float).std().item()


if is_torch_available():  # pragma: no cover
    if not ReducerRegistry.has_reducer("torch"):
        ReducerRegistry.add_reducer("torch", TorchReducer())
