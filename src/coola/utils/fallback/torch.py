r"""Contain fallback implementations used when ``torch`` dependency is
not available."""

from __future__ import annotations

__all__ = ["torch"]

from types import ModuleType
from typing import Any

from coola.utils.imports import raise_error_torch_missing


class PackedSequence:
    r"""Fake PackedSequence class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_torch_missing()


class Tensor:
    r"""Fake Tensor class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_error_torch_missing()


# Create a fake torch package
torch: ModuleType = ModuleType("torch")
torch.nn = ModuleType("torch.nn")
torch.nn.utils = ModuleType("torch.nn.utils")
torch.nn.utils.rnn = ModuleType("torch.nn.utils.rnn")
torch.nn.utils.rnn.PackedSequence = PackedSequence
torch.Tensor = Tensor
