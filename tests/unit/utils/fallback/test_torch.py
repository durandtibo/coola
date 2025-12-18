from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.torch import torch


def test_torch_is_module_type() -> None:
    assert isinstance(torch, ModuleType)


def test_torch_module_name() -> None:
    assert torch.__name__ == "torch"


def test_torch_nested_module_access() -> None:
    assert hasattr(torch, "nn")
    assert hasattr(torch.nn, "utils")
    assert hasattr(torch.nn.utils, "rnn")


def test_torch_packed_sequence_class_exists() -> None:
    assert hasattr(torch.nn.utils.rnn, "PackedSequence")


def test_torch_packed_sequence_is_class() -> None:
    assert isinstance(torch.nn.utils.rnn.PackedSequence, type)


def test_torch_packed_sequence_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."):
        torch.nn.utils.rnn.PackedSequence()


def test_torch_tensor_class_exists() -> None:
    assert hasattr(torch, "Tensor")


def test_torch_tensor_class_is_class() -> None:
    assert isinstance(torch.Tensor, type)


def test_torch_tensor_class_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."):
        torch.Tensor()


def test_torch_tensor_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."):
        torch.tensor()
