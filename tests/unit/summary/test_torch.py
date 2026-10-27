from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.summary import SummaryConfig, TensorSummarizer
from coola.testing.fixtures import torch_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch

    cuda_available = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Requires a CUDA device"
    )
else:
    torch = Mock()
    cuda_available = pytest.mark.skipif(False, reason="Requires PyTorch and a CUDA device")

CONFIG = SummaryConfig()
SHOW_DATA = SummaryConfig(show_data=True)


######################################
#     Tests for TensorSummarizer     #
######################################


@torch_available
def test_tensor_summarizer_summarize_1d_tensor_default() -> None:
    """Test summarizing a 1D tensor with default settings (metadata
    only)."""
    result = TensorSummarizer().summarize(torch.arange(11), CONFIG, compact=False)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([11]) | dtype=torch.int64 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_multidimensional_tensor() -> None:
    """Test summarizing a multi-dimensional tensor."""
    result = TensorSummarizer().summarize(torch.empty(2, 3, 4), CONFIG, compact=False)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([2, 3, 4]) | dtype=torch.float32 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_dtype_int32() -> None:
    result = TensorSummarizer().summarize(
        torch.tensor([1, 2, 3], dtype=torch.int32), CONFIG, compact=False
    )
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([3]) | dtype=torch.int32 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_dtype_float64() -> None:
    result = TensorSummarizer().summarize(
        torch.tensor([1.0, 2.0], dtype=torch.float64), CONFIG, compact=False
    )
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([2]) | dtype=torch.float64 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_dtype_bool() -> None:
    result = TensorSummarizer().summarize(
        torch.tensor([True, False], dtype=torch.bool), CONFIG, compact=False
    )
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([2]) | dtype=torch.bool | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_show_data_true() -> None:
    """Test summarizing with show_data=True returns full tensor
    representation."""
    result = TensorSummarizer().summarize(torch.arange(5), SHOW_DATA, compact=False)
    assert result == "tensor([0, 1, 2, 3, 4])"


@torch_available
def test_tensor_summarizer_summarize_empty_tensor() -> None:
    """Test summarizing an empty tensor."""
    result = TensorSummarizer().summarize(torch.tensor([]), CONFIG, compact=False)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([0]) | dtype=torch.float32 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_scalar_tensor() -> None:
    """Test summarizing a scalar tensor (0-dimensional)."""
    result = TensorSummarizer().summarize(torch.tensor(42), CONFIG, compact=False)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([]) | dtype=torch.int64 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_meta_tensor() -> None:
    """Test summarizing a tensor on a device that is not the CPU (the
    meta device exists without a GPU)."""
    tensor = torch.empty(5, device="meta")
    result = TensorSummarizer().summarize(tensor, CONFIG, compact=False)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([5]) | dtype=torch.float32 | "
        "device=meta | requires_grad=False"
    )


@torch_available
@cuda_available
def test_tensor_summarizer_summarize_cuda_tensor() -> None:
    """Test summarizing a tensor on CUDA device."""
    tensor = torch.empty(5).cuda()
    result = TensorSummarizer().summarize(tensor, CONFIG, compact=False)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([5]) | dtype=torch.float32 | "
        "device=cuda:0 | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_large_tensor() -> None:
    """Test summarizing a very large tensor (metadata should still be
    compact)."""
    tensor = torch.empty(1000, 1000, 10)
    result = TensorSummarizer().summarize(tensor, CONFIG, compact=False)
    assert (
        result
        == "<class 'torch.Tensor'> | shape=torch.Size([1000, 1000, 10]) | dtype=torch.float32 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_requires_grad_tensor() -> None:
    """Test summarizing a tensor with requires_grad=True."""
    tensor = torch.empty(3, 3, requires_grad=True)
    result = TensorSummarizer().summarize(tensor, CONFIG, compact=False)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([3, 3]) | dtype=torch.float32 | "
        "device=cpu | requires_grad=True"
    )


@torch_available
def test_tensor_summarizer_summarize_compact_is_ignored() -> None:
    summarizer = TensorSummarizer()
    assert summarizer.summarize(torch.arange(5), CONFIG, compact=True) == summarizer.summarize(
        torch.arange(5), CONFIG, compact=False
    )
