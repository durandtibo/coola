from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.summary import SummarizerRegistry, TensorSummarizer
from coola.testing import torch_available
from coola.utils import is_torch_available

if is_torch_available():
    import torch

    cuda_available = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Requires a CUDA device"
    )
else:
    torch = Mock()
    cuda_available = pytest.mark.skipif(False, reason="Requires PyTorch and a CUDA device")


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


######################################
#     Tests for TensorSummarizer     #
######################################


@torch_available
def test_tensor_summarizer_init_default() -> None:
    """Test TensorSummarizer initialization with default parameters."""
    summarizer = TensorSummarizer()
    assert summarizer._show_data is False


@torch_available
def test_tensor_summarizer_init_show_data_true() -> None:
    """Test TensorSummarizer initialization with show_data=True."""
    summarizer = TensorSummarizer(show_data=True)
    assert summarizer._show_data is True


@torch_available
def test_tensor_summarizer_init_show_data_false() -> None:
    """Test TensorSummarizer initialization with show_data=False."""
    summarizer = TensorSummarizer(show_data=False)
    assert summarizer._show_data is False


@torch_available
def test_tensor_summarizer_repr_default() -> None:
    """Test __repr__ with default parameters."""
    assert repr(TensorSummarizer()) == "TensorSummarizer(show_data=False)"


@torch_available
def test_tensor_summarizer_repr_show_data_true() -> None:
    """Test __repr__ with show_data=True."""
    assert repr(TensorSummarizer(show_data=True)) == "TensorSummarizer(show_data=True)"


@torch_available
def test_tensor_summarizer_str_default() -> None:
    assert str(TensorSummarizer()) == "TensorSummarizer(show_data=False)"


@torch_available
def test_tensor_summarizer_str_show_data_true() -> None:
    assert str(TensorSummarizer(show_data=True)) == "TensorSummarizer(show_data=True)"


@torch_available
def test_tensor_summarizer_equal_same_instance() -> None:
    """Test equality between two summarizers with same configuration."""
    summarizer = TensorSummarizer(show_data=False)
    assert summarizer.equal(summarizer)


@torch_available
def test_tensor_summarizer_equal_same_config() -> None:
    """Test equality between two summarizers with same configuration."""
    summarizer1 = TensorSummarizer(show_data=False)
    summarizer2 = TensorSummarizer(show_data=False)
    assert summarizer1.equal(summarizer2)


@torch_available
def test_tensor_summarizer_equal_different_show_data() -> None:
    """Test inequality between summarizers with different show_data."""
    summarizer1 = TensorSummarizer(show_data=False)
    summarizer2 = TensorSummarizer(show_data=True)
    assert not summarizer1.equal(summarizer2)


@torch_available
def test_tensor_summarizer_equal_different_type() -> None:
    """Test inequality when comparing with different type."""
    assert not TensorSummarizer().equal(42)


@torch_available
def test_tensor_summarizer_equal_different_type_child() -> None:
    """Test inequality when comparing with different type."""

    class Child(TensorSummarizer): ...

    assert not TensorSummarizer().equal(Child())


@torch_available
def test_tensor_summarizer_summarize_1d_tensor_default(registry: SummarizerRegistry) -> None:
    """Test summarizing a 1D tensor with default settings (metadata
    only)."""
    result = TensorSummarizer().summarize(registry, torch.arange(11))
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([11]) | dtype=torch.int64 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_multidimensional_tensor(registry: SummarizerRegistry) -> None:
    """Test summarizing a multi-dimensional tensor."""
    result = TensorSummarizer().summarize(registry, torch.randn(2, 3, 4))
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([2, 3, 4]) | dtype=torch.float32 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_dtype_int32(registry: SummarizerRegistry) -> None:
    result = TensorSummarizer().summarize(registry, torch.tensor([1, 2, 3], dtype=torch.int32))
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([3]) | dtype=torch.int32 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_dtype_float64(registry: SummarizerRegistry) -> None:
    result = TensorSummarizer().summarize(registry, torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([2]) | dtype=torch.float64 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_dtype_bool(registry: SummarizerRegistry) -> None:
    result = TensorSummarizer().summarize(registry, torch.tensor([True, False], dtype=torch.bool))
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([2]) | dtype=torch.bool | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_show_data_true(registry: SummarizerRegistry) -> None:
    """Test summarizing with show_data=True returns full tensor
    representation."""
    tensor = torch.arange(5)
    result = TensorSummarizer(show_data=True).summarize(registry, tensor)
    assert result == "tensor([0, 1, 2, 3, 4])"


@torch_available
def test_tensor_summarizer_summarize_empty_tensor(registry: SummarizerRegistry) -> None:
    """Test summarizing an empty tensor."""
    tensor = torch.tensor([])
    result = TensorSummarizer().summarize(registry, tensor)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([0]) | dtype=torch.float32 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_scalar_tensor(registry: SummarizerRegistry) -> None:
    """Test summarizing a scalar tensor (0-dimensional)."""
    tensor = torch.tensor(42)
    result = TensorSummarizer().summarize(registry, tensor)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([]) | dtype=torch.int64 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_depth_ignored(registry: SummarizerRegistry) -> None:
    """Test that depth parameter is ignored."""
    summarizer = TensorSummarizer()
    tensor = torch.arange(5)
    result1 = summarizer.summarize(registry, tensor, depth=0)
    result2 = summarizer.summarize(registry, tensor, depth=5)
    assert result1 == result2


@torch_available
def test_tensor_summarizer_summarize_max_depth_ignored(registry: SummarizerRegistry) -> None:
    """Test that max_depth parameter is ignored."""
    summarizer = TensorSummarizer()
    tensor = torch.arange(5)
    result1 = summarizer.summarize(registry, tensor, max_depth=1)
    result2 = summarizer.summarize(registry, tensor, max_depth=10)
    assert result1 == result2


@torch_available
@cuda_available
def test_tensor_summarizer_summarize_cuda_tensor(registry: SummarizerRegistry) -> None:
    """Test summarizing a tensor on CUDA device."""
    tensor = torch.arange(5).cuda()
    result = TensorSummarizer().summarize(registry, tensor)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([3, 3]) | dtype=torch.float32 | "
        "device=cuda:0 | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_large_tensor(registry: SummarizerRegistry) -> None:
    """Test summarizing a very large tensor (metadata should still be
    compact)."""
    tensor = torch.randn(1000, 1000, 10)
    result = TensorSummarizer().summarize(registry, tensor)
    assert (
        result
        == "<class 'torch.Tensor'> | shape=torch.Size([1000, 1000, 10]) | dtype=torch.float32 | "
        "device=cpu | requires_grad=False"
    )


@torch_available
def test_tensor_summarizer_summarize_requires_grad_tensor(registry: SummarizerRegistry) -> None:
    """Test summarizing a tensor with requires_grad=True."""
    tensor = torch.randn(3, 3, requires_grad=True)
    result = TensorSummarizer().summarize(registry, tensor)
    assert (
        result == "<class 'torch.Tensor'> | shape=torch.Size([3, 3]) | dtype=torch.float32 | "
        "device=cpu | requires_grad=True"
    )
