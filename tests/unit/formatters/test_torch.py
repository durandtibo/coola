from unittest.mock import Mock

import pytest

from coola import Summarizer, summary
from coola.formatters.torch_ import TensorFormatter
from coola.testing import torch_available
from coola.utils import is_torch_available

if is_torch_available():
    import torch

    cuda_available = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Requires a CUDA device"
    )
    DEVICES = ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",)
else:
    torch = Mock()
    cuda_available = pytest.mark.skipif(False, reason="Requires PyTorch and a CUDA device")
    DEVICES = ()


@torch_available
def test_summary_tensor() -> None:
    assert (
        summary(torch.ones(2, 3))
        == "<class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu"
    )


#####################################
#     Tests for TensorFormatter     #
#####################################


@torch_available
def test_tensor_formatter_str() -> None:
    assert str(TensorFormatter()).startswith("TensorFormatter(")


@torch_available
def test_tensor_formatter_clone_show_data_10() -> None:
    formatter = TensorFormatter()
    formatter_cloned = formatter.clone()
    formatter.set_show_data(True)
    assert formatter is not formatter_cloned
    assert formatter.equal(TensorFormatter(show_data=True))
    assert formatter_cloned.equal(TensorFormatter())


@torch_available
def test_tensor_formatter_equal_true() -> None:
    assert TensorFormatter().equal(TensorFormatter())


@torch_available
def test_tensor_formatter_equal_false_different_show_data() -> None:
    assert not TensorFormatter().equal(TensorFormatter(show_data=True))


@torch_available
def test_tensor_formatter_equal_false_different_type() -> None:
    assert not TensorFormatter().equal(42)


@torch_available
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float, torch.long, torch.bool])
@pytest.mark.parametrize("device", DEVICES)
def test_tensor_formatter_format(shape: tuple[int, ...], dtype: torch.dtype, device: str) -> None:
    assert (
        TensorFormatter()
        .format(Summarizer(), torch.ones(*shape, device=device, dtype=dtype))
        .startswith("<class 'torch.Tensor'>")
    )


@torch_available
def test_tensor_formatter_format_show_data_false() -> None:
    assert (
        TensorFormatter().format(Summarizer(), torch.ones(2, 3))
        == "<class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | device=cpu"
    )


@torch_available
def test_tensor_formatter_format_show_data_true() -> None:
    assert (
        TensorFormatter(show_data=True).format(Summarizer(), torch.ones(2, 3))
        == "tensor([[1., 1., 1.],\n        [1., 1., 1.]])"
    )


@torch_available
def test_tensor_formatter_load_state_dict() -> None:
    formatter = TensorFormatter()
    formatter.load_state_dict({"show_data": True})
    assert formatter.equal(TensorFormatter(show_data=True))


@torch_available
def test_tensor_formatter_state_dict() -> None:
    assert TensorFormatter().state_dict() == {"show_data": False}


@torch_available
def test_tensor_formatter_get_show_data() -> None:
    assert not TensorFormatter().get_show_data()


@torch_available
@pytest.mark.parametrize("show_data", [True, False])
def test_tensor_formatter_set_show_data_int(show_data: bool) -> None:
    formatter = TensorFormatter()
    assert not formatter.get_show_data()
    formatter.set_show_data(show_data)
    assert formatter.get_show_data() == show_data


@torch_available
def test_tensor_formatter_set_show_data_incorrect_type() -> None:
    formatter = TensorFormatter()
    with pytest.raises(TypeError, match="Incorrect type for show_data. Expected bool value"):
        formatter.set_show_data(4.2)
