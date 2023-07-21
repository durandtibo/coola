from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark, raises

from coola.allclose import AllCloseTester
from coola.equality import EqualityTester
from coola.testing import torch_available
from coola.torch_ import (
    PackedSequenceAllCloseOperator,
    PackedSequenceEqualityOperator,
    TensorAllCloseOperator,
    TensorEqualityOperator,
)
from coola.utils import is_torch_available

if is_torch_available():
    import torch

    cuda_available = mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
    DEVICES = ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",)
else:
    torch = Mock()
    cuda_available = mark.skipif(False, reason="Requires PyTorch and a CUDA device")
    DEVICES = tuple()


####################################################
#     Tests for PackedSequenceAllCloseOperator     #
####################################################


@torch_available
def test_packed_sequence_allclose_operator_str() -> None:
    assert str(PackedSequenceAllCloseOperator()) == "PackedSequenceAllCloseOperator()"


@torch_available
def test_packed_sequence_allclose_operator__eq__true() -> None:
    assert PackedSequenceAllCloseOperator() == PackedSequenceAllCloseOperator()


@torch_available
def test_packed_sequence_allclose_operator__eq__false() -> None:
    assert PackedSequenceAllCloseOperator() != 123


@torch_available
def test_packed_sequence_allclose_operator_allclose_true() -> None:
    assert PackedSequenceAllCloseOperator().allclose(
        AllCloseTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
def test_packed_sequence_allclose_operator_allclose_true_same_obj() -> None:
    obj = torch.nn.utils.rnn.pack_padded_sequence(
        input=torch.arange(10, dtype=torch.float).view(2, 5),
        lengths=torch.tensor([5, 3], dtype=torch.long),
        batch_first=True,
    )
    assert PackedSequenceAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@torch_available
def test_packed_sequence_allclose_operator_allclose_false_different_value() -> None:
    assert not PackedSequenceAllCloseOperator().allclose(
        AllCloseTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10).view(2, 5).add(1).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
def test_packed_sequence_allclose_operator_allclose_false_different_lengths() -> None:
    assert not PackedSequenceAllCloseOperator().allclose(
        AllCloseTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 2], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
def test_packed_sequence_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not PackedSequenceAllCloseOperator().allclose(
            AllCloseTester(),
            torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10).view(2, 5).add(1).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("`torch.nn.utils.rnn.PackedSequence` are different")


@torch_available
def test_packed_sequence_allclose_operator_allclose_false_different_type() -> None:
    assert not PackedSequenceAllCloseOperator().allclose(
        AllCloseTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.arange(10, dtype=torch.float).view(2, 5),
    )


@torch_available
def test_packed_sequence_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not PackedSequenceAllCloseOperator().allclose(
            AllCloseTester(),
            torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            torch.arange(10, dtype=torch.float).view(2, 5),
            show_difference=True,
        )
        assert caplog.messages[0].startswith(
            "object2 is not a `torch.nn.utils.rnn.PackedSequence`:"
        )


@torch_available
@mark.parametrize(
    "tensor,atol",
    (
        (torch.full((2, 3), 1.5), 1),
        (torch.full((2, 3), 1.05), 1e-1),
        (torch.full((2, 3), 1.005), 1e-2),
    ),
)
def test_packed_sequence_allclose_operator_allclose_true_atol(
    tensor: torch.Tensor, atol: float
) -> None:
    assert PackedSequenceAllCloseOperator().allclose(
        AllCloseTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.ones(2, 3),
            lengths=torch.tensor([3, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=tensor,
            lengths=torch.tensor([3, 3], dtype=torch.long),
            batch_first=True,
        ),
        atol=atol,
        rtol=0,
    )


@torch_available
@mark.parametrize(
    "tensor,rtol",
    (
        (torch.full((2, 3), 1.5), 1),
        (torch.full((2, 3), 1.05), 1e-1),
        (torch.full((2, 3), 1.005), 1e-2),
    ),
)
def test_packed_sequence_allclose_operator_allclose_true_rtol(
    tensor: torch.Tensor, rtol: float
) -> None:
    assert PackedSequenceAllCloseOperator().allclose(
        AllCloseTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.ones(2, 3),
            lengths=torch.tensor([3, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=tensor,
            lengths=torch.tensor([3, 3], dtype=torch.long),
            batch_first=True,
        ),
        rtol=rtol,
    )


@torch_available
def test_packed_sequence_allclose_operator_clone() -> None:
    op = PackedSequenceAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
def test_packed_sequence_allclose_operator_no_torch() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`torch` package is required but not installed."):
            PackedSequenceAllCloseOperator()


####################################################
#     Tests for PackedSequenceEqualityOperator     #
####################################################


@torch_available
def test_packed_sequence_equality_operator_str() -> None:
    assert str(PackedSequenceEqualityOperator()) == "PackedSequenceEqualityOperator()"


@torch_available
def test_packed_sequence_equality_operator__eq__true() -> None:
    assert PackedSequenceEqualityOperator() == PackedSequenceEqualityOperator()


@torch_available
def test_packed_sequence_equality_operator__eq__false() -> None:
    assert PackedSequenceEqualityOperator() != 123


@torch_available
def test_packed_sequence_equality_operator_clone() -> None:
    op = PackedSequenceEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
def test_packed_sequence_equality_operator_equal_true() -> None:
    assert PackedSequenceEqualityOperator().equal(
        EqualityTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
def test_packed_sequence_equality_operator_equal_true_same_obj() -> None:
    obj = torch.nn.utils.rnn.pack_padded_sequence(
        input=torch.arange(10, dtype=torch.float).view(2, 5),
        lengths=torch.tensor([5, 3], dtype=torch.long),
        batch_first=True,
    )
    assert PackedSequenceEqualityOperator().equal(EqualityTester(), obj, obj)


@torch_available
def test_packed_sequence_equality_operator_equal_false_different_value() -> None:
    assert not PackedSequenceEqualityOperator().equal(
        EqualityTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10).view(2, 5).add(1).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
def test_packed_sequence_equality_operator_equal_false_different_lengths() -> None:
    assert not PackedSequenceEqualityOperator().equal(
        EqualityTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 2], dtype=torch.long),
            batch_first=True,
        ),
    )


@torch_available
def test_packed_sequence_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not PackedSequenceEqualityOperator().equal(
            EqualityTester(),
            torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10).view(2, 5).add(1).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("`torch.nn.utils.rnn.PackedSequence` are different")


@torch_available
def test_packed_sequence_equality_operator_equal_false_different_type() -> None:
    assert not PackedSequenceEqualityOperator().equal(
        EqualityTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.arange(10, dtype=torch.float).view(2, 5),
    )


@torch_available
def test_packed_sequence_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not PackedSequenceEqualityOperator().equal(
            EqualityTester(),
            torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            torch.arange(10, dtype=torch.float).view(2, 5),
            show_difference=True,
        )
        assert caplog.messages[0].startswith(
            "object2 is not a `torch.nn.utils.rnn.PackedSequence`:"
        )


@torch_available
def test_packed_sequence_equality_operator_no_torch() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`torch` package is required but not installed."):
            PackedSequenceEqualityOperator()


############################################
#     Tests for TensorAllCloseOperator     #
############################################


@torch_available
def test_tensor_allclose_operator_str() -> None:
    assert str(TensorAllCloseOperator()) == "TensorAllCloseOperator()"


@torch_available
def test_tensor_allclose_operator__eq__true() -> None:
    assert TensorAllCloseOperator() == TensorAllCloseOperator()


@torch_available
def test_tensor_allclose_operator__eq__false() -> None:
    assert TensorAllCloseOperator() != 123


@torch_available
@mark.parametrize("device", DEVICES)
@mark.parametrize(
    "tensor", (torch.ones(2, 3), torch.full((2, 3), 1.0 + 1e-9), torch.full((2, 3), 1.0 - 1e-9))
)
def test_tensor_allclose_operator_allclose_true(tensor: torch.Tensor, device: str) -> None:
    device = torch.device(device)
    assert TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3, device=device), tensor.to(device=device)
    )


@torch_available
def test_tensor_allclose_operator_allclose_true_same_obj() -> None:
    obj = torch.ones(2, 3)
    assert TensorAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@torch_available
def test_tensor_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert TensorAllCloseOperator().allclose(
            AllCloseTester(),
            torch.ones(2, 3),
            torch.ones(2, 3),
            show_difference=True,
        )
        assert not caplog.messages


@torch_available
def test_tensor_allclose_operator_allclose_false_nan_equal_nan_false() -> None:
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(),
        torch.tensor([0.0, 1.0, float("nan")]),
        torch.tensor([0.0, 1.0, float("nan")]),
    )


@torch_available
def test_tensor_allclose_operator_allclose_true_nan_equal_nan_false() -> None:
    assert TensorAllCloseOperator().allclose(
        AllCloseTester(),
        torch.tensor([0.0, 1.0, float("nan")]),
        torch.tensor([0.0, 1.0, float("nan")]),
        equal_nan=True,
    )


@torch_available
def test_tensor_allclose_operator_allclose_false_different_value() -> None:
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3), torch.zeros(2, 3)
    )


@torch_available
def test_tensor_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            AllCloseTester(),
            torch.ones(2, 3),
            torch.zeros(2, 3),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensors are different")


@torch_available
def test_tensor_allclose_operator_allclose_false_different_dtype() -> None:
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3, dtype=torch.float), torch.ones(2, 3, dtype=torch.long)
    )


@torch_available
def test_tensor_allclose_operator_allclose_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            AllCloseTester(),
            torch.ones(2, 3, dtype=torch.float),
            torch.ones(2, 3, dtype=torch.long),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor data types are different:")


@torch_available
@cuda_available
def test_tensor_allclose_operator_equal_false_different_device() -> None:
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(),
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.ones(2, 3, device=torch.device("cuda:0")),
    )


@torch_available
def test_tensor_allclose_operator_equal_false_different_device_mock() -> None:
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(),
        Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cpu")),
        Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cuda:0")),
    )


@torch_available
def test_tensor_allclose_operator_equal_false_different_device_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            AllCloseTester(),
            Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cpu")),
            Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cuda:0")),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor devices are different:")


@torch_available
def test_tensor_allclose_operator_allclose_false_different_shape() -> None:
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3), torch.ones(2, 4)
    )


@torch_available
def test_tensor_allclose_operator_allclose_false_different_shape_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            AllCloseTester(),
            torch.ones(2, 3),
            torch.ones(2, 4),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor shapes are different:")


@torch_available
def test_tensor_allclose_operator_allclose_false_different_type() -> None:
    assert not TensorAllCloseOperator().allclose(AllCloseTester(), torch.ones(2, 3), 42)


@torch_available
def test_tensor_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            AllCloseTester(),
            torch.ones(2, 3),
            42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a torch.Tensor:")


@torch_available
@mark.parametrize(
    "tensor,atol",
    (
        (torch.full((2, 3), 1.5), 1),
        (torch.full((2, 3), 1.05), 1e-1),
        (torch.full((2, 3), 1.005), 1e-2),
    ),
)
def test_tensor_allclose_operator_allclose_true_atol(tensor: torch.Tensor, atol: float) -> None:
    assert TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3), tensor, atol=atol, rtol=0
    )


@torch_available
@mark.parametrize(
    "tensor,rtol",
    (
        (torch.full((2, 3), 1.5), 1),
        (torch.full((2, 3), 1.05), 1e-1),
        (torch.full((2, 3), 1.005), 1e-2),
    ),
)
def test_tensor_allclose_operator_allclose_true_rtol(tensor: torch.Tensor, rtol: float) -> None:
    assert TensorAllCloseOperator().allclose(AllCloseTester(), torch.ones(2, 3), tensor, rtol=rtol)


@torch_available
def test_tensor_allclose_operator_clone() -> None:
    op = TensorAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
def test_tensor_allclose_operator_no_torch() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`torch` package is required but not installed."):
            TensorAllCloseOperator()


############################################
#     Tests for TensorEqualityOperator     #
############################################


@torch_available
def test_tensor_equality_operator_str() -> None:
    assert str(TensorEqualityOperator()) == "TensorEqualityOperator()"


@torch_available
def test_tensor_equality_operator__eq__true() -> None:
    assert TensorEqualityOperator() == TensorEqualityOperator()


@torch_available
def test_tensor_equality_operator__eq__false() -> None:
    assert TensorEqualityOperator() != 123


@torch_available
def test_tensor_equality_operator_clone() -> None:
    op = TensorEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
@mark.parametrize("device", DEVICES)
def test_tensor_equality_operator_equal_true(device: str) -> None:
    device = torch.device(device)
    assert TensorEqualityOperator().equal(
        EqualityTester(),
        torch.ones(2, 3, device=device),
        torch.ones(2, 3, device=device),
    )


@torch_available
def test_tensor_equality_operator_equal_true_same_object() -> None:
    obj = torch.ones(2, 3)
    assert TensorEqualityOperator().equal(EqualityTester(), obj, obj)


@torch_available
def test_tensor_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert TensorEqualityOperator().equal(
            EqualityTester(),
            torch.ones(2, 3),
            torch.ones(2, 3),
            show_difference=True,
        )
        assert not caplog.messages


@torch_available
def test_tensor_equality_operator_equal_false_different_value() -> None:
    assert not TensorEqualityOperator().equal(EqualityTester(), torch.ones(2, 3), torch.zeros(2, 3))


@torch_available
def test_tensor_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            EqualityTester(),
            torch.ones(2, 3),
            torch.zeros(2, 3),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensors are different")


@torch_available
def test_tensor_equality_operator_equal_false_different_dtype() -> None:
    assert not TensorEqualityOperator().equal(
        EqualityTester(),
        torch.ones(2, 3, dtype=torch.float),
        torch.ones(2, 3, dtype=torch.long),
    )


@torch_available
def test_tensor_equality_operator_equal_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            EqualityTester(),
            torch.ones(2, 3, dtype=torch.float),
            torch.ones(2, 3, dtype=torch.long),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor data types are different:")


@torch_available
@cuda_available
def test_tensor_equality_operator_equal_false_different_device() -> None:
    assert not TensorEqualityOperator().equal(
        EqualityTester(),
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.ones(2, 3, device=torch.device("cuda:0")),
    )


@torch_available
def test_tensor_equality_operator_equal_false_different_device_mock() -> None:
    assert not TensorEqualityOperator().equal(
        EqualityTester(),
        Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cpu")),
        Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cuda:0")),
    )


@torch_available
def test_tensor_equality_operator_equal_false_different_device_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            EqualityTester(),
            Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cpu")),
            Mock(spec=torch.Tensor, dtype=torch.float, device=torch.device("cuda:0")),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor devices are different:")


@torch_available
def test_tensor_equality_operator_equal_false_different_shape() -> None:
    assert not TensorEqualityOperator().equal(EqualityTester(), torch.ones(2, 3), torch.ones(2, 4))


@torch_available
def test_tensor_equality_operator_equal_false_different_shape_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            EqualityTester(),
            torch.ones(2, 3),
            torch.ones(2, 4),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensors are different")


@torch_available
def test_tensor_equality_operator_equal_false_different_type() -> None:
    assert not TensorEqualityOperator().equal(EqualityTester(), torch.ones(2, 3), 42)


@torch_available
def test_tensor_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            EqualityTester(),
            torch.ones(2, 3),
            42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a torch.Tensor:")


@torch_available
def test_tensor_equality_operator_no_torch() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`torch` package is required but not installed."):
            TensorEqualityOperator()
