import logging
from unittest.mock import Mock, patch

import torch
from pytest import LogCaptureFixture, mark, raises
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from coola.allclose import AllCloseTester
from coola.equality import EqualityTester
from coola.pytorch import (
    PackedSequenceAllCloseOperator,
    PackedSequenceEqualityOperator,
    TensorAllCloseOperator,
    TensorEqualityOperator,
)

cuda_available = mark.skipif(not torch.cuda.is_available(), reason="Requires a device with CUDA")

DEVICES = ("cpu", "cuda:0") if torch.cuda.is_available() else ("cpu",)


####################################################
#     Tests for PackedSequenceAllCloseOperator     #
####################################################


def test_packed_sequence_allclose_operator_str():
    assert str(PackedSequenceAllCloseOperator()) == "PackedSequenceAllCloseOperator()"


def test_packed_sequence_allclose_operator_allclose_true():
    assert PackedSequenceAllCloseOperator().allclose(
        tester=AllCloseTester(),
        object1=pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        object2=pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


def test_packed_sequence_allclose_operator_allclose_false_different_value():
    assert not PackedSequenceAllCloseOperator().allclose(
        tester=AllCloseTester(),
        object1=pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        object2=pack_padded_sequence(
            input=torch.arange(10).view(2, 5).add(1).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


def test_packed_sequence_allclose_operator_allclose_false_different_lengths():
    assert not PackedSequenceAllCloseOperator().allclose(
        tester=AllCloseTester(),
        object1=pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        object2=pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 2], dtype=torch.long),
            batch_first=True,
        ),
    )


def test_packed_sequence_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not PackedSequenceAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=pack_padded_sequence(
                input=torch.arange(10).view(2, 5).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            object2=pack_padded_sequence(
                input=torch.arange(10).view(2, 5).add(1).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("`torch.nn.utils.rnn.PackedSequence` are different")


def test_packed_sequence_allclose_operator_allclose_false_different_type():
    assert not PackedSequenceAllCloseOperator().allclose(
        tester=AllCloseTester(),
        object1=pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        object2=torch.arange(10).view(2, 5).float(),
    )


def test_packed_sequence_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not PackedSequenceAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=pack_padded_sequence(
                input=torch.arange(10).view(2, 5).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            object2=torch.arange(10).view(2, 5).float(),
            show_difference=True,
        )
        assert caplog.messages[0].startswith(
            "object2 is not a `torch.nn.utils.rnn.PackedSequence`:"
        )


@mark.parametrize(
    "tensor,atol",
    ((torch.ones(2, 3) + 0.5, 1), (torch.ones(2, 3) + 0.05, 1e-1), (torch.ones(2, 3) + 5e-3, 1e-2)),
)
def test_packed_sequence_allclose_operator_allclose_true_atol(tensor: Tensor, atol: float):
    assert PackedSequenceAllCloseOperator().allclose(
        tester=AllCloseTester(),
        object1=pack_padded_sequence(
            input=torch.ones(2, 3),
            lengths=torch.tensor([3, 3], dtype=torch.long),
            batch_first=True,
        ),
        object2=pack_padded_sequence(
            input=tensor,
            lengths=torch.tensor([3, 3], dtype=torch.long),
            batch_first=True,
        ),
        atol=atol,
        rtol=0,
    )


@mark.parametrize(
    "tensor,rtol",
    ((torch.ones(2, 3) + 0.5, 1), (torch.ones(2, 3) + 0.05, 1e-1), (torch.ones(2, 3) + 5e-3, 1e-2)),
)
def test_packed_sequence_allclose_operator_allclose_true_rtol(tensor: Tensor, rtol: float):
    assert PackedSequenceAllCloseOperator().allclose(
        tester=AllCloseTester(),
        object1=pack_padded_sequence(
            input=torch.ones(2, 3),
            lengths=torch.tensor([3, 3], dtype=torch.long),
            batch_first=True,
        ),
        object2=pack_padded_sequence(
            input=tensor,
            lengths=torch.tensor([3, 3], dtype=torch.long),
            batch_first=True,
        ),
        rtol=rtol,
    )


def test_packed_sequence_allclose_operator_no_torch():
    with patch("coola.pytorch.is_torch_available", lambda *args, **kwargs: False):
        with raises(RuntimeError):
            PackedSequenceAllCloseOperator()


####################################################
#     Tests for PackedSequenceEqualityOperator     #
####################################################


def test_packed_sequence_equality_operator_str():
    assert str(PackedSequenceEqualityOperator()) == "PackedSequenceEqualityOperator()"


def test_packed_sequence_equality_operator_equal_true():
    assert PackedSequenceEqualityOperator().equal(
        EqualityTester(),
        pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


def test_packed_sequence_equality_operator_equal_false_different_value():
    assert not PackedSequenceEqualityOperator().equal(
        EqualityTester(),
        pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        pack_padded_sequence(
            input=torch.arange(10).view(2, 5).add(1).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


def test_packed_sequence_equality_operator_equal_false_different_lengths():
    assert not PackedSequenceEqualityOperator().equal(
        EqualityTester(),
        pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 2], dtype=torch.long),
            batch_first=True,
        ),
    )


def test_packed_sequence_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not PackedSequenceEqualityOperator().equal(
            EqualityTester(),
            pack_padded_sequence(
                input=torch.arange(10).view(2, 5).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            pack_padded_sequence(
                input=torch.arange(10).view(2, 5).add(1).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("`torch.nn.utils.rnn.PackedSequence` are different")


def test_packed_sequence_equality_operator_equal_false_different_type():
    assert not PackedSequenceEqualityOperator().equal(
        EqualityTester(),
        pack_padded_sequence(
            input=torch.arange(10).view(2, 5).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.arange(10).view(2, 5).float(),
    )


def test_packed_sequence_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not PackedSequenceEqualityOperator().equal(
            EqualityTester(),
            pack_padded_sequence(
                input=torch.arange(10).view(2, 5).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            torch.arange(10).view(2, 5).float(),
            show_difference=True,
        )
        assert caplog.messages[0].startswith(
            "object2 is not a `torch.nn.utils.rnn.PackedSequence`:"
        )


def test_packed_sequence_equality_operator_no_torch():
    with patch("coola.pytorch.is_torch_available", lambda *args, **kwargs: False):
        with raises(RuntimeError):
            PackedSequenceEqualityOperator()


############################################
#     Tests for TensorAllCloseOperator     #
############################################


def test_tensor_allclose_operator_str():
    assert str(TensorAllCloseOperator()) == "TensorAllCloseOperator()"


@mark.parametrize("device", DEVICES)
@mark.parametrize("tensor", (torch.ones(2, 3), torch.ones(2, 3) + 1e-9, torch.ones(2, 3) - 1e-9))
def test_tensor_allclose_operator_allclose_true(tensor: Tensor, device: str):
    device = torch.device(device)
    assert TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3, device=device), tensor.to(device=device)
    )


def test_tensor_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert TensorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=torch.ones(2, 3),
            object2=torch.ones(2, 3),
            show_difference=True,
        )
        assert not caplog.messages


def test_tensor_allclose_operator_allclose_false_different_value():
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3), torch.zeros(2, 3)
    )


def test_tensor_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=torch.ones(2, 3),
            object2=torch.zeros(2, 3),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensors are different")


def test_tensor_allclose_operator_allclose_false_different_dtype():
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3, dtype=torch.float), torch.ones(2, 3, dtype=torch.long)
    )


def test_tensor_allclose_operator_allclose_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=torch.ones(2, 3, dtype=torch.float),
            object2=torch.ones(2, 3, dtype=torch.long),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor data types are different:")


@cuda_available
def test_tensor_allclose_operator_equal_false_different_device():
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(),
        object1=torch.ones(2, 3, device=torch.device("cpu")),
        object2=torch.ones(2, 3, device=torch.device("cuda:0")),
    )


def test_tensor_allclose_operator_equal_false_different_device_mock():
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(),
        Mock(spec=Tensor, dtype=torch.float, device=torch.device("cpu")),
        Mock(spec=Tensor, dtype=torch.float, device=torch.device("cuda:0")),
    )


def test_tensor_allclose_operator_equal_false_different_device_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=Mock(spec=Tensor, dtype=torch.float, device=torch.device("cpu")),
            object2=Mock(spec=Tensor, dtype=torch.float, device=torch.device("cuda:0")),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor devices are different:")


def test_tensor_allclose_operator_allclose_false_different_shape():
    assert not TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3), torch.ones(2, 4)
    )


def test_tensor_allclose_operator_allclose_false_different_shape_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=torch.ones(2, 3),
            object2=torch.ones(2, 4),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor shapes are different:")


def test_tensor_allclose_operator_allclose_false_different_type():
    assert not TensorAllCloseOperator().allclose(AllCloseTester(), torch.ones(2, 3), 42)


def test_tensor_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=torch.ones(2, 3),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a torch.Tensor:")


@mark.parametrize(
    "tensor,atol",
    ((torch.ones(2, 3) + 0.5, 1), (torch.ones(2, 3) + 0.05, 1e-1), (torch.ones(2, 3) + 5e-3, 1e-2)),
)
def test_tensor_allclose_operator_allclose_true_atol(tensor: Tensor, atol: float):
    assert TensorAllCloseOperator().allclose(
        AllCloseTester(), torch.ones(2, 3), tensor, atol=atol, rtol=0
    )


@mark.parametrize(
    "tensor,rtol",
    ((torch.ones(2, 3) + 0.5, 1), (torch.ones(2, 3) + 0.05, 1e-1), (torch.ones(2, 3) + 5e-3, 1e-2)),
)
def test_tensor_allclose_operator_allclose_true_rtol(tensor: Tensor, rtol: float):
    assert TensorAllCloseOperator().allclose(AllCloseTester(), torch.ones(2, 3), tensor, rtol=rtol)


def test_tensor_allclose_operator_no_torch():
    with patch("coola.pytorch.is_torch_available", lambda *args, **kwargs: False):
        with raises(RuntimeError):
            TensorAllCloseOperator()


############################################
#     Tests for TensorEqualityOperator     #
############################################


def test_tensor_equality_operator_str():
    assert str(TensorEqualityOperator()) == "TensorEqualityOperator()"


@mark.parametrize("device", DEVICES)
def test_tensor_equality_operator_equal_true(device: str):
    device = torch.device(device)
    assert TensorEqualityOperator().equal(
        EqualityTester(),
        torch.ones(2, 3, device=device),
        torch.ones(2, 3, device=device),
    )


def test_tensor_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert TensorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=torch.ones(2, 3),
            object2=torch.ones(2, 3),
            show_difference=True,
        )
        assert not caplog.messages


def test_tensor_equality_operator_equal_false_different_value():
    assert not TensorEqualityOperator().equal(EqualityTester(), torch.ones(2, 3), torch.zeros(2, 3))


def test_tensor_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=torch.ones(2, 3),
            object2=torch.zeros(2, 3),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensors are different")


def test_tensor_equality_operator_equal_false_different_dtype():
    assert not TensorEqualityOperator().equal(
        EqualityTester(),
        torch.ones(2, 3, dtype=torch.float),
        torch.ones(2, 3, dtype=torch.long),
    )


def test_tensor_equality_operator_equal_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=torch.ones(2, 3, dtype=torch.float),
            object2=torch.ones(2, 3, dtype=torch.long),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor data types are different:")


@cuda_available
def test_tensor_equality_operator_equal_false_different_device():
    assert not TensorEqualityOperator().equal(
        EqualityTester(),
        object1=torch.ones(2, 3, device=torch.device("cpu")),
        object2=torch.ones(2, 3, device=torch.device("cuda:0")),
    )


def test_tensor_equality_operator_equal_false_different_device_mock():
    assert not TensorEqualityOperator().equal(
        EqualityTester(),
        Mock(spec=Tensor, dtype=torch.float, device=torch.device("cpu")),
        Mock(spec=Tensor, dtype=torch.float, device=torch.device("cuda:0")),
    )


def test_tensor_equality_operator_equal_false_different_device_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=Mock(spec=Tensor, dtype=torch.float, device=torch.device("cpu")),
            object2=Mock(spec=Tensor, dtype=torch.float, device=torch.device("cuda:0")),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensor devices are different:")


def test_tensor_equality_operator_equal_false_different_shape():
    assert not TensorEqualityOperator().equal(EqualityTester(), torch.ones(2, 3), torch.ones(2, 4))


def test_tensor_equality_operator_equal_false_different_shape_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=torch.ones(2, 3),
            object2=torch.ones(2, 4),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("torch.Tensors are different")


def test_tensor_equality_operator_equal_false_different_type():
    assert not TensorEqualityOperator().equal(EqualityTester(), torch.ones(2, 3), 42)


def test_tensor_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not TensorEqualityOperator().equal(
            tester=EqualityTester(),
            object1=torch.ones(2, 3),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a torch.Tensor:")


def test_tensor_equality_operator_no_torch():
    with patch("coola.pytorch.is_torch_available", lambda *args, **kwargs: False):
        with raises(RuntimeError):
            TensorEqualityOperator()
