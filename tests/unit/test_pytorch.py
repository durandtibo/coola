import logging

import torch
from pytest import LogCaptureFixture
from torch.nn.utils.rnn import pack_padded_sequence

from coola.equal import EqualityTester
from coola.pytorch import PackedSequenceEqualityOperator, TensorEqualityOperator

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


############################################
#     Tests for TensorEqualityOperator     #
############################################


def test_tensor_equality_operator_str():
    assert str(TensorEqualityOperator()) == "TensorEqualityOperator()"


def test_tensor_equality_operator_equal_true():
    assert TensorEqualityOperator().equal(EqualityTester(), torch.ones(2, 3), torch.ones(2, 3))


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
