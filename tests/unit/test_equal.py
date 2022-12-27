import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import torch
from pytest import LogCaptureFixture, mark, raises
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from coola import (
    BaseEqualityOperator,
    DefaultEqualityOperator,
    EqualityTester,
    MappingEqualityOperator,
    NDArrayEqualityOperator,
    PackedSequenceEqualityOperator,
    SequenceEqualityOperator,
    TensorEqualityOperator,
    objects_are_equal,
)

####################################
#     Tests for EqualityTester     #
####################################


def test_equality_tester_str():
    assert str(EqualityTester()).startswith("EqualityTester(")


def test_equality_tester_registry_default():
    assert len(EqualityTester.registry) == 8
    assert isinstance(EqualityTester.registry[PackedSequence], PackedSequenceEqualityOperator)
    assert isinstance(EqualityTester.registry[Sequence], SequenceEqualityOperator)
    assert isinstance(EqualityTester.registry[Tensor], TensorEqualityOperator)
    assert isinstance(EqualityTester.registry[dict], MappingEqualityOperator)
    assert isinstance(EqualityTester.registry[list], SequenceEqualityOperator)
    assert isinstance(EqualityTester.registry[np.ndarray], NDArrayEqualityOperator)
    assert isinstance(EqualityTester.registry[object], DefaultEqualityOperator)
    assert isinstance(EqualityTester.registry[tuple], SequenceEqualityOperator)


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_equality_operator():
    tester = EqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_equality_operator(int, operator)
    assert tester.registry[int] == operator


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_equality_operator_duplicate_exist_ok_true():
    tester = EqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_equality_operator(int, Mock(spec=BaseEqualityOperator))
    tester.add_equality_operator(int, operator, exist_ok=True)
    assert tester.registry[int] == operator


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_equality_operator_duplicate_exist_ok_false():
    tester = EqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_equality_operator(int, Mock(spec=BaseEqualityOperator))
    with raises(ValueError):
        tester.add_equality_operator(int, operator)


def test_equality_tester_has_equality_operator_true():
    assert EqualityTester().has_equality_operator(dict)


def test_equality_tester_has_equality_operator_false():
    assert not EqualityTester().has_equality_operator(int)


def test_equality_tester_find_equality_operator_direct():
    assert isinstance(EqualityTester().find_equality_operator(dict), MappingEqualityOperator)


def test_equality_tester_find_equality_operator_indirect():
    assert isinstance(EqualityTester().find_equality_operator(str), DefaultEqualityOperator)


def test_equality_tester_find_equality_operator_incorrect_type():
    with raises(TypeError):
        EqualityTester().find_equality_operator(Mock(__mro__=[]))


#######################################
#     Tests for objects_are_equal     #
#######################################


def test_objects_are_equal_different_type():
    assert not objects_are_equal([], ())


def test_objects_are_equal_different_type_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal([], (), show_difference=True)
        assert caplog.messages[0].startswith("The sequences have different types:")


@mark.parametrize("object1,object2", ((1, 1), (2.5, 2.5)))
def test_objects_are_equal_scalar_true(object1: Number, object2: Number):
    assert objects_are_equal(object1, object2)


def test_objects_are_equal_scalar_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert objects_are_equal(1, 1, show_difference=True)
        assert not caplog.messages


@mark.parametrize("object1,object2", ((1, 1.1), (2.5, 0)))
def test_objects_are_equal_scalar_false(object1: Number, object2: Number):
    assert not objects_are_equal(object1, object2)


def test_objects_are_equal_scalar_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(1, 2, show_difference=True)
        assert caplog.messages[0].startswith("Objects are different:")


def test_objects_are_equal_torch_tensor_true():
    assert objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))


def test_objects_are_equal_torch_tensor_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert objects_are_equal(torch.ones(2, 3), torch.ones(2, 3), show_difference=True)
        assert not caplog.messages


def test_objects_are_equal_torch_tensor_false():
    assert not objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3))


def test_objects_are_equal_torch_tensor_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3), show_difference=True)
        assert caplog.messages[0].startswith("torch.Tensors are different")


def test_objects_are_equal_numpy_array_true():
    assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)))


def test_objects_are_equal_numpy_array_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)), show_difference=True)
        assert not caplog.messages


def test_objects_are_equal_numpy_array_false():
    assert not objects_are_equal(np.ones((2, 3)), np.zeros((2, 3)))


def test_objects_are_equal_numpy_array_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(np.ones((2, 3)), np.zeros((2, 3)), show_difference=True)
        assert caplog.messages[0].startswith("numpy.arrays are different")


@mark.parametrize(
    "object1,object2",
    (
        ([], []),
        ((), ()),
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.zeros(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.zeros(2))),
        (
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
        ),
    ),
)
def test_objects_are_equal_sequence_true(object1: Sequence, object2: Sequence):
    assert objects_are_equal(object1, object2)


def test_objects_are_equal_sequence_false():
    assert not objects_are_equal([1, 2], [1, 3])


def test_objects_are_equal_sequence_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal([1, 2], [1, 3], show_difference=True)
        assert caplog.messages


@mark.parametrize(
    "object1,object2",
    (
        ({}, {}),
        (
            {"1": torch.ones(2, 3), "2": torch.zeros(2)},
            {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        ),
        (
            {"1": torch.ones(2, 3), "2": {"3": torch.zeros(2), "4": torch.ones(2)}},
            {"1": torch.ones(2, 3), "2": {"3": torch.zeros(2), "4": torch.ones(2)}},
        ),
        (
            OrderedDict([("1", torch.ones(2, 3)), ("2", torch.zeros(2))]),
            OrderedDict([("1", torch.ones(2, 3)), ("2", torch.zeros(2))]),
        ),
    ),
)
def test_objects_are_equal_mapping_true(object1: Mapping, object2: Mapping):
    assert objects_are_equal(object1, object2)


def test_objects_are_equal_mapping_false():
    assert not objects_are_equal({"abc": 1, "def": 2}, {"abc": 1, "def": 3})


def test_objects_are_equal_mapping_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(
            {"abc": 1, "def": 2}, {"abc": 1, "def": 3}, show_difference=True
        )
        assert caplog.messages


@mark.parametrize(
    "object1,object2",
    (
        ("abc", "abc"),
        (set(), set()),
        ({1, 2, 3}, {1, 2, 3}),
    ),
)
def test_objects_are_equal_other_types_true(object1: Any, object2: Any):
    assert objects_are_equal(object1, object2)


@mark.parametrize(
    "object1,object2",
    (
        ("abc", "abcd"),
        (set(), tuple()),
        ({1, 2}, {1, 2, 3}),
        ({1, 2, 4}, {1, 2, 3}),
    ),
)
def test_objects_are_equal_other_types_false(object1: Any, object2: Any):
    assert not objects_are_equal(object1, object2)


#############################################
#     Tests for DefaultEqualityOperator     #
#############################################


def test_default_equality_operator_str():
    assert str(DefaultEqualityOperator()) == "DefaultEqualityOperator()"


@mark.parametrize("object1,object2", ((1, 1), (2.5, 2.5)))
def test_default_equality_operator_equal_true_scalar(object1: Number, object2: Number):
    assert DefaultEqualityOperator().equal(EqualityTester(), object1, object2)


@mark.parametrize("object1,object2", ((1, 1.1), (2.5, 0)))
def test_default_equality_operator_equal_false_scalar(object1: Number, object2: Number):
    assert not DefaultEqualityOperator().equal(EqualityTester(), object1, object2)


def test_default_equality_operator_equal_different_value_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not DefaultEqualityOperator().equal(
            tester=EqualityTester(), object1=1, object2=2, show_difference=True
        )
        assert caplog.messages[0].startswith("Objects are different:")


def test_default_equality_operator_equal_different_type():
    assert not DefaultEqualityOperator().equal(EqualityTester(), [], ())


def test_default_equality_operator_equal_different_type_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not DefaultEqualityOperator().equal(EqualityTester(), [], (), show_difference=True)
        assert caplog.messages[0].startswith("Objects have different types:")


#############################################
#     Tests for MappingEqualityOperator     #
#############################################


def test_mapping_equality_operator_str():
    assert str(MappingEqualityOperator()) == "MappingEqualityOperator()"


@mark.parametrize(
    "object1,object2",
    (
        ({}, {}),
        (
            {"1": torch.ones(2, 3), "2": torch.zeros(2)},
            {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        ),
        (
            {"1": torch.ones(2, 3), "2": {"3": torch.zeros(2), "4": torch.ones(2)}},
            {"1": torch.ones(2, 3), "2": {"3": torch.zeros(2), "4": torch.ones(2)}},
        ),
        (
            OrderedDict([("1", torch.ones(2, 3)), ("2", torch.zeros(2))]),
            OrderedDict([("1", torch.ones(2, 3)), ("2", torch.zeros(2))]),
        ),
    ),
)
def test_mapping_equality_operator_equal_true(object1: Mapping, object2: Mapping):
    assert MappingEqualityOperator().equal(EqualityTester(), object1, object2)


def test_mapping_equality_operator_equal_false_different_value():
    assert not MappingEqualityOperator().equal(
        EqualityTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"1": torch.ones(2, 3), "2": torch.ones(2)},
    )


def test_mapping_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not MappingEqualityOperator().equal(
            tester=EqualityTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"1": torch.ones(2, 3), "2": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("The mappings have a different value for the key 2:")


def test_mapping_equality_operator_equal_false_different_keys():
    assert not MappingEqualityOperator().equal(
        EqualityTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"10": torch.ones(2, 3), "20": torch.ones(2)},
    )


def test_mapping_equality_operator_equal_false_different_keys_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not MappingEqualityOperator().equal(
            tester=EqualityTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"10": torch.ones(2, 3), "20": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The mappings have different keys:")


def test_mapping_equality_operator_equal_false_different_length():
    assert not MappingEqualityOperator().equal(
        EqualityTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"1": torch.ones(2, 3), "2": torch.zeros(2), "3": torch.ones(2)},
    )


def test_mapping_equality_operator_equal_false_different_length_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not MappingEqualityOperator().equal(
            tester=EqualityTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"1": torch.ones(2, 3), "2": torch.zeros(2), "3": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The mappings have different sizes:")


def test_mapping_equality_operator_equal_false_different_type():
    assert not MappingEqualityOperator().equal(EqualityTester(), {}, OrderedDict([]))


def test_mapping_equality_operator_equal_different_type_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not MappingEqualityOperator().equal(
            tester=EqualityTester(), object1={}, object2=OrderedDict([]), show_difference=True
        )
        assert caplog.messages[0].startswith("The mappings have different types:")


#############################################
#     Tests for NDArrayEqualityOperator     #
#############################################


def test_ndarray_equality_operator_str():
    assert str(NDArrayEqualityOperator()) == "NDArrayEqualityOperator()"


def test_ndarray_equality_operator_equal_true():
    assert NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.ones((2, 3)))


def test_ndarray_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


def test_ndarray_equality_operator_equal_false_different_value():
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.zeros((2, 3)))


def test_ndarray_equality_operator_equal_false_different_type():
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), 42)


def test_ndarray_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.arrays are different")


##############################################
#     Tests for SequenceEqualityOperator     #
##############################################


def test_sequence_equality_operator_str():
    assert str(SequenceEqualityOperator()) == "SequenceEqualityOperator()"


@mark.parametrize(
    "object1,object2",
    (
        ([], []),
        ((), ()),
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), (1, 2, 3)),
        (["abc", "def"], ["abc", "def"]),
        (("abc", "def"), ("abc", "def")),
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.zeros(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.zeros(2))),
        (
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
        ),
    ),
)
def test_sequence_equality_operator_equal_true(object1: Sequence, object2: Sequence):
    assert SequenceEqualityOperator().equal(EqualityTester(), object1, object2)


@mark.parametrize(
    "object1,object2",
    (
        (["abc", "deg"], ["abc", "def"]),
        (("abc", "deg"), ("abc", "def")),
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.ones(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.ones(2))),
    ),
)
def test_sequence_equality_operator_equal_false_different_value(
    object1: Sequence, object2: Sequence
):
    assert not SequenceEqualityOperator().equal(EqualityTester(), object1, object2)


def test_sequence_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not SequenceEqualityOperator().equal(
            tester=EqualityTester(),
            object1=[1, 2],
            object2=[1, 3],
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("The sequences have at least one different value:")


@mark.parametrize(
    "object1,object2",
    (
        (["abc", "defg"], ["abc", "def"]),
        (("abc", "defg"), ("abc", "def")),
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.ones(2), torch.ones(3)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.ones(2), torch.ones(3))),
    ),
)
def test_sequence_equality_operator_equal_false_different_length(
    object1: Sequence, object2: Sequence
):
    assert not SequenceEqualityOperator().equal(EqualityTester(), object1, object2)


def test_sequence_equality_operator_equal_false_different_length_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not SequenceEqualityOperator().equal(
            tester=EqualityTester(),
            object1=[1, 2, 3],
            object2=[1, 2],
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The sequences have different sizes:")


def test_sequence_equality_operator_equal_false_different_type():
    assert not SequenceEqualityOperator().equal(EqualityTester(), [], ())


def test_sequence_equality_operator_equal_different_type_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not SequenceEqualityOperator().equal(EqualityTester(), [], (), show_difference=True)
        assert caplog.messages[0].startswith("The sequences have different types:")


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
