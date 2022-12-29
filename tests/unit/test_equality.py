import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Union
from unittest.mock import Mock, patch

import numpy as np
import torch
from pytest import LogCaptureFixture, mark, raises
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from coola.equality import (
    BaseEqualityOperator,
    DefaultEqualityOperator,
    EqualityTester,
    MappingEqualityOperator,
    SequenceEqualityOperator,
    objects_are_equal,
)
from coola.ndarray import NDArrayEqualityOperator
from coola.pytorch import PackedSequenceEqualityOperator, TensorEqualityOperator

####################################
#     Tests for EqualityTester     #
####################################


def test_equality_tester_str():
    assert str(EqualityTester()).startswith("EqualityTester(")


def test_equality_tester_registry_default():
    assert len(EqualityTester.registry) == 9
    assert isinstance(EqualityTester.registry[Mapping], MappingEqualityOperator)
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


def test_objects_are_equal_false_different_type():
    assert not objects_are_equal([], ())


@mark.parametrize(
    "object1,object2",
    (
        (1, 1),
        (0, 0),
        (-1, -1),
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, -1.0),
        (True, True),
        (False, False),
    ),
)
def test_objects_are_equal_scalar_true(
    object1: Union[bool, int, float], object2: Union[bool, int, float]
):
    assert objects_are_equal(object1, object2)


@mark.parametrize(
    "object1,object2",
    (
        (1, 2),
        (1.0, 2.0),
        (True, False),
        (1, 1.0),
        (1, True),
        (1.0, True),
    ),
)
def test_objects_are_equal_scalar_false(
    object1: Union[bool, int, float], object2: Union[bool, int, float]
):
    assert not objects_are_equal(object1, object2)


def test_objects_are_equal_torch_tensor_true():
    assert objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))


def test_objects_are_equal_torch_tensor_false():
    assert not objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3))


def test_objects_are_equal_numpy_array_true():
    assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)))


def test_objects_are_equal_numpy_array_false():
    assert not objects_are_equal(np.ones((2, 3)), np.zeros((2, 3)))


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


def test_objects_are_equal_true_complex_objects():
    assert objects_are_equal(
        {
            "list": [1, 2.0, torch.arange(5), np.arange(3), [1, 2, 3]],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5,
            "torch": torch.ones(5),
            "numpy": np.ones(4),
        },
        {
            "list": [1, 2.0, torch.arange(5), np.arange(3), [1, 2, 3]],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5,
            "torch": torch.ones(5),
            "numpy": np.ones(4),
        },
    )


#############################################
#     Tests for DefaultEqualityOperator     #
#############################################


def test_default_equality_operator_str():
    assert str(DefaultEqualityOperator()) == "DefaultEqualityOperator()"


@mark.parametrize(
    "object1,object2",
    (
        (1, 1),
        (0, 0),
        (-1, -1),
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, -1.0),
        (True, True),
        (False, False),
    ),
)
def test_default_equality_operator_equal_true_scalar(
    object1: Union[bool, int, float], object2: Union[bool, int, float]
):
    assert DefaultEqualityOperator().equal(EqualityTester(), object1, object2)


@mark.parametrize(
    "object1,object2",
    (
        (1, 2),
        (1.0, 2.0),
        (True, False),
        (1, 1.0),
        (1, True),
        (1.0, True),
    ),
)
def test_default_equality_operator_equal_false_scalar(
    object1: Union[bool, int, float], object2: Union[bool, int, float]
):
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
        assert caplog.messages[-1].startswith(
            "The mappings have a different value for the key '2':"
        )


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
