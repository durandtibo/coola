import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Union
from unittest.mock import Mock, patch

import numpy as np
import torch
from pytest import LogCaptureFixture, mark, raises
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from coola.allclose import (
    AllCloseTester,
    BaseAllCloseOperator,
    DefaultAllCloseOperator,
    MappingAllCloseOperator,
    ScalarAllCloseOperator,
    SequenceAllCloseOperator,
    objects_are_allclose,
)
from coola.ndarray import NDArrayAllCloseOperator
from coola.pytorch import PackedSequenceAllCloseOperator, TensorAllCloseOperator

####################################
#     Tests for AllCloseTester     #
####################################


def test_allclose_tester_str():
    assert str(AllCloseTester()).startswith("AllCloseTester(")


def test_allclose_tester_registry_default():
    assert len(AllCloseTester.registry) == 12
    assert isinstance(AllCloseTester.registry[Mapping], MappingAllCloseOperator)
    assert isinstance(AllCloseTester.registry[PackedSequence], PackedSequenceAllCloseOperator)
    assert isinstance(AllCloseTester.registry[Sequence], SequenceAllCloseOperator)
    assert isinstance(AllCloseTester.registry[Tensor], TensorAllCloseOperator)
    assert isinstance(AllCloseTester.registry[bool], ScalarAllCloseOperator)
    assert isinstance(AllCloseTester.registry[dict], MappingAllCloseOperator)
    assert isinstance(AllCloseTester.registry[float], ScalarAllCloseOperator)
    assert isinstance(AllCloseTester.registry[int], ScalarAllCloseOperator)
    assert isinstance(AllCloseTester.registry[list], SequenceAllCloseOperator)
    assert isinstance(AllCloseTester.registry[np.ndarray], NDArrayAllCloseOperator)
    assert isinstance(AllCloseTester.registry[object], DefaultAllCloseOperator)
    assert isinstance(AllCloseTester.registry[tuple], SequenceAllCloseOperator)


@patch.dict(AllCloseTester.registry, {}, clear=True)
def test_allclose_tester_add_allclose_operator():
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_allclose_operator(str, operator)
    assert tester.registry[str] == operator


@patch.dict(AllCloseTester.registry, {}, clear=True)
def test_allclose_tester_add_allclose_operator_duplicate_exist_ok_true():
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_allclose_operator(str, Mock(spec=BaseAllCloseOperator))
    tester.add_allclose_operator(str, operator, exist_ok=True)
    assert tester.registry[str] == operator


@patch.dict(AllCloseTester.registry, {}, clear=True)
def test_allclose_tester_add_allclose_operator_duplicate_exist_ok_false():
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_allclose_operator(str, Mock(spec=BaseAllCloseOperator))
    with raises(ValueError):
        tester.add_allclose_operator(str, operator)


def test_allclose_tester_has_allclose_operator_true():
    assert AllCloseTester().has_allclose_operator(dict)


def test_allclose_tester_has_allclose_operator_false():
    assert not AllCloseTester().has_allclose_operator(str)


def test_allclose_tester_find_allclose_operator_direct():
    assert isinstance(AllCloseTester().find_allclose_operator(dict), MappingAllCloseOperator)


def test_allclose_tester_find_allclose_operator_indirect():
    assert isinstance(AllCloseTester().find_allclose_operator(str), DefaultAllCloseOperator)


def test_allclose_tester_find_allclose_operator_incorrect_type():
    with raises(TypeError):
        AllCloseTester().find_allclose_operator(Mock(__mro__=[]))


##########################################
#     Tests for objects_are_allclose     #
##########################################


def test_objects_are_allclose_different_type():
    assert not objects_are_allclose([], ())


def test_objects_are_allclose_different_type_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_allclose(None, [], show_difference=True)
        assert caplog.messages[0].startswith("Objects have different types:")


@mark.parametrize("data1,data2", ((1, 1), (2.5, 2.5), (1 + 1e-9, 1), (1 - 1e-9, 1)))
def test_objects_are_allclose_scalar_true(data1: float, data2: float):
    assert objects_are_allclose(data1, data2)


def test_objects_are_allclose_scalar_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert objects_are_allclose(1, 1, show_difference=True)
        assert not caplog.messages


@mark.parametrize("data1,data2", ((1, 1.1), (2.5, 0), (1 + 1e-7, 1), (1 - 1e-7, 1)))
def test_objects_are_allclose_scalar_false(data1: float, data2: float):
    assert not objects_are_allclose(data1, data2, rtol=0)


def test_objects_are_allclose_scalar_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_allclose(1, 2, show_difference=True)
        assert caplog.messages[0].startswith("The numbers are different:")


@mark.parametrize("value,atol", ((1.5, 1), (1.05, 1e-1), (1 + 5e-3, 1e-2)))
def test_objects_are_allclose_scalar_true_atol(value: float, atol: float):
    assert objects_are_allclose(value, 1, atol=atol, rtol=0)


@mark.parametrize("value,rtol", ((1.5, 1), (1.05, 1e-1), (1 + 5e-3, 1e-2)))
def test_objects_are_allclose_scalar_true_rtol(value, rtol: float):
    assert objects_are_allclose(value, 1, rtol=rtol)


@mark.parametrize("tensor", (torch.ones(2, 3), torch.ones(2, 3) + 1e-9, torch.ones(2, 3) - 1e-9))
def test_objects_are_allclose_torch_tensor_true(tensor: Tensor):
    assert objects_are_allclose(tensor, torch.ones(2, 3))


def test_objects_are_allclose_torch_tensor_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3), show_difference=True)
        assert not caplog.messages


@mark.parametrize("tensor", (torch.zeros(2, 3), torch.ones(2, 3) + 1e-7, torch.ones(2, 3) - 1e-7))
def test_objects_are_allclose_torch_tensor_false(tensor):
    assert not objects_are_allclose(tensor, torch.ones(2, 3), rtol=0)


def test_objects_are_allclose_torch_tensor_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_allclose(torch.zeros(2, 3), torch.ones(2, 3), show_difference=True)
        assert caplog.messages


@mark.parametrize(
    "tensor,atol",
    ((torch.ones(2, 3) + 0.5, 1), (torch.ones(2, 3) + 0.05, 1e-1), (torch.ones(2, 3) + 5e-3, 1e-2)),
)
def test_objects_are_allclose_torch_tensor_true_atol(tensor, atol: float):
    assert objects_are_allclose(tensor, torch.ones(2, 3), atol=atol, rtol=0)


@mark.parametrize(
    "tensor,rtol",
    ((torch.ones(2, 3) + 0.5, 1), (torch.ones(2, 3) + 0.05, 1e-1), (torch.ones(2, 3) + 5e-3, 1e-2)),
)
def test_objects_are_allclose_torch_tensor_true_rtol(tensor, rtol: float):
    assert objects_are_allclose(tensor, torch.ones(2, 3), rtol=rtol)


@mark.parametrize("array", (np.ones((2, 3)), np.ones((2, 3)) + 1e-9, np.ones((2, 3)) - 1e-9))
def test_objects_are_allclose_numpy_array_true(array):
    assert objects_are_allclose(array, np.ones((2, 3)))


def test_objects_are_allclose_numpy_array_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert objects_are_allclose(np.ones((2, 3)), np.ones((2, 3)), show_difference=True)
        assert not caplog.messages


@mark.parametrize("array", (np.zeros((2, 3)), np.ones((2, 3)) + 1e-7, np.ones((2, 3)) - 1e-7))
def test_objects_are_allclose_numpy_array_false(array):
    assert not objects_are_allclose(array, np.ones((2, 3)), rtol=0)


def test_objects_are_allclose_numpy_array_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_allclose(np.zeros((2, 3)), np.ones((2, 3)), show_difference=True)
        assert caplog.messages


@mark.parametrize(
    "tensor,atol",
    ((np.ones((2, 3)) + 0.5, 1), (np.ones((2, 3)) + 0.05, 1e-1), (np.ones((2, 3)) + 5e-3, 1e-2)),
)
def test_objects_are_allclose_numpy_array_true_atol(tensor, atol: float):
    assert objects_are_allclose(tensor, np.ones((2, 3)), atol=atol, rtol=0)


@mark.parametrize(
    "tensor,rtol",
    ((np.ones((2, 3)) + 0.5, 1), (np.ones((2, 3)) + 0.05, 1e-1), (np.ones((2, 3)) + 5e-3, 1e-2)),
)
def test_objects_are_allclose_numpy_array_true_rtol(tensor, rtol: float):
    assert objects_are_allclose(tensor, np.ones((2, 3)), rtol=rtol)


@mark.parametrize(
    "data1,data2",
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
def test_objects_are_allclose_sequence_true(data1, data2):
    assert objects_are_allclose(data1, data2)


def test_objects_are_allclose_sequence_false(caplog: LogCaptureFixture):
    assert not objects_are_allclose([1, 2], [1, 3])


def test_objects_are_allclose_sequence_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_allclose([1, 2], [1, 3], show_difference=True)
        assert caplog.messages


@mark.parametrize(
    "data1,data2",
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
def test_objects_are_allclose_mapping_true(data1, data2):
    assert objects_are_allclose(data1, data2)


def test_objects_are_allclose_mapping_false():
    assert not objects_are_allclose({"abc": 1, "def": 2}, {"abc": 1, "def": 3})


def test_objects_are_allclose_mapping_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_allclose(
            {"abc": 1, "def": 2}, {"abc": 1, "def": 3}, show_difference=True
        )
        assert caplog.messages


@mark.parametrize(
    "data1,data2",
    (
        ("abc", "abc"),
        (set(), set()),
        ({1, 2, 3}, {1, 2, 3}),
    ),
)
def test_objects_are_allclose_other_types_true(data1, data2):
    assert objects_are_allclose(data1, data2)


@mark.parametrize(
    "data1,data2",
    (
        ("abc", "abcd"),
        (set(), tuple()),
        ({1, 2}, {1, 2, 3}),
        ({1, 2, 4}, {1, 2, 3}),
    ),
)
def test_objects_are_allclose_other_types_false(data1, data2):
    assert not objects_are_allclose(data1, data2)


def test_objects_are_allclose_other_types_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not objects_are_allclose("abc", "abcd", show_difference=True)
        assert caplog.messages


#############################################
#     Tests for DefaultAllCloseOperator     #
#############################################


def test_default_allclose_operator_str():
    assert str(DefaultAllCloseOperator()) == "DefaultAllCloseOperator()"


@mark.parametrize("object1,object2", ((1, 1), (2.5, 2.5)))
def test_default_allclose_operator_allclose_true_scalar(object1: Number, object2: Number):
    assert DefaultAllCloseOperator().allclose(AllCloseTester(), object1, object2)


@mark.parametrize("object1,object2", ((1, 1.1), (2.5, 0)))
def test_default_allclose_operator_allclose_false_scalar(object1: Number, object2: Number):
    assert not DefaultAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_default_allclose_operator_allclose_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not DefaultAllCloseOperator().allclose(
            tester=AllCloseTester(), object1=1, object2=2, show_difference=True
        )
        assert caplog.messages[0].startswith("Objects are different:")


def test_default_allclose_operator_allclose_different_type():
    assert not DefaultAllCloseOperator().allclose(AllCloseTester(), [], ())


def test_default_allclose_operator_allclose_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not DefaultAllCloseOperator().allclose(
            AllCloseTester(), [], (), show_difference=True
        )
        assert caplog.messages[0].startswith("Objects have different types:")


#############################################
#     Tests for MappingAllCloseOperator     #
#############################################


def test_mapping_allclose_operator_str():
    assert str(MappingAllCloseOperator()) == "MappingAllCloseOperator()"


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
def test_mapping_allclose_operator_allclose_true(object1: Mapping, object2: Mapping):
    assert MappingAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_mapping_allclose_operator_allclose_false_different_value():
    assert not MappingAllCloseOperator().allclose(
        AllCloseTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"1": torch.ones(2, 3), "2": torch.ones(2)},
    )


def test_mapping_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not MappingAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"1": torch.ones(2, 3), "2": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("The mappings have a different value for the key 2:")


def test_mapping_allclose_operator_allclose_false_different_keys():
    assert not MappingAllCloseOperator().allclose(
        AllCloseTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"10": torch.ones(2, 3), "20": torch.ones(2)},
    )


def test_mapping_allclose_operator_allclose_false_different_keys_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not MappingAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"10": torch.ones(2, 3), "20": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The mappings have different keys:")


def test_mapping_allclose_operator_allclose_false_different_length():
    assert not MappingAllCloseOperator().allclose(
        AllCloseTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"1": torch.ones(2, 3), "2": torch.zeros(2), "3": torch.ones(2)},
    )


def test_mapping_allclose_operator_allclose_false_different_length_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not MappingAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"1": torch.ones(2, 3), "2": torch.zeros(2), "3": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The mappings have different sizes:")


def test_mapping_allclose_operator_allclose_false_different_type():
    assert not MappingAllCloseOperator().allclose(AllCloseTester(), {}, OrderedDict([]))


def test_mapping_allclose_operator_allclose_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not MappingAllCloseOperator().allclose(
            tester=AllCloseTester(), object1={}, object2=OrderedDict([]), show_difference=True
        )
        assert caplog.messages[0].startswith("The mappings have different types:")


@mark.parametrize(
    "mapping,atol",
    (
        ({"key": torch.ones(2, 3) + 0.5}, 1),
        ({"key": torch.ones(2, 3) + 0.05}, 1e-1),
        ({"key": torch.ones(2, 3) + 5e-3}, 1e-2),
    ),
)
def test_mapping_allclose_operator_allclose_true_atol(mapping: Mapping, atol: float):
    assert SequenceAllCloseOperator().allclose(
        AllCloseTester(), {"key": torch.ones(2, 3)}, mapping, atol=atol, rtol=0
    )


@mark.parametrize(
    "mapping,rtol",
    (
        ({"key": torch.ones(2, 3) + 0.5}, 1),
        ({"key": torch.ones(2, 3) + 0.05}, 1e-1),
        ({"key": torch.ones(2, 3) + 5e-3}, 1e-2),
    ),
)
def test_mapping_allclose_operator_allclose_true_rtol(mapping: Mapping, rtol: float):
    assert SequenceAllCloseOperator().allclose(
        AllCloseTester(), {"key": torch.ones(2, 3)}, mapping, rtol=rtol
    )


############################################
#     Tests for ScalarAllCloseOperator     #
############################################


def test_scalar_allclose_operator_str():
    assert str(ScalarAllCloseOperator()) == "ScalarAllCloseOperator()"


@mark.parametrize("value", (2, 2.0, 2 + 1e-9, 2 - 1e-9))
def test_scalar_allclose_operator_allclose_true(value: Union[int, float]):
    assert ScalarAllCloseOperator().allclose(AllCloseTester(), 2, value)


def test_scalar_allclose_operator_allclose_true_bool():
    assert ScalarAllCloseOperator().allclose(AllCloseTester(), True, True)


def test_scalar_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert ScalarAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=1,
            object2=1,
            show_difference=True,
        )
        assert not caplog.messages


@mark.parametrize("object1,object2", ((1, 1.1), (2.5, 0), (1 + 1e-7, 1), (1 - 1e-7, 1)))
def test_scalar_allclose_operator_allclose_false_different_value(
    object1: Union[int, float], object2: Union[int, float]
):
    assert not ScalarAllCloseOperator().allclose(AllCloseTester(), object1, object2, rtol=0)


def test_scalar_allclose_operator_allclose_false_bool():
    assert not ScalarAllCloseOperator().allclose(AllCloseTester(), True, False)


def test_scalar_allclose_operator_allclose_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not ScalarAllCloseOperator().allclose(AllCloseTester(), 1, 2, show_difference=True)
        assert caplog.messages[0].startswith("The numbers are different:")


def test_scalar_allclose_operator_allclose_false_incorrect_type():
    assert not ScalarAllCloseOperator().allclose(AllCloseTester(), 1, "abc")


def test_scalar_allclose_operator_allclose_false_incorrect_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not ScalarAllCloseOperator().allclose(
            AllCloseTester(), 1, "abc", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a scalar (bool or int or float):")


##############################################
#     Tests for SequenceAllCloseOperator     #
##############################################


def test_sequence_allclose_operator_str():
    assert str(SequenceAllCloseOperator()) == "SequenceAllCloseOperator()"


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
        ([torch.ones(2, 3) + 1e-9, torch.zeros(2)], [torch.ones(2, 3), torch.zeros(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.zeros(2))),
        ((torch.ones(2, 3), torch.zeros(2) - 1e-9), (torch.ones(2, 3), torch.zeros(2))),
        (
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
        ),
    ),
)
def test_sequence_allclose_operator_allclose_true(object1: Sequence, object2: Sequence):
    assert SequenceAllCloseOperator().allclose(AllCloseTester(), object1, object2)


@mark.parametrize(
    "object1,object2",
    (
        (["abc", "deg"], ["abc", "def"]),
        (("abc", "deg"), ("abc", "def")),
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.ones(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.ones(2))),
    ),
)
def test_sequence_allclose_operator_allclose_false_different_value(
    object1: Sequence, object2: Sequence
):
    assert not SequenceAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_sequence_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not SequenceAllCloseOperator().allclose(
            tester=AllCloseTester(),
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
def test_sequence_allclose_operator_allclose_false_different_length(
    object1: Sequence, object2: Sequence
):
    assert not SequenceAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_sequence_allclose_operator_allclose_false_different_length_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not SequenceAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=[1, 2, 3],
            object2=[1, 2],
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The sequences have different sizes:")


def test_sequence_allclose_operator_allclose_false_different_type():
    assert not SequenceAllCloseOperator().allclose(AllCloseTester(), [], ())


def test_sequence_allclose_operator_allclose_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not SequenceAllCloseOperator().allclose(
            AllCloseTester(), [], (), show_difference=True
        )
        assert caplog.messages[0].startswith("The sequences have different types:")


@mark.parametrize(
    "sequence,atol",
    (
        ([torch.ones(2, 3) + 0.5], 1),
        ([torch.ones(2, 3) + 0.05], 1e-1),
        ([torch.ones(2, 3) + 5e-3], 1e-2),
    ),
)
def test_sequence_allclose_operator_allclose_true_atol(sequence: Sequence, atol: float):
    assert SequenceAllCloseOperator().allclose(
        AllCloseTester(), [torch.ones(2, 3)], sequence, atol=atol, rtol=0
    )


@mark.parametrize(
    "sequence,rtol",
    (
        ([torch.ones(2, 3) + 0.5], 1),
        ([torch.ones(2, 3) + 0.05], 1e-1),
        ([torch.ones(2, 3) + 5e-3], 1e-2),
    ),
)
def test_sequence_allclose_operator_allclose_true_rtol(sequence: Sequence, rtol: float):
    assert SequenceAllCloseOperator().allclose(
        AllCloseTester(), [torch.ones(2, 3)], sequence, rtol=rtol
    )
