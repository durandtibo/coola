from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark

from coola.comparators import (
    DefaultEqualityOperator,
    MappingEqualityOperator,
    SequenceEqualityOperator,
)
from coola.testers import EqualityTester
from coola.testing import numpy_available, torch_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()

#############################################
#     Tests for DefaultEqualityOperator     #
#############################################


def test_default_equality_operator_str() -> None:
    assert str(DefaultEqualityOperator()) == "DefaultEqualityOperator()"


@numpy_available
def test_default_equality_operator__eq__true() -> None:
    assert DefaultEqualityOperator() == DefaultEqualityOperator()


@numpy_available
def test_default_equality_operator__eq__false() -> None:
    assert DefaultEqualityOperator() != 123


def test_default_equality_operator_clone() -> None:
    op = DefaultEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


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
        (None, None),
    ),
)
def test_default_equality_operator_equal_true_scalar(
    object1: bool | int | float | None, object2: bool | int | float | None
) -> None:
    assert DefaultEqualityOperator().equal(EqualityTester(), object1, object2)


def test_default_equality_operator_equal_true_same_object() -> None:
    obj = Mock()
    assert DefaultEqualityOperator().equal(EqualityTester(), obj, obj)


@mark.parametrize(
    "object1,object2",
    (
        (1, 2),
        (1.0, 2.0),
        (True, False),
        (1, 1.0),
        (1, True),
        (1.0, True),
        (1.0, None),
    ),
)
def test_default_equality_operator_equal_false_scalar(
    object1: bool | int | float, object2: bool | int | float | None
) -> None:
    assert not DefaultEqualityOperator().equal(EqualityTester(), object1, object2)


def test_default_equality_operator_equal_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DefaultEqualityOperator().equal(
            tester=EqualityTester(), object1=1, object2=2, show_difference=True
        )
        assert caplog.messages[0].startswith("Objects are different:")


def test_default_equality_operator_equal_different_type() -> None:
    assert not DefaultEqualityOperator().equal(EqualityTester(), [], ())


def test_default_equality_operator_equal_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DefaultEqualityOperator().equal(EqualityTester(), [], (), show_difference=True)
        assert caplog.messages[0].startswith("Objects have different types:")


#############################################
#     Tests for MappingEqualityOperator     #
#############################################


def test_mapping_equality_operator_str() -> None:
    assert str(MappingEqualityOperator()) == "MappingEqualityOperator()"


@numpy_available
def test_mapping_equality_operator__eq__true() -> None:
    assert MappingEqualityOperator() == MappingEqualityOperator()


@numpy_available
def test_mapping_equality_operator__eq__false() -> None:
    assert MappingEqualityOperator() != 123


def test_mapping_equality_operator_clone() -> None:
    op = MappingEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
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
def test_mapping_equality_operator_equal_true(object1: Mapping, object2: Mapping) -> None:
    assert MappingEqualityOperator().equal(EqualityTester(), object1, object2)


def test_mapping_equality_operator_equal_true_same_object() -> None:
    obj = {"cat": "meow"}
    assert MappingEqualityOperator().equal(EqualityTester(), obj, obj)


@torch_available
def test_mapping_equality_operator_equal_false_different_value() -> None:
    assert not MappingEqualityOperator().equal(
        EqualityTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"1": torch.ones(2, 3), "2": torch.ones(2)},
    )


@torch_available
def test_mapping_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
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


@torch_available
def test_mapping_equality_operator_equal_false_different_keys() -> None:
    assert not MappingEqualityOperator().equal(
        EqualityTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"10": torch.ones(2, 3), "20": torch.ones(2)},
    )


@torch_available
def test_mapping_equality_operator_equal_false_different_keys_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not MappingEqualityOperator().equal(
            tester=EqualityTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"10": torch.ones(2, 3), "20": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The mappings have different keys:")


@torch_available
def test_mapping_equality_operator_equal_false_different_length() -> None:
    assert not MappingEqualityOperator().equal(
        EqualityTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"1": torch.ones(2, 3), "2": torch.zeros(2), "3": torch.ones(2)},
    )


@torch_available
def test_mapping_equality_operator_equal_false_different_length_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not MappingEqualityOperator().equal(
            tester=EqualityTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"1": torch.ones(2, 3), "2": torch.zeros(2), "3": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The mappings have different sizes:")


def test_mapping_equality_operator_equal_false_different_type() -> None:
    assert not MappingEqualityOperator().equal(EqualityTester(), {}, OrderedDict([]))


def test_mapping_equality_operator_equal_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not MappingEqualityOperator().equal(
            tester=EqualityTester(), object1={}, object2=OrderedDict([]), show_difference=True
        )
        assert caplog.messages[0].startswith("The mappings have different types:")


##############################################
#     Tests for SequenceEqualityOperator     #
##############################################


def test_sequence_equality_operator_str() -> None:
    assert str(SequenceEqualityOperator()) == "SequenceEqualityOperator()"


@numpy_available
def test_sequence_equality_operator__eq__true() -> None:
    assert SequenceEqualityOperator() == SequenceEqualityOperator()


@numpy_available
def test_sequence_equality_operator__eq__false() -> None:
    assert SequenceEqualityOperator() != 123


def test_sequence_equality_operator_clone() -> None:
    op = SequenceEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
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
def test_sequence_equality_operator_equal_true(object1: Sequence, object2: Sequence) -> None:
    assert SequenceEqualityOperator().equal(EqualityTester(), object1, object2)


def test_sequence_equality_operator_equal_true_same_object() -> None:
    obj = [1, 2, 3]
    assert SequenceEqualityOperator().equal(EqualityTester(), obj, obj)


@torch_available
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
) -> None:
    assert not SequenceEqualityOperator().equal(EqualityTester(), object1, object2)


def test_sequence_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SequenceEqualityOperator().equal(
            tester=EqualityTester(),
            object1=[1, 2],
            object2=[1, 3],
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("The sequences have at least one different value:")


@torch_available
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
) -> None:
    assert not SequenceEqualityOperator().equal(EqualityTester(), object1, object2)


def test_sequence_equality_operator_equal_false_different_length_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SequenceEqualityOperator().equal(
            tester=EqualityTester(),
            object1=[1, 2, 3],
            object2=[1, 2],
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The sequences have different sizes:")


def test_sequence_equality_operator_equal_false_different_type() -> None:
    assert not SequenceEqualityOperator().equal(EqualityTester(), [], ())


def test_sequence_equality_operator_equal_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SequenceEqualityOperator().equal(EqualityTester(), [], (), show_difference=True)
        assert caplog.messages[0].startswith("The sequences have different types:")
