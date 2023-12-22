from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark

from coola.comparators import (
    DefaultAllCloseOperator,
    MappingAllCloseOperator,
    ScalarAllCloseOperator,
    SequenceAllCloseOperator,
)
from coola.comparators.allclose import get_mapping_allclose
from coola.testers import AllCloseTester
from coola.testing import torch_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()


#############################################
#     Tests for DefaultAllCloseOperator     #
#############################################


def test_default_allclose_operator_str() -> None:
    assert str(DefaultAllCloseOperator()) == "DefaultAllCloseOperator()"


def test_default_allclose_operator__eq__true() -> None:
    assert DefaultAllCloseOperator() == DefaultAllCloseOperator()


def test_default_allclose_operator__eq__false() -> None:
    assert DefaultAllCloseOperator() != 123


def test_default_allclose_operator_allclose_true_same_object() -> None:
    obj = Mock()
    assert DefaultAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@mark.parametrize("object1,object2", ((1, 1), (2.5, 2.5)))
def test_default_allclose_operator_allclose_true_scalar(object1: Number, object2: Number) -> None:
    assert DefaultAllCloseOperator().allclose(AllCloseTester(), object1, object2)


@mark.parametrize("object1,object2", ((1, 1.1), (2.5, 0)))
def test_default_allclose_operator_allclose_false_scalar(object1: Number, object2: Number) -> None:
    assert not DefaultAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_default_allclose_operator_allclose_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DefaultAllCloseOperator().allclose(
            tester=AllCloseTester(), object1=1, object2=2, show_difference=True
        )
        assert caplog.messages[0].startswith("Objects are different:")


def test_default_allclose_operator_allclose_different_type() -> None:
    assert not DefaultAllCloseOperator().allclose(AllCloseTester(), [], ())


def test_default_allclose_operator_allclose_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DefaultAllCloseOperator().allclose(
            AllCloseTester(), [], (), show_difference=True
        )
        assert caplog.messages[0].startswith("Objects have different types:")


def test_default_allclose_operator_clone() -> None:
    op = DefaultAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


#############################################
#     Tests for MappingAllCloseOperator     #
#############################################


def test_mapping_allclose_operator_str() -> None:
    assert str(MappingAllCloseOperator()) == "MappingAllCloseOperator()"


def test_mapping_allclose_operator__eq__true() -> None:
    assert MappingAllCloseOperator() == MappingAllCloseOperator()


def test_mapping_allclose_operator__eq__false() -> None:
    assert MappingAllCloseOperator() != 123


@mark.parametrize(
    "object1,object2",
    (
        ({}, {}),
        (OrderedDict({}), OrderedDict({})),
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
def test_mapping_allclose_operator_allclose_true(object1: Mapping, object2: Mapping) -> None:
    assert MappingAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_mapping_allclose_operator_allclose_true_same_object() -> None:
    obj = {"cat": "meow"}
    assert MappingAllCloseOperator().allclose(AllCloseTester(), obj, obj)


def test_mapping_allclose_operator_allclose_false_different_value() -> None:
    assert not MappingAllCloseOperator().allclose(
        AllCloseTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"1": torch.ones(2, 3), "2": torch.ones(2)},
    )


def test_mapping_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not MappingAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"1": torch.ones(2, 3), "2": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[-1].startswith(
            "The mappings have a different value for the key '2':"
        )


def test_mapping_allclose_operator_allclose_false_different_keys() -> None:
    assert not MappingAllCloseOperator().allclose(
        AllCloseTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"10": torch.ones(2, 3), "20": torch.ones(2)},
    )


def test_mapping_allclose_operator_allclose_false_different_keys_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not MappingAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"10": torch.ones(2, 3), "20": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("The mappings have different keys:")


def test_mapping_allclose_operator_allclose_false_different_length() -> None:
    assert not MappingAllCloseOperator().allclose(
        AllCloseTester(),
        {"1": torch.ones(2, 3), "2": torch.zeros(2)},
        {"1": torch.ones(2, 3), "2": torch.zeros(2), "3": torch.ones(2)},
    )


def test_mapping_allclose_operator_allclose_false_different_length_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not MappingAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1={"1": torch.ones(2, 3), "2": torch.zeros(2)},
            object2={"1": torch.ones(2, 3), "2": torch.zeros(2), "3": torch.ones(2)},
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The mappings have different sizes:")


def test_mapping_allclose_operator_allclose_false_different_type() -> None:
    assert not MappingAllCloseOperator().allclose(AllCloseTester(), {}, OrderedDict([]))


def test_mapping_allclose_operator_allclose_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not MappingAllCloseOperator().allclose(
            tester=AllCloseTester(), object1={}, object2=OrderedDict([]), show_difference=True
        )
        assert caplog.messages[0].startswith("The mappings have different types:")


@torch_available
@mark.parametrize(
    "mapping,atol",
    (
        ({"key": torch.full((2, 3), 1.5)}, 1),
        ({"key": torch.full((2, 3), 1.05)}, 1e-1),
        ({"key": torch.full((2, 3), 1.005)}, 1e-2),
    ),
)
def test_mapping_allclose_operator_allclose_true_atol(mapping: Mapping, atol: float) -> None:
    assert MappingAllCloseOperator().allclose(
        AllCloseTester(), {"key": torch.ones(2, 3)}, mapping, atol=atol, rtol=0.0
    )


@torch_available
@mark.parametrize(
    "mapping,rtol",
    (
        ({"key": torch.full((2, 3), 1.5)}, 1),
        ({"key": torch.full((2, 3), 1.05)}, 1e-1),
        ({"key": torch.full((2, 3), 1.005)}, 1e-2),
    ),
)
def test_mapping_allclose_operator_allclose_true_rtol(mapping: Mapping, rtol: float) -> None:
    assert MappingAllCloseOperator().allclose(
        AllCloseTester(), {"key": torch.ones(2, 3)}, mapping, rtol=rtol
    )


@torch_available
@mark.parametrize(
    "mapping,atol",
    (
        ({1.5: torch.ones(2, 3)}, 1),
        ({1.05: torch.ones(2, 3)}, 1e-1),
        ({1.005: torch.ones(2, 3)}, 1e-2),
    ),
)
def test_mapping_allclose_operator_allclose_true_atol_keys(mapping: Mapping, atol: float) -> None:
    assert MappingAllCloseOperator().allclose(
        AllCloseTester(), {1.0: torch.ones(2, 3)}, mapping, atol=atol, rtol=0.0
    )


@torch_available
@mark.parametrize(
    "mapping,rtol",
    (
        ({1.5: torch.ones(2, 3)}, 1),
        ({1.05: torch.ones(2, 3)}, 1e-1),
        ({1.005: torch.ones(2, 3)}, 1e-2),
    ),
)
def test_mapping_allclose_operator_allclose_true_rtol_keys(mapping: Mapping, rtol: float) -> None:
    assert MappingAllCloseOperator().allclose(
        AllCloseTester(), {1.0: torch.ones(2, 3)}, mapping, rtol=rtol
    )


def test_mapping_allclose_operator_clone() -> None:
    op = MappingAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


############################################
#     Tests for ScalarAllCloseOperator     #
############################################


def test_scalar_allclose_operator_str() -> None:
    assert str(ScalarAllCloseOperator()) == "ScalarAllCloseOperator()"


def test_scalar_allclose_operator__eq__true() -> None:
    assert ScalarAllCloseOperator() == ScalarAllCloseOperator()


def test_scalar_allclose_operator__eq__false() -> None:
    assert ScalarAllCloseOperator() != 123


@mark.parametrize(
    "object1,object2",
    (
        (-1.0, -1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (float("inf"), float("inf")),
        (float("-inf"), float("-inf")),
    ),
)
def test_scalar_allclose_operator_allclose_true_float(object1: float, object2: float) -> None:
    assert ScalarAllCloseOperator().allclose(AllCloseTester(), object1, object2)


@mark.parametrize("object1,object2", ((-1, -1), (0, 0), (1, 1)))
def test_scalar_allclose_operator_allclose_true_int(object1: int, object2: int) -> None:
    assert ScalarAllCloseOperator().allclose(AllCloseTester(), object1, object2)


@mark.parametrize("object1,object2", ((True, True), (False, False)))
def test_scalar_allclose_operator_allclose_true_bool(object1: bool, object2: bool) -> None:
    assert ScalarAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_scalar_allclose_operator_allclose_true_same_object() -> None:
    obj = 42
    assert ScalarAllCloseOperator().allclose(AllCloseTester(), obj, obj)


def test_scalar_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert ScalarAllCloseOperator().allclose(
            tester=AllCloseTester(), object1=1, object2=1, show_difference=True
        )
        assert not caplog.messages


def test_scalar_allclose_operator_allclose_equal_nan_true() -> None:
    assert ScalarAllCloseOperator().allclose(
        AllCloseTester(), float("nan"), float("nan"), equal_nan=True
    )


def test_scalar_allclose_operator_allclose_equal_nan_false() -> None:
    assert not ScalarAllCloseOperator().allclose(AllCloseTester(), float("nan"), float("nan"))


@mark.parametrize(
    "object1,object2,atol",
    (
        (0, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 5, 10),
        (1.0, 1.0 + 1e-4, 1e-3),
        (1.0, 1.0 - 1e-4, 1e-3),
        (False, True, 1),
    ),
)
def test_scalar_allclose_operator_allclose_true_atol(
    object1: int | float, object2: int | float, atol: float
) -> None:
    assert ScalarAllCloseOperator().allclose(
        AllCloseTester(), object1, object2, atol=atol, rtol=0.0
    )


@mark.parametrize(
    "object1,object2,rtol",
    (
        (0, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 5, 10),
        (1.0, 1.0 + 1e-4, 1e-3),
        (1.0, 1.0 - 1e-4, 1e-3),
        (False, True, 1),
    ),
)
def test_scalar_allclose_operator_allclose_true_rtol(
    object1: int | float, object2: int | float, rtol: float
) -> None:
    assert ScalarAllCloseOperator().allclose(
        AllCloseTester(), object1, object2, atol=0.0, rtol=rtol
    )


@mark.parametrize(
    "object1,object2",
    (
        (1.0, 1.1),
        (2.5, 0.0),
        (1.0 + 1e-7, 1.0),
        (1.0 - 1e-7, 1.0),
        (float("inf"), 1.0),
        (float("NaN"), 1.0),
        (float("NaN"), float("NaN")),
        (0, 1),
        (1, -1),
        (True, False),
    ),
)
def test_scalar_allclose_operator_allclose_false_different_value(
    object1: bool | int | float, object2: bool | int | float
) -> None:
    assert not ScalarAllCloseOperator().allclose(AllCloseTester(), object1, object2, rtol=0.0)


def test_scalar_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ScalarAllCloseOperator().allclose(AllCloseTester(), 1, 2, show_difference=True)
        assert caplog.messages[0].startswith("The numbers are different:")


@mark.parametrize("object1,object2", ((1.0, 1), (1.0, True), (1, True), (1, "1")))
def test_scalar_allclose_operator_allclose_false_incorrect_type(
    object1: bool | int | float, object2: Any
) -> None:
    assert not ScalarAllCloseOperator().allclose(AllCloseTester(), object1, object2)


@mark.parametrize("object1,object2", ((1.0, 1), (1.0, True), (1, True), (1, "1")))
def test_scalar_allclose_operator_allclose_false_incorrect_type_show_difference(
    caplog: LogCaptureFixture, object1: bool | int | float, object2: Any
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ScalarAllCloseOperator().allclose(
            AllCloseTester(), object1, object2, show_difference=True
        )
        assert caplog.messages[0].startswith("Objects have different types:")


def test_scalar_allclose_operator_clone() -> None:
    op = ScalarAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


##############################################
#     Tests for SequenceAllCloseOperator     #
##############################################


def test_sequence_allclose_operator_str() -> None:
    assert str(SequenceAllCloseOperator()) == "SequenceAllCloseOperator()"


def test_sequence_allclose_operator__eq__true() -> None:
    assert SequenceAllCloseOperator() == SequenceAllCloseOperator()


def test_sequence_allclose_operator__eq__false() -> None:
    assert SequenceAllCloseOperator() != 123


@mark.parametrize(
    "object1,object2",
    (
        ([], []),
        ((), ()),
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), (1, 2, 3)),
        (["abc", "def"], ["abc", "def"]),
        (("abc", "def"), ("abc", "def")),
    ),
)
def test_sequence_allclose_operator_allclose_true(object1: Sequence, object2: Sequence) -> None:
    assert SequenceAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_sequence_allclose_operator_allclose_true_same_object() -> None:
    obj = [1, 2, 3]
    assert SequenceAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@torch_available
@mark.parametrize(
    "object1,object2",
    (
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.zeros(2)]),
        ([torch.full((2, 3), 1.0 + 1e-9), torch.zeros(2)], [torch.ones(2, 3), torch.zeros(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.zeros(2))),
        ((torch.ones(2, 3), torch.full((2,), -1e-9)), (torch.ones(2, 3), torch.zeros(2))),
        (
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
        ),
    ),
)
def test_sequence_allclose_operator_allclose_true_torch(
    object1: Sequence, object2: Sequence
) -> None:
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
) -> None:
    assert not SequenceAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_sequence_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
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
) -> None:
    assert not SequenceAllCloseOperator().allclose(AllCloseTester(), object1, object2)


def test_sequence_allclose_operator_allclose_false_different_length_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SequenceAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=[1, 2, 3],
            object2=[1, 2],
            show_difference=True,
        )
        assert caplog.messages[0].startswith("The sequences have different sizes:")


def test_sequence_allclose_operator_allclose_false_different_type() -> None:
    assert not SequenceAllCloseOperator().allclose(AllCloseTester(), [], ())


def test_sequence_allclose_operator_allclose_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SequenceAllCloseOperator().allclose(
            AllCloseTester(), [], (), show_difference=True
        )
        assert caplog.messages[0].startswith("The sequences have different types:")


@torch_available
@mark.parametrize(
    "sequence,atol",
    (
        ([torch.full((2, 3), 1.5)], 1),
        ([torch.full((2, 3), 1.05)], 1e-1),
        ([torch.full((2, 3), 1.005)], 1e-2),
    ),
)
def test_sequence_allclose_operator_allclose_true_atol(sequence: Sequence, atol: float) -> None:
    assert SequenceAllCloseOperator().allclose(
        AllCloseTester(), [torch.ones(2, 3)], sequence, atol=atol, rtol=0.0
    )


@torch_available
@mark.parametrize(
    "sequence,rtol",
    (
        ([torch.full((2, 3), 1.5)], 1),
        ([torch.full((2, 3), 1.05)], 1e-1),
        ([torch.full((2, 3), 1.005)], 1e-2),
    ),
)
def test_sequence_allclose_operator_allclose_true_rtol(sequence: Sequence, rtol: float) -> None:
    assert SequenceAllCloseOperator().allclose(
        AllCloseTester(), [torch.ones(2, 3)], sequence, rtol=rtol
    )


def test_sequence_allclose_operator_clone() -> None:
    op = SequenceAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


##########################################
#     Tests for get_mapping_allclose     #
##########################################


def test_get_mapping_allclose() -> None:
    mapping = get_mapping_allclose()
    assert len(mapping) == 9
    assert isinstance(mapping[Mapping], MappingAllCloseOperator)
    assert isinstance(mapping[Sequence], SequenceAllCloseOperator)
    assert isinstance(mapping[bool], ScalarAllCloseOperator)
    assert isinstance(mapping[dict], MappingAllCloseOperator)
    assert isinstance(mapping[float], ScalarAllCloseOperator)
    assert isinstance(mapping[int], ScalarAllCloseOperator)
    assert isinstance(mapping[list], SequenceAllCloseOperator)
    assert isinstance(mapping[object], DefaultAllCloseOperator)
    assert isinstance(mapping[tuple], SequenceAllCloseOperator)
