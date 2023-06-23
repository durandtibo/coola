from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark, raises

from coola import is_numpy_available, is_torch_available
from coola._numpy import NDArrayAllCloseOperator
from coola._torch import PackedSequenceAllCloseOperator, TensorAllCloseOperator
from coola.allclose import (
    AllCloseTester,
    BaseAllCloseOperator,
    DefaultAllCloseOperator,
    MappingAllCloseOperator,
    ScalarAllCloseOperator,
    SequenceAllCloseOperator,
    objects_are_allclose,
)
from coola.testing import numpy_available, torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch
else:
    torch = Mock()


####################################
#     Tests for AllCloseTester     #
####################################


def test_allclose_tester_str() -> None:
    assert str(AllCloseTester()).startswith("AllCloseTester(")


@numpy_available
@torch_available
def test_allclose_tester_registry_default() -> None:
    assert len(AllCloseTester.registry) >= 9
    assert isinstance(AllCloseTester.registry[Mapping], MappingAllCloseOperator)
    assert isinstance(AllCloseTester.registry[Sequence], SequenceAllCloseOperator)
    assert isinstance(AllCloseTester.registry[bool], ScalarAllCloseOperator)
    assert isinstance(AllCloseTester.registry[dict], MappingAllCloseOperator)
    assert isinstance(AllCloseTester.registry[float], ScalarAllCloseOperator)
    assert isinstance(AllCloseTester.registry[int], ScalarAllCloseOperator)
    assert isinstance(AllCloseTester.registry[list], SequenceAllCloseOperator)
    assert isinstance(AllCloseTester.registry[object], DefaultAllCloseOperator)
    assert isinstance(AllCloseTester.registry[tuple], SequenceAllCloseOperator)


@numpy_available
def test_allclose_tester_registry_numpy() -> None:
    assert isinstance(AllCloseTester.registry[np.ndarray], NDArrayAllCloseOperator)


@torch_available
def test_allclose_tester_registry_torch() -> None:
    assert isinstance(AllCloseTester.registry[torch.Tensor], TensorAllCloseOperator)
    assert isinstance(
        AllCloseTester.registry[torch.nn.utils.rnn.PackedSequence], PackedSequenceAllCloseOperator
    )


@patch.dict(AllCloseTester.registry, {}, clear=True)
def test_allclose_tester_add_allclose_operator() -> None:
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_allclose_operator(str, operator)
    assert tester.registry[str] == operator


@patch.dict(AllCloseTester.registry, {}, clear=True)
def test_allclose_tester_add_allclose_operator_duplicate_exist_ok_true() -> None:
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_allclose_operator(str, Mock(spec=BaseAllCloseOperator))
    tester.add_allclose_operator(str, operator, exist_ok=True)
    assert tester.registry[str] == operator


@patch.dict(AllCloseTester.registry, {}, clear=True)
def test_allclose_tester_add_allclose_operator_duplicate_exist_ok_false() -> None:
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_allclose_operator(str, Mock(spec=BaseAllCloseOperator))
    with raises(RuntimeError, match="An operator (.*) is already registered"):
        tester.add_allclose_operator(str, operator)


def test_allclose_tester_has_allclose_operator_true() -> None:
    assert AllCloseTester().has_allclose_operator(dict)


def test_allclose_tester_has_allclose_operator_false() -> None:
    assert not AllCloseTester().has_allclose_operator(str)


def test_allclose_tester_find_allclose_operator_direct() -> None:
    assert isinstance(AllCloseTester().find_allclose_operator(dict), MappingAllCloseOperator)


def test_allclose_tester_find_allclose_operator_indirect() -> None:
    assert isinstance(AllCloseTester().find_allclose_operator(str), DefaultAllCloseOperator)


def test_allclose_tester_find_allclose_operator_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect data type:"):
        AllCloseTester().find_allclose_operator(Mock(__mro__=[]))


##########################################
#     Tests for objects_are_allclose     #
##########################################


def test_objects_are_allclose_false_different_type() -> None:
    assert not objects_are_allclose([], ())


@mark.parametrize(
    "object1,object2",
    (
        (1, 1),
        (0, 0),
        (-1, -1),
        (1.0, 1.0),
        (1.0, 1.0 + 1e-9),
        (0.0, 0.0),
        (0.0, 1e-9),
        (-1.0, -1.0),
        (True, True),
        (False, False),
    ),
)
def test_objects_are_allclose_scalar_true_float(
    object1: bool | int | float, object2: bool | int | float
) -> None:
    assert objects_are_allclose(object1, object2)


@mark.parametrize("value,atol", ((1.5, 1.0), (1.05, 1e-1), (1.005, 1e-2)))
def test_objects_are_allclose_scalar_true_atol(value: float, atol: float) -> None:
    assert objects_are_allclose(value, 1.0, atol=atol, rtol=0.0)


@mark.parametrize("value,rtol", ((1.5, 1.0), (1.05, 1e-1), (1.005, 1e-2)))
def test_objects_are_allclose_scalar_true_rtol(value: float, rtol: float) -> None:
    assert objects_are_allclose(value, 1.0, rtol=rtol)


@mark.parametrize(
    "object1,object2",
    (
        (1, 2),
        (1.0, 2.0),
        (1.0, 1.0 + 1e-7),
        (True, False),
        (1, 1.0),
        (1, True),
        (1.0, True),
    ),
)
def test_objects_are_allclose_scalar_false(object1: float, object2: float) -> None:
    assert not objects_are_allclose(object1, object2, rtol=0.0)


@torch_available
@mark.parametrize(
    "tensor", (torch.ones(2, 3), torch.full((2, 3), 1.0 + 1e-9), torch.full((2, 3), 1.0 - 1e-9))
)
def test_objects_are_allclose_torch_tensor_true(tensor: torch.Tensor) -> None:
    assert objects_are_allclose(tensor, torch.ones(2, 3))


@torch_available
@mark.parametrize(
    "tensor",
    (torch.zeros(2, 3), torch.full((2, 3), 1.0 + 1e-7), torch.full((2, 3), 1.0 - 1e-7)),
)
def test_objects_are_allclose_torch_tensor_false(tensor: torch.Tensor) -> None:
    assert not objects_are_allclose(tensor, torch.ones(2, 3), rtol=0.0)


@torch_available
@mark.parametrize(
    "tensor,atol",
    (
        (torch.full((2, 3), 1.5), 1),
        (torch.full((2, 3), 1.05), 1e-1),
        (torch.full((2, 3), 1.005), 1e-2),
    ),
)
def test_objects_are_allclose_torch_tensor_true_atol(tensor: torch.Tensor, atol: float) -> None:
    assert objects_are_allclose(tensor, torch.ones(2, 3), atol=atol, rtol=0.0)


@torch_available
@mark.parametrize(
    "tensor,rtol",
    (
        (torch.full((2, 3), 1.5), 1),
        (torch.full((2, 3), 1.05), 1e-1),
        (torch.full((2, 3), 1.005), 1e-2),
    ),
)
def test_objects_are_allclose_torch_tensor_true_rtol(tensor: torch.Tensor, rtol: float) -> None:
    assert objects_are_allclose(tensor, torch.ones(2, 3), rtol=rtol)


@numpy_available
@mark.parametrize(
    "array",
    (np.ones((2, 3)), np.full((2, 3), 1.0 + 1e-9), np.full((2, 3), 1.0 - 1e-9)),
)
def test_objects_are_allclose_numpy_array_true(array: np.ndarray) -> None:
    assert objects_are_allclose(array, np.ones((2, 3)))


@numpy_available
@mark.parametrize(
    "array",
    (np.zeros((2, 3)), np.full((2, 3), 1.0 + 1e-7), np.full((2, 3), 1.0 - 1e-7)),
)
def test_objects_are_allclose_numpy_array_false(array: np.ndarray) -> None:
    assert not objects_are_allclose(array, np.ones((2, 3)), rtol=0.0)


@numpy_available
@mark.parametrize(
    "array,atol",
    (
        (np.full((2, 3), 1.5), 1.0),
        (np.full((2, 3), 1.05), 1e-1),
        (np.full((2, 3), 1.005), 1e-2),
    ),
)
def test_objects_are_allclose_numpy_array_true_atol(array: np.ndarray, atol: float) -> None:
    assert objects_are_allclose(array, np.ones((2, 3)), atol=atol, rtol=0.0)


@numpy_available
@mark.parametrize(
    "array,rtol",
    (
        (np.full((2, 3), 1.5), 1.0),
        (np.full((2, 3), 1.05), 1e-1),
        (np.full((2, 3), 1.005), 1e-2),
    ),
)
def test_objects_are_allclose_numpy_array_true_rtol(array: np.ndarray, rtol: float) -> None:
    assert objects_are_allclose(array, np.ones((2, 3)), rtol=rtol)


@torch_available
@mark.parametrize(
    "object1,object2",
    (([], []), ((), ()), ([1, 2, 3], [1, 2, 3])),
)
def test_objects_are_allclose_sequence_true(object1: Sequence, object2: Sequence) -> None:
    assert objects_are_allclose(object1, object2)


@torch_available
@mark.parametrize(
    "object1,object2",
    (
        ([torch.ones(2, 3), torch.zeros(2)], [torch.ones(2, 3), torch.zeros(2)]),
        ((torch.ones(2, 3), torch.zeros(2)), (torch.ones(2, 3), torch.zeros(2))),
        (
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
            (torch.ones(2, 3), [torch.zeros(2), torch.ones(2)]),
        ),
    ),
)
def test_objects_are_allclose_sequence_true_torch(object1: Sequence, object2: Sequence) -> None:
    assert objects_are_allclose(object1, object2)


def test_objects_are_allclose_sequence_false() -> None:
    assert not objects_are_allclose([1, 2], [1, 3])


@torch_available
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
def test_objects_are_allclose_mapping_true(object1: Mapping, object2: Mapping) -> None:
    assert objects_are_allclose(object1, object2)


def test_objects_are_allclose_mapping_false() -> None:
    assert not objects_are_allclose({"abc": 1, "def": 2}, {"abc": 1, "def": 3})


@mark.parametrize(
    "object1,object2",
    (
        ("abc", "abc"),
        (set(), set()),
        ({1, 2, 3}, {1, 2, 3}),
    ),
)
def test_objects_are_allclose_other_types_true(object1: Any, object2: Any) -> None:
    assert objects_are_allclose(object1, object2)


@mark.parametrize(
    "object1,object2",
    (
        ("abc", "abcd"),
        (set(), tuple()),
        ({1, 2}, {1, 2, 3}),
        ({1, 2, 4}, {1, 2, 3}),
    ),
)
def test_objects_are_allclose_other_types_false(object1: Any, object2: Any) -> None:
    assert not objects_are_allclose(object1, object2)


@numpy_available
@torch_available
def test_objects_are_equal_true_complex_objects() -> None:
    assert objects_are_allclose(
        {
            "list": [
                1,
                2.0,
                torch.arange(5, dtype=torch.float),
                np.arange(3, dtype=float),
                [1, 2, 3],
            ],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5,
            "torch": torch.ones(5),
            "numpy": np.ones(4),
        },
        {
            "list": [
                1,
                2.0,
                torch.arange(5, dtype=torch.float).add(3e-9),
                np.arange(3, dtype=float) + 3e-9,
                [1, 2, 3],
            ],
            "tuple": ("1", (1, 2, torch.ones(2, 3), np.ones((2, 3)))),
            "dict": {"torch": torch.zeros(2, 3), "numpy": np.zeros((2, 3)), "list": []},
            "str": "abc",
            "int": 1,
            "float": 2.5 + 1e-9,
            "torch": torch.ones(5).add(2e-9),
            "numpy": np.ones(4) + 2e-9,
        },
    )


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
        assert caplog.messages[0].startswith("The mappings have different keys:")


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
