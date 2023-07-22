from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import Mock, patch

from pytest import mark, raises

from coola.allclose import objects_are_allclose
from coola.comparators import (
    BaseAllCloseOperator,
    DefaultAllCloseOperator,
    MappingAllCloseOperator,
    ScalarAllCloseOperator,
    SequenceAllCloseOperator,
)
from coola.comparators.numpy_ import NDArrayAllCloseOperator
from coola.comparators.torch_ import (
    PackedSequenceAllCloseOperator,
    TensorAllCloseOperator,
)
from coola.testers import AllCloseTester, LocalAllCloseTester
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

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
def test_allclose_tester_add_operator() -> None:
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_operator(str, operator)
    assert tester.registry[str] == operator


@patch.dict(AllCloseTester.registry, {}, clear=True)
def test_allclose_tester_add_operator_duplicate_exist_ok_true() -> None:
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_operator(str, Mock(spec=BaseAllCloseOperator))
    tester.add_operator(str, operator, exist_ok=True)
    assert tester.registry[str] == operator


@patch.dict(AllCloseTester.registry, {}, clear=True)
def test_allclose_tester_add_operator_duplicate_exist_ok_false() -> None:
    tester = AllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_operator(str, Mock(spec=BaseAllCloseOperator))
    with raises(RuntimeError, match="An operator (.*) is already registered"):
        tester.add_operator(str, operator)


def test_allclose_tester_has_operator_true() -> None:
    assert AllCloseTester().has_operator(dict)


def test_allclose_tester_has_operator_false() -> None:
    assert not AllCloseTester().has_operator(str)


def test_allclose_tester_find_operator_direct() -> None:
    assert isinstance(AllCloseTester().find_operator(dict), MappingAllCloseOperator)


def test_allclose_tester_find_operator_indirect() -> None:
    assert isinstance(AllCloseTester().find_operator(str), DefaultAllCloseOperator)


def test_allclose_tester_find_operator_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect data type:"):
        AllCloseTester().find_operator(Mock(__mro__=[]))


@patch.dict(AllCloseTester.registry, {object: DefaultAllCloseOperator()}, clear=True)
def test_allclose_tester_local_copy() -> None:
    tester = AllCloseTester.local_copy()
    tester.add_operator(dict, MappingAllCloseOperator())
    assert AllCloseTester.registry == {object: DefaultAllCloseOperator()}
    assert tester == LocalAllCloseTester(
        {dict: MappingAllCloseOperator(), object: DefaultAllCloseOperator()}
    )


#########################################
#     Tests for LocalAllCloseTester     #
#########################################


def test_local_allclose_tester_str() -> None:
    assert str(LocalAllCloseTester()).startswith("LocalAllCloseTester(")


def test_local_allclose_tester__eq__true() -> None:
    assert LocalAllCloseTester({object: DefaultAllCloseOperator()}) == LocalAllCloseTester(
        {object: DefaultAllCloseOperator()}
    )


def test_local_allclose_tester__eq__true_empty() -> None:
    assert LocalAllCloseTester(None) == LocalAllCloseTester({})


def test_local_allclose_tester__eq__false_different_key() -> None:
    assert not LocalAllCloseTester({object: DefaultAllCloseOperator()}) == LocalAllCloseTester(
        {int: DefaultAllCloseOperator()}
    )


def test_local_allclose_tester__eq__false_different_value() -> None:
    assert not LocalAllCloseTester({object: DefaultAllCloseOperator()}) == LocalAllCloseTester(
        {object: MappingAllCloseOperator()}
    )


def test_local_allclose_tester__eq__false_different_type() -> None:
    assert not LocalAllCloseTester() == 1


def test_local_allclose_tester_registry_default() -> None:
    assert LocalAllCloseTester().registry == {}


def test_local_allclose_tester_add_operator() -> None:
    tester = LocalAllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_operator(int, operator)
    assert tester == LocalAllCloseTester({int: operator})


def test_local_allclose_tester_add_operator_duplicate_exist_ok_true() -> None:
    tester = LocalAllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_operator(int, Mock(spec=BaseAllCloseOperator))
    tester.add_operator(int, operator, exist_ok=True)
    assert tester == LocalAllCloseTester({int: operator})


def test_local_allclose_tester_add_operator_duplicate_exist_ok_false() -> None:
    tester = LocalAllCloseTester()
    operator = Mock(spec=BaseAllCloseOperator)
    tester.add_operator(int, Mock(spec=BaseAllCloseOperator))
    with raises(RuntimeError, match="An operator (.*) is already registered"):
        tester.add_operator(int, operator)


def test_local_allclose_tester_clone() -> None:
    tester = LocalAllCloseTester({dict: MappingAllCloseOperator()})
    tester_cloned = tester.clone()
    tester.add_operator(list, SequenceAllCloseOperator())
    tester_cloned.add_operator(object, DefaultAllCloseOperator())
    assert tester == LocalAllCloseTester(
        {dict: MappingAllCloseOperator(), list: SequenceAllCloseOperator()}
    )
    assert tester_cloned == LocalAllCloseTester(
        {
            dict: MappingAllCloseOperator(),
            object: DefaultAllCloseOperator(),
        }
    )


def test_local_allclose_tester_allclose_true() -> None:
    assert LocalAllCloseTester({object: DefaultAllCloseOperator()}).allclose(1, 1)


def test_local_allclose_tester_allclose_false() -> None:
    assert not LocalAllCloseTester({object: DefaultAllCloseOperator()}).allclose(1, 2)


def test_local_allclose_tester_has_operator_true() -> None:
    assert LocalAllCloseTester({dict: MappingAllCloseOperator()}).has_operator(dict)


def test_local_allclose_tester_has_operator_false() -> None:
    assert not LocalAllCloseTester().has_operator(int)


def test_local_allclose_tester_find_operator_direct() -> None:
    assert isinstance(
        LocalAllCloseTester({dict: MappingAllCloseOperator()}).find_operator(dict),
        MappingAllCloseOperator,
    )


def test_local_allclose_tester_find_operator_indirect() -> None:
    assert isinstance(
        LocalAllCloseTester(
            {dict: MappingAllCloseOperator(), object: DefaultAllCloseOperator()}
        ).find_operator(str),
        DefaultAllCloseOperator,
    )


def test_local_allclose_tester_find_operator_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect data type:"):
        LocalAllCloseTester().find_operator(Mock(__mro__=[]))


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
