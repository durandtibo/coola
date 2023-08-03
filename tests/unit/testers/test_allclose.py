from __future__ import annotations

from collections.abc import Mapping, Sequence
from unittest.mock import Mock, patch

from pytest import raises

from coola.comparators import (
    BaseAllCloseOperator,
    DefaultAllCloseOperator,
    MappingAllCloseOperator,
    ScalarAllCloseOperator,
    SequenceAllCloseOperator,
)
from coola.comparators.numpy_ import ArrayAllCloseOperator
from coola.comparators.torch_ import (
    PackedSequenceAllCloseOperator,
    TensorAllCloseOperator,
)
from coola.testers import AllCloseTester, LocalAllCloseTester
from coola.testing import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

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
    assert isinstance(AllCloseTester.registry[np.ndarray], ArrayAllCloseOperator)


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
