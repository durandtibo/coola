from __future__ import annotations

from collections.abc import Mapping, Sequence
from unittest.mock import Mock, patch

import pytest

from coola.comparators import (
    ArrayEqualityOperator,
    BaseEqualityOperator,
    DefaultEqualityOperator,
    MappingEqualityOperator,
    PackedSequenceEqualityOperator,
    SequenceEqualityOperator,
    TensorEqualityOperator,
)
from coola.testers import EqualityTester, LocalEqualityTester
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
#     Tests for EqualityTester     #
####################################


def test_equality_tester_str() -> None:
    assert str(EqualityTester()).startswith("EqualityTester(")


def test_equality_tester_registry_default() -> None:
    assert len(EqualityTester.registry) >= 6
    assert isinstance(EqualityTester.registry[Mapping], MappingEqualityOperator)
    assert isinstance(EqualityTester.registry[Sequence], SequenceEqualityOperator)
    assert isinstance(EqualityTester.registry[dict], MappingEqualityOperator)
    assert isinstance(EqualityTester.registry[list], SequenceEqualityOperator)
    assert isinstance(EqualityTester.registry[object], DefaultEqualityOperator)
    assert isinstance(EqualityTester.registry[tuple], SequenceEqualityOperator)


@numpy_available
def test_equality_tester_registry_numpy() -> None:
    assert isinstance(EqualityTester.registry[np.ndarray], ArrayEqualityOperator)


@torch_available
def test_equality_tester_registry_torch() -> None:
    assert isinstance(EqualityTester.registry[torch.Tensor], TensorEqualityOperator)
    assert isinstance(
        EqualityTester.registry[torch.nn.utils.rnn.PackedSequence], PackedSequenceEqualityOperator
    )


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_operator() -> None:
    tester = EqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_operator(int, operator)
    assert tester.registry[int] == operator


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_operator_duplicate_exist_ok_true() -> None:
    tester = EqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_operator(int, Mock(spec=BaseEqualityOperator))
    tester.add_operator(int, operator, exist_ok=True)
    assert tester.registry[int] == operator


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_operator_duplicate_exist_ok_false() -> None:
    tester = EqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_operator(int, Mock(spec=BaseEqualityOperator))
    with pytest.raises(RuntimeError, match="An operator (.*) is already registered"):
        tester.add_operator(int, operator)


def test_equality_tester_equal_true() -> None:
    assert EqualityTester().equal(1, 1)


def test_equality_tester_equal_false() -> None:
    assert not EqualityTester().equal(1, 2)


def test_equality_tester_has_operator_true() -> None:
    assert EqualityTester().has_operator(dict)


def test_equality_tester_has_operator_false() -> None:
    assert not EqualityTester().has_operator(int)


def test_equality_tester_find_operator_direct() -> None:
    assert isinstance(EqualityTester().find_operator(dict), MappingEqualityOperator)


def test_equality_tester_find_operator_indirect() -> None:
    assert isinstance(EqualityTester().find_operator(str), DefaultEqualityOperator)


def test_equality_tester_find_operator_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect data type:"):
        EqualityTester().find_operator(Mock(__mro__=[]))


@patch.dict(EqualityTester.registry, {object: DefaultEqualityOperator()}, clear=True)
def test_equality_tester_local_copy() -> None:
    tester = EqualityTester.local_copy()
    tester.add_operator(dict, MappingEqualityOperator())
    assert EqualityTester.registry == {object: DefaultEqualityOperator()}
    assert tester == LocalEqualityTester(
        {dict: MappingEqualityOperator(), object: DefaultEqualityOperator()}
    )


#########################################
#     Tests for LocalEqualityTester     #
#########################################


def test_local_equality_tester_str() -> None:
    assert str(LocalEqualityTester()).startswith("LocalEqualityTester(")


def test_local_equality_tester__eq__true() -> None:
    assert LocalEqualityTester({object: DefaultEqualityOperator()}) == LocalEqualityTester(
        {object: DefaultEqualityOperator()}
    )


def test_local_equality_tester__eq__true_empty() -> None:
    assert LocalEqualityTester(None) == LocalEqualityTester({})


def test_local_equality_tester__eq__false_different_key() -> None:
    assert LocalEqualityTester({object: DefaultEqualityOperator()}) != LocalEqualityTester(
        {int: DefaultEqualityOperator()}
    )


def test_local_equality_tester__eq__false_different_value() -> None:
    assert LocalEqualityTester({object: DefaultEqualityOperator()}) != LocalEqualityTester(
        {object: MappingEqualityOperator()}
    )


def test_local_equality_tester__eq__false_different_type() -> None:
    assert LocalEqualityTester() != 1


def test_local_equality_tester_registry_default() -> None:
    assert LocalEqualityTester().registry == {}


def test_local_equality_tester_add_operator() -> None:
    tester = LocalEqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_operator(int, operator)
    assert tester == LocalEqualityTester({int: operator})


def test_local_equality_tester_add_operator_duplicate_exist_ok_true() -> None:
    tester = LocalEqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_operator(int, Mock(spec=BaseEqualityOperator))
    tester.add_operator(int, operator, exist_ok=True)
    assert tester == LocalEqualityTester({int: operator})


def test_local_equality_tester_add_operator_duplicate_exist_ok_false() -> None:
    tester = LocalEqualityTester()
    operator = Mock(spec=BaseEqualityOperator)
    tester.add_operator(int, Mock(spec=BaseEqualityOperator))
    with pytest.raises(RuntimeError, match="An operator (.*) is already registered"):
        tester.add_operator(int, operator)


def test_local_equality_tester_clone() -> None:
    tester = LocalEqualityTester({dict: MappingEqualityOperator()})
    tester_cloned = tester.clone()
    tester.add_operator(list, SequenceEqualityOperator())
    tester_cloned.add_operator(object, DefaultEqualityOperator())
    assert tester == LocalEqualityTester(
        {dict: MappingEqualityOperator(), list: SequenceEqualityOperator()}
    )
    assert tester_cloned == LocalEqualityTester(
        {
            dict: MappingEqualityOperator(),
            object: DefaultEqualityOperator(),
        }
    )


def test_local_equality_tester_equal_true() -> None:
    assert LocalEqualityTester({object: DefaultEqualityOperator()}).equal(1, 1)


def test_local_equality_tester_equal_false() -> None:
    assert not LocalEqualityTester({object: DefaultEqualityOperator()}).equal(1, 2)


def test_local_equality_tester_has_operator_true() -> None:
    assert LocalEqualityTester({dict: MappingEqualityOperator()}).has_operator(dict)


def test_local_equality_tester_has_operator_false() -> None:
    assert not LocalEqualityTester().has_operator(int)


def test_local_equality_tester_find_operator_direct() -> None:
    assert isinstance(
        LocalEqualityTester({dict: MappingEqualityOperator()}).find_operator(dict),
        MappingEqualityOperator,
    )


def test_local_equality_tester_find_operator_indirect() -> None:
    assert isinstance(
        LocalEqualityTester(
            {dict: MappingEqualityOperator(), object: DefaultEqualityOperator()}
        ).find_operator(str),
        DefaultEqualityOperator,
    )


def test_local_equality_tester_find_operator_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect data type:"):
        LocalEqualityTester().find_operator(Mock(__mro__=[]))
