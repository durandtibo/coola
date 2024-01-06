from __future__ import annotations

import logging
from collections.abc import Sequence
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators import SequenceEqualityComparator
from coola.equality.comparators.collection import get_type_comparator_mapping
from coola.testers import EqualityTester
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


################################################
#     Tests for SequenceEqualityComparator     #
################################################


def test_sequence_equality_comparator_str() -> None:
    assert str(SequenceEqualityComparator()) == "SequenceEqualityComparator()"


@numpy_available
def test_sequence_equality_comparator__eq__true() -> None:
    assert SequenceEqualityComparator() == SequenceEqualityComparator()


@numpy_available
def test_sequence_equality_comparator__eq__false() -> None:
    assert SequenceEqualityComparator() != 123


def test_sequence_equality_comparator_clone() -> None:
    op = SequenceEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([], []),
        ((), ()),
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), (1, 2, 3)),
        (["abc", "def"], ["abc", "def"]),
        (("abc", "def"), ("abc", "def")),
        ([0, ("a", "b", "c"), 2], [0, ("a", "b", "c"), 2]),
    ],
)
def test_sequence_equality_comparator_equal_true(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert SequenceEqualityComparator().equal(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.zeros(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.zeros(2))),
    ],
)
def test_sequence_equality_comparator_equal_true_numpy(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert SequenceEqualityComparator().equal(object1, object2, config)


def test_sequence_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = [1, 2, 3]
    assert SequenceEqualityComparator().equal(obj, obj, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.ones(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.ones(2))),
    ],
)
def test_sequence_equality_comparator_equal_false_numpy(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceEqualityComparator().equal(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([1, 2, 3], [1, 2, 4]),
        ((1, 2, 3), (1, 2, 4)),
        (["abc", "deg"], ["abc", "def"]),
        (("abc", "deg"), ("abc", "def")),
        ([0, ("a", "b", "c"), 2], [0, ("a", "b", "d"), 2]),
    ],
)
def test_sequence_equality_comparator_equal_false_different_value(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceEqualityComparator().equal(object1, object2, config)


def test_sequence_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=[1, 2], object2=[1, 3], config=config)
        assert caplog.messages[-1].startswith("sequences have at least one different value:")


@torch_available
@pytest.mark.parametrize(("object1", "object2"), [([1, 2, 3], [1, 2, 3, 4]), ((), ("abc", "def"))])
def test_sequence_equality_comparator_equal_false_different_length(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceEqualityComparator().equal(object1, object2, config)


def test_sequence_equality_comparator_equal_false_different_length_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=[1, 2, 3], object2=[1, 2], config=config)
        assert caplog.messages[0].startswith("objects have different lengths:")


def test_sequence_equality_comparator_equal_false_different_type(config: EqualityConfig) -> None:
    assert not SequenceEqualityComparator().equal(object1=[], object2=(), config=config)


def test_sequence_equality_comparator_equal_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=[], object2=(), config=config)
        assert caplog.messages[0].startswith("objects have different types:")


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        Sequence: SequenceEqualityComparator(),
        list: SequenceEqualityComparator(),
        tuple: SequenceEqualityComparator(),
    }
