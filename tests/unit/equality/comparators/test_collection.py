from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators import SequenceEqualityComparator
from coola.equality.comparators.collection import (
    MappingEqualityComparator,
    get_type_comparator_mapping,
)
from coola.testers import EqualityTester
from coola.testing import numpy_available
from coola.utils import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


###############################################
#     Tests for MappingEqualityComparator     #
###############################################


def test_mapping_equality_comparator_str() -> None:
    assert str(MappingEqualityComparator()) == "MappingEqualityComparator()"


@numpy_available
def test_mapping_equality_comparator__eq__true() -> None:
    assert MappingEqualityComparator() == MappingEqualityComparator()


@numpy_available
def test_mapping_equality_comparator__eq__false() -> None:
    assert MappingEqualityComparator() != 123


def test_mapping_equality_comparator_clone() -> None:
    op = MappingEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_mapping_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = [1, 2, 3]
    assert MappingEqualityComparator().equal(obj, obj, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({}, {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
        ({"a": 1, "b": {"k": 1}}, {"a": 1, "b": {"k": 1}}),
    ],
)
def test_mapping_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1, object2, config)
        assert not caplog.messages


def test_mapping_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1={"a": 1, "b": 2}, object2={"a": 1, "b": 2}, config=config)
        assert not caplog.messages


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
        ({"a": 1, "b": {"k": 1}}, {"a": 1, "b": {"k": 2}}),
    ],
)
def test_mapping_equality_comparator_equal_false_different_value(
    caplog: pytest.LogCaptureFixture, object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1, object2, config)
        assert not caplog.messages


def test_mapping_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1={"a": 1, "b": 2}, object2={"a": 1, "b": 3}, config=config
        )
        assert caplog.messages[-1].startswith("mappings have at least one different value:")


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}),
        ({}, {"a": 1, "b": 2}),
    ],
)
def test_mapping_equality_comparator_equal_false_different_length(
    caplog: pytest.LogCaptureFixture, object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1, object2, config)
        assert not caplog.messages


def test_mapping_equality_comparator_equal_false_different_length_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1={"a": 1, "b": 2}, object2={"a": 1, "b": 2, "c": 3}, config=config
        )
        assert caplog.messages[0].startswith("objects have different lengths:")


def test_mapping_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1={}, object2=(), config=config)
        assert not caplog.messages


def test_mapping_equality_comparator_equal_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = MappingEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1={}, object2=(), config=config)
        assert caplog.messages[0].startswith("objects have different types:")


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": np.ones((2, 3)), "b": np.zeros(2)}, {"a": np.ones((2, 3)), "b": np.zeros(2)}),
        (
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
        ),
    ],
)
def test_mapping_equality_comparator_equal_true_numpy(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert MappingEqualityComparator().equal(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ({"a": np.ones((2, 3)), "b": np.zeros(2)}, {"a": np.ones((2, 3)), "b": np.ones(2)}),
        (
            {"a": np.ones((2, 3)), "b": {"k": np.zeros(2)}},
            {"a": np.ones((2, 3)), "b": {"k": np.ones(2)}},
        ),
    ],
)
def test_mapping_equality_comparator_equal_false_numpy(
    object1: Mapping, object2: Mapping, config: EqualityConfig
) -> None:
    assert not MappingEqualityComparator().equal(object1, object2, config)


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


def test_sequence_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = [1, 2, 3]
    assert SequenceEqualityComparator().equal(obj, obj, config)


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
    caplog: pytest.LogCaptureFixture, object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1, object2, config)
        assert not caplog.messages


def test_sequence_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=[1], object2=[1], config=config)
        assert not caplog.messages


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
    caplog: pytest.LogCaptureFixture, object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=object1, object2=object2, config=config)
        assert not caplog.messages


def test_sequence_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=[1, 2], object2=[1, 3], config=config)
        assert caplog.messages[-1].startswith("sequences have at least one different value:")


@pytest.mark.parametrize(("object1", "object2"), [([1, 2, 3], [1, 2, 3, 4]), ((), ("abc", "def"))])
def test_sequence_equality_comparator_equal_false_different_length(
    caplog: pytest.LogCaptureFixture, object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=object1, object2=object2, config=config)
        assert not caplog.messages


def test_sequence_equality_comparator_equal_false_different_length_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=[1, 2, 3], object2=[1, 2], config=config)
        assert caplog.messages[0].startswith("objects have different lengths:")


def test_sequence_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=[], object2=(), config=config)
        assert not caplog.messages


def test_sequence_equality_comparator_equal_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = SequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=[], object2=(), config=config)
        assert caplog.messages[0].startswith("objects have different types:")


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


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        Mapping: MappingEqualityComparator(),
        Sequence: SequenceEqualityComparator(),
        dict: MappingEqualityComparator(),
        list: SequenceEqualityComparator(),
        tuple: SequenceEqualityComparator(),
    }
