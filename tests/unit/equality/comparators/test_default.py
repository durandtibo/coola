from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.comparators import DefaultEqualityComparator
from coola.equality.comparators.default import get_type_comparator_mapping


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


###############################################
#     Tests for DefaultEqualityComparator     #
###############################################


def test_default_equality_comparator_str() -> None:
    assert str(DefaultEqualityComparator()) == "DefaultEqualityComparator()"


def test_default_equality_comparator__eq__true() -> None:
    assert DefaultEqualityComparator() == DefaultEqualityComparator()


def test_default_equality_comparator__eq__false() -> None:
    assert DefaultEqualityComparator() != 123


def test_default_equality_comparator_clone() -> None:
    op = DefaultEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (1, 1),
        (0, 0),
        (-1, -1),
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, -1.0),
        (True, True),
        (False, False),
        (None, None),
    ],
)
def test_default_equality_comparator_equal_true_scalar(
    object1: bool | float | None, object2: bool | float | None, config: EqualityConfig
) -> None:
    assert DefaultEqualityComparator().equal(object1, object2, config)


def test_default_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = Mock()
    assert DefaultEqualityComparator().equal(obj, obj, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (1, 2),
        (1.0, 2.0),
        (True, False),
        (1, 1.0),
        (1, True),
        (1.0, True),
        (1.0, None),
    ],
)
def test_default_equality_comparator_equal_false_scalar(
    object1: bool | float, object2: bool | float | None, config: EqualityConfig
) -> None:
    assert not DefaultEqualityComparator().equal(object1, object2, config)


def test_default_equality_comparator_equal_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=1, object2=2, config=config)
        assert caplog.messages[0].startswith("objects are different:")


def test_default_equality_comparator_equal_different_type(config: EqualityConfig) -> None:
    assert not DefaultEqualityComparator().equal(object1=[], object2=(), config=config)


def test_default_equality_comparator_equal_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=[], object2=(), config=config)
        assert caplog.messages[0].startswith("objects have different types:")


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {object: DefaultEqualityComparator()}
