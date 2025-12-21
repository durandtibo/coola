from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators import DefaultEqualityComparator
from coola.equality.comparators.default import get_type_comparator_mapping
from coola.equality.testers import EqualityTester
from tests.unit.equality.comparators.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


@dataclass
class Person:
    name: str
    age: int


DEFAULT_EQUAL = [
    pytest.param(ExamplePair(actual=4.2, expected=4.2), id="float"),
    pytest.param(ExamplePair(actual=42, expected=42), id="int"),
    pytest.param(ExamplePair(actual="abc", expected="abc"), id="str"),
    pytest.param(ExamplePair(actual=True, expected=True), id="bool"),
    pytest.param(ExamplePair(actual=None, expected=None), id="none"),
    pytest.param(
        ExamplePair(actual=Person(name="Alice", age=30), expected=Person(name="Alice", age=30)),
        id="dataclass",
    ),
]


DEFAULT_NOT_EQUAL = [
    pytest.param(
        ExamplePair(actual="abc", expected="def", expected_message="objects are different:"),
        id="different values - str",
    ),
    pytest.param(
        ExamplePair(actual=4.2, expected="meow", expected_message="objects have different types:"),
        id="float vs str",
    ),
    pytest.param(
        ExamplePair(actual=1.0, expected=1, expected_message="objects have different types:"),
        id="float vs int",
    ),
    pytest.param(
        ExamplePair(actual=1.0, expected=None, expected_message="objects have different types:"),
        id="float vs none",
    ),
    pytest.param(
        ExamplePair(
            actual=Person(name="Alice", age=30),
            expected=Person(name="Bob", age=30),
            expected_message="objects are different:",
        ),
        id="dataclass with different values",
    ),
]

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


def test_default_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = Mock()
    assert DefaultEqualityComparator().equal(obj, obj, config)


@pytest.mark.parametrize("example", DEFAULT_EQUAL)
def test_default_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", DEFAULT_EQUAL)
def test_default_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize(
    ("actual", "expected"),
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
    caplog: pytest.LogCaptureFixture,
    actual: bool | float,
    expected: bool | float | None,
    config: EqualityConfig,
) -> None:
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual, expected, config)
        assert not caplog.messages


def test_default_equality_comparator_equal_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=1, expected=2, config=config)
        assert caplog.messages[0].startswith("objects are different:")


def test_default_equality_comparator_equal_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=[], expected=(), config=config)
        assert not caplog.messages


def test_default_equality_comparator_equal_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=[], expected=(), config=config)
        assert caplog.messages[0].startswith("objects have different types:")


@pytest.mark.parametrize("example", DEFAULT_NOT_EQUAL)
def test_default_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", DEFAULT_NOT_EQUAL)
def test_default_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = DefaultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {object: DefaultEqualityComparator()}
