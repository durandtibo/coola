from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig2
from coola.equality.tester import DefaultEqualityTester
from tests.unit.equality.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig2:
    return EqualityConfig2()


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

###########################################
#     Tests for DefaultEqualityTester     #
###########################################


def test_default_equality_tester_repr() -> None:
    assert repr(DefaultEqualityTester()) == "DefaultEqualityTester()"


def test_default_equality_tester_str() -> None:
    assert str(DefaultEqualityTester()) == "DefaultEqualityTester()"


def test_default_equality_tester_equal_true() -> None:
    assert DefaultEqualityTester().equal(DefaultEqualityTester())


def test_default_equality_tester_equal_false_different_type() -> None:
    assert not DefaultEqualityTester().equal(42)


def test_default_equality_tester_equal_false_different_type_child() -> None:
    class Child(DefaultEqualityTester): ...

    assert not DefaultEqualityTester().equal(Child())


def test_default_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig2,
) -> None:
    obj = Mock()
    assert DefaultEqualityTester().objects_are_equal(obj, obj, config)


@pytest.mark.parametrize("example", DEFAULT_EQUAL)
def test_default_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = DefaultEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", DEFAULT_EQUAL)
def test_default_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = DefaultEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
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
def test_default_equality_tester_objects_are_equal_false_scalar(
    caplog: pytest.LogCaptureFixture,
    actual: bool | float,
    expected: bool | float | None,
    config: EqualityConfig2,
) -> None:
    tester = DefaultEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(actual, expected, config)
        assert not caplog.messages


def test_default_equality_tester_objects_are_equal_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig2
) -> None:
    config.show_difference = True
    tester = DefaultEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(actual=1, expected=2, config=config)
        assert caplog.messages[0].startswith("objects are different:")


def test_default_equality_tester_objects_are_equal_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig2
) -> None:
    tester = DefaultEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(actual=[], expected=(), config=config)
        assert not caplog.messages


def test_default_equality_tester_objects_are_equal_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig2
) -> None:
    config.show_difference = True
    tester = DefaultEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(actual=[], expected=(), config=config)
        assert caplog.messages[0].startswith("objects have different types:")


@pytest.mark.parametrize("example", DEFAULT_NOT_EQUAL)
def test_default_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = DefaultEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", DEFAULT_NOT_EQUAL)
def test_default_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = DefaultEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)
