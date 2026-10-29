from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import DefaultEqualityTester
from tests.unit.equality.utils import ExamplePair, explaining, messages, run_tester


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


@dataclass
class Person:
    name: str
    age: int


DEFAULT_EQUAL = [
    pytest.param(ExamplePair(actual=4.2, expected=4.2), id="float"),
    pytest.param(ExamplePair(actual=42, expected=42), id="int"),
    pytest.param(ExamplePair(actual=complex(4, 2), expected=complex(4, 2)), id="complex"),
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
]

# Other testers are more specific for these types
DEFAULT_SPECIFIC_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=complex(4, 2), expected=complex(4, 3), expected_message="objects are different:"
        ),
        id="different values - complex",
    ),
    pytest.param(
        ExamplePair(
            actual=Person(name="Alice", age=30),
            expected=Person(name="Bob", age=30),
            expected_message="objects are different:",
        ),
        id="dataclass with different values",
    ),
    pytest.param(
        ExamplePair(actual=1, expected=2, expected_message="objects are different:"),
        id="different values - int",
    ),
    pytest.param(
        ExamplePair(actual=1.0, expected=2.0, expected_message="objects are different:"),
        id="different values - float",
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
    assert DefaultEqualityTester() == DefaultEqualityTester()


def test_default_equality_tester_equal_false_different_type() -> None:
    assert DefaultEqualityTester() != 42


def test_default_equality_tester_equal_false_different_type_child() -> None:
    class Child(DefaultEqualityTester): ...

    assert DefaultEqualityTester() != Child()


def test_default_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = Mock()
    assert run_tester(DefaultEqualityTester(), obj, obj, config)


@pytest.mark.parametrize("example", DEFAULT_EQUAL)
def test_default_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = DefaultEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", DEFAULT_EQUAL)
def test_default_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = DefaultEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", DEFAULT_EQUAL)
def test_default_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = DefaultEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", DEFAULT_EQUAL)
def test_default_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = DefaultEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


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
    actual: bool | float,
    expected: bool | float | None,
    config: EqualityConfig,
) -> None:
    tester = DefaultEqualityTester()
    assert not run_tester(tester, actual, expected, config)


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
def test_default_equality_tester_objects_are_equal_false_scalar_does_not_log(
    actual: bool | float,
    expected: bool | float | None,
    config: EqualityConfig,
) -> None:
    tester = DefaultEqualityTester()
    run_tester(tester, actual, expected, config)
    assert not messages(config)


def test_default_equality_tester_objects_are_equal_different_value_explaining(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = DefaultEqualityTester()
    assert not run_tester(tester, 1, 2, config)


def test_default_equality_tester_objects_are_equal_different_value_explaining_reports(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = DefaultEqualityTester()
    run_tester(tester, 1, 2, config)
    assert messages(config)[0].startswith("objects are different:")


def test_default_equality_tester_objects_are_equal_different_type(config: EqualityConfig) -> None:
    tester = DefaultEqualityTester()
    assert not run_tester(tester, [], (), config)


def test_default_equality_tester_objects_are_equal_different_type_does_not_log(
    config: EqualityConfig,
) -> None:
    tester = DefaultEqualityTester()
    run_tester(tester, [], (), config)
    assert not messages(config)


def test_default_equality_tester_objects_are_equal_different_type_explaining(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = DefaultEqualityTester()
    assert not run_tester(tester, [], (), config)


def test_default_equality_tester_objects_are_equal_different_type_explaining_reports(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = DefaultEqualityTester()
    run_tester(tester, [], (), config)
    assert messages(config)[0].startswith("objects have different types:")


@pytest.mark.parametrize("example", DEFAULT_NOT_EQUAL + DEFAULT_SPECIFIC_NOT_EQUAL)
def test_default_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = DefaultEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", DEFAULT_NOT_EQUAL + DEFAULT_SPECIFIC_NOT_EQUAL)
def test_default_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = DefaultEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", DEFAULT_NOT_EQUAL + DEFAULT_SPECIFIC_NOT_EQUAL)
def test_default_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = DefaultEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", DEFAULT_NOT_EQUAL + DEFAULT_SPECIFIC_NOT_EQUAL)
def test_default_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = DefaultEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


class RaisingEq:
    def __eq__(self, other: object) -> bool:
        msg = "cannot compare"
        raise ValueError(msg)

    def __hash__(self) -> int:
        return 0


def test_default_equality_tester_objects_are_equal_eq_raises_a_type_error(
    config: EqualityConfig,
) -> None:
    with pytest.raises(TypeError, match="Register an equality tester for this type"):
        run_tester(DefaultEqualityTester(), RaisingEq(), RaisingEq(), config)
