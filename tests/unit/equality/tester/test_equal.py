from __future__ import annotations

import math
from dataclasses import replace
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import EqualEqualityTester, EqualNanEqualityTester
from tests.unit.equality.utils import ExamplePair, explaining, messages, run_tester


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


class MyFloat:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self._value == other._value


class MyFloatNan:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        if equal_nan and math.isnan(self._value) and math.isnan(other._value):
            return True
        return self._value == other._value


CUSTOM_EQUAL_NAN_EQUAL = [
    pytest.param(ExamplePair(actual=MyFloatNan(4.2), expected=MyFloatNan(4.2)), id="float"),
    pytest.param(ExamplePair(actual=MyFloatNan(42), expected=MyFloatNan(42)), id="int"),
]


CUSTOM_EQUAL_NAN_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=MyFloatNan(4.2),
            expected=MyFloatNan(1),
            expected_message="objects are different:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloatNan(4.2),
            expected=MyFloatNan(float("nan")),
            expected_message="objects are different:",
        ),
        id="different values - NaN",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloatNan(4.2),
            expected="meow",
            expected_message="objects have different types:",
        ),
        id="float vs str",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloatNan(4.2), expected=None, expected_message="objects have different types:"
        ),
        id="float vs none",
    ),
]

CUSTOM_EQUAL_EQUAL = [
    pytest.param(ExamplePair(actual=MyFloat(4.2), expected=MyFloat(4.2)), id="float"),
    pytest.param(ExamplePair(actual=MyFloat(42), expected=MyFloat(42)), id="int"),
    *CUSTOM_EQUAL_NAN_EQUAL,
]


CUSTOM_EQUAL_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=MyFloat(4.2), expected=MyFloat(1), expected_message="objects are different:"
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloat(4.2), expected="meow", expected_message="objects have different types:"
        ),
        id="float vs str",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloat(4.2), expected=None, expected_message="objects have different types:"
        ),
        id="float vs none",
    ),
    *CUSTOM_EQUAL_NAN_NOT_EQUAL,
]

#########################################
#     Tests for EqualEqualityTester     #
#########################################


def test_equal_equality_tester_repr() -> None:
    assert repr(EqualEqualityTester()) == "EqualEqualityTester()"


def test_equal_equality_tester_str() -> None:
    assert str(EqualEqualityTester()) == "EqualEqualityTester()"


def test_equal_equality_tester_equal_true() -> None:
    assert EqualEqualityTester() == EqualEqualityTester()


def test_equal_equality_tester_equal_false_different_type() -> None:
    assert EqualEqualityTester() != 42


def test_equal_equality_tester_equal_false_different_type_child() -> None:
    class Child(EqualEqualityTester): ...

    assert EqualEqualityTester() != Child()


def test_equal_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = Mock()
    assert run_tester(EqualEqualityTester(), obj, obj, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_EQUAL)
def test_equal_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = EqualEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_EQUAL)
def test_equal_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = EqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_EQUAL)
def test_equal_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = EqualEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_EQUAL)
def test_equal_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = EqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NOT_EQUAL)
def test_equal_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = EqualEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NOT_EQUAL)
def test_equal_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = EqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NOT_EQUAL)
def test_equal_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = EqualEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NOT_EQUAL)
def test_equal_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = EqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


def test_equal_equality_tester_objects_are_equal_true_nan(config: EqualityConfig) -> None:
    config = replace(config, equal_nan=True)
    assert not run_tester(
        EqualEqualityTester(), MyFloat(float("nan")), MyFloat(float("nan")), config
    )


def test_equal_equality_tester_objects_are_equal_false_nan(config: EqualityConfig) -> None:
    assert not run_tester(
        EqualEqualityTester(), MyFloat(float("nan")), MyFloat(float("nan")), config
    )


############################################
#     Tests for EqualNanEqualityTester     #
############################################


def test_equal_nan_equality_tester_repr() -> None:
    assert repr(EqualNanEqualityTester()) == "EqualNanEqualityTester()"


def test_equal_nan_equality_tester_str() -> None:
    assert str(EqualNanEqualityTester()) == "EqualNanEqualityTester()"


def test_equal_nan_equality_tester_equal_true() -> None:
    assert EqualNanEqualityTester() == EqualNanEqualityTester()


def test_equal_nan_equality_tester_equal_false_different_type() -> None:
    assert EqualNanEqualityTester() != 42


def test_equal_nan_equality_tester_equal_false_different_type_child() -> None:
    class Child(EqualNanEqualityTester): ...

    assert EqualNanEqualityTester() != Child()


def test_equal_nan_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = Mock()
    assert run_tester(EqualNanEqualityTester(), obj, obj, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = EqualNanEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = EqualNanEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = EqualNanEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = EqualNanEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_NOT_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = EqualNanEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_NOT_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = EqualNanEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_NOT_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = EqualNanEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_NOT_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = EqualNanEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


def test_equal_nan_equality_tester_objects_are_equal_true_nan(config: EqualityConfig) -> None:
    config = replace(config, equal_nan=True)
    assert run_tester(
        EqualNanEqualityTester(), MyFloatNan(float("nan")), MyFloatNan(float("nan")), config
    )


def test_equal_nan_equality_tester_objects_are_equal_false_nan(config: EqualityConfig) -> None:
    assert not run_tester(
        EqualNanEqualityTester(), MyFloatNan(float("nan")), MyFloatNan(float("nan")), config
    )


class NoEqual:
    """An object without an ``equal`` method."""


def test_equal_equality_tester_object_without_equal_method(config: EqualityConfig) -> None:
    assert not run_tester(EqualEqualityTester(), NoEqual(), NoEqual(), config)


def test_equal_nan_equality_tester_object_without_equal_method(config: EqualityConfig) -> None:
    assert not run_tester(EqualNanEqualityTester(), NoEqual(), NoEqual(), config)
