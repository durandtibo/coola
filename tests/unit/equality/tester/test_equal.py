from __future__ import annotations

import logging
import math
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import EqualEqualityTester, EqualNanEqualityTester
from tests.unit.equality.utils import ExamplePair


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
            expected_message="objects are not equal:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloatNan(4.2),
            expected=MyFloatNan(float("nan")),
            expected_message="objects are not equal:",
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
            actual=MyFloat(4.2), expected=MyFloat(1), expected_message="objects are not equal:"
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
    assert EqualEqualityTester().equal(EqualEqualityTester())


def test_equal_equality_tester_equal_false_different_type() -> None:
    assert not EqualEqualityTester().equal(42)


def test_equal_equality_tester_equal_false_different_type_child() -> None:
    class Child(EqualEqualityTester): ...

    assert not EqualEqualityTester().equal(Child())


def test_equal_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = Mock()
    assert EqualEqualityTester().objects_are_equal(obj, obj, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_EQUAL)
def test_equal_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = EqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", CUSTOM_EQUAL_EQUAL)
def test_equal_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = EqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NOT_EQUAL)
def test_equal_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = EqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NOT_EQUAL)
def test_equal_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = EqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)


def test_equal_equality_tester_objects_are_equal_true_nan(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert not EqualEqualityTester().objects_are_equal(
        actual=MyFloat(float("nan")), expected=MyFloat(float("nan")), config=config
    )


def test_equal_equality_tester_objects_are_equal_false_nan(config: EqualityConfig) -> None:
    assert not EqualEqualityTester().objects_are_equal(
        actual=MyFloat(float("nan")), expected=MyFloat(float("nan")), config=config
    )


############################################
#     Tests for EqualNanEqualityTester     #
############################################


def test_equal_nan_equality_tester_repr() -> None:
    assert repr(EqualNanEqualityTester()) == "EqualNanEqualityTester()"


def test_equal_nan_equality_tester_str() -> None:
    assert str(EqualNanEqualityTester()) == "EqualNanEqualityTester()"


def test_equal_nan_equality_tester_equal_true() -> None:
    assert EqualNanEqualityTester().equal(EqualNanEqualityTester())


def test_equal_nan_equality_tester_equal_false_different_type() -> None:
    assert not EqualNanEqualityTester().equal(42)


def test_equal_nan_equality_tester_equal_false_different_type_child() -> None:
    class Child(EqualNanEqualityTester): ...

    assert not EqualNanEqualityTester().equal(Child())


def test_equal_nan_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = Mock()
    assert EqualNanEqualityTester().objects_are_equal(obj, obj, config)


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = EqualNanEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = EqualNanEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_NOT_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = EqualNanEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", CUSTOM_EQUAL_NAN_NOT_EQUAL)
def test_equal_nan_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = EqualNanEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)


def test_equal_nan_equality_tester_objects_are_equal_true_nan(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert EqualNanEqualityTester().objects_are_equal(
        actual=MyFloatNan(float("nan")), expected=MyFloatNan(float("nan")), config=config
    )


def test_equal_nan_equality_tester_objects_are_equal_false_nan(config: EqualityConfig) -> None:
    assert not EqualNanEqualityTester().objects_are_equal(
        actual=MyFloatNan(float("nan")), expected=MyFloatNan(float("nan")), config=config
    )
