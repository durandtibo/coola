from __future__ import annotations

import math
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import TolerantEqualEqualityTester
from tests.unit.equality.utils import ExamplePair, messages, run_tester


class MyFloat:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def allclose(
        self,
        other: object,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        if type(other) is not type(self):
            return False
        if equal_nan and math.isnan(self._value) and math.isnan(other._value):
            return True
        return math.isclose(self._value, other._value, rel_tol=rtol, abs_tol=atol)

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        if equal_nan and math.isnan(self._value) and math.isnan(other._value):
            return True
        return self._value == other._value


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


TOLERANT_EQUAL_EQUAL = [
    pytest.param(ExamplePair(actual=MyFloat(4.2), expected=MyFloat(4.2)), id="float"),
    pytest.param(ExamplePair(actual=MyFloat(42), expected=MyFloat(42)), id="int"),
]

TOLERANT_EQUAL_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=MyFloat(4.2),
            expected=MyFloat(1),
            expected_message="objects are different:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloat(4.2),
            expected=MyFloat(float("nan")),
            expected_message="objects are different:",
        ),
        id="different values - NaN",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloat(4.2),
            expected="meow",
            expected_message="objects have different types:",
        ),
        id="float vs str",
    ),
    pytest.param(
        ExamplePair(
            actual=MyFloat(4.2),
            expected=None,
            expected_message="objects have different types:",
        ),
        id="float vs none",
    ),
]

TOLERANT_EQUAL_EQUAL_WITH_TOLERANCE = [
    pytest.param(
        ExamplePair(actual=MyFloat(1.0), expected=MyFloat(1.4)),
        id="within atol",
    ),
    pytest.param(
        ExamplePair(actual=MyFloat(1.0), expected=MyFloat(1.4)),
        id="within rtol",
    ),
]

TOLERANT_EQUAL_NOT_EQUAL_WITH_TOLERANCE = [
    pytest.param(
        ExamplePair(
            actual=MyFloat(1.0),
            expected=MyFloat(2.0),
            expected_message="objects are different:",
        ),
        id="outside tolerance",
    ),
]


#################################################
#     Tests for TolerantEqualEqualityTester     #
#################################################


def test_tolerant_equal_equality_tester_repr() -> None:
    assert repr(TolerantEqualEqualityTester()) == "TolerantEqualEqualityTester()"


def test_tolerant_equal_equality_tester_str() -> None:
    assert str(TolerantEqualEqualityTester()) == "TolerantEqualEqualityTester()"


def test_tolerant_equal_equality_tester_equal_true() -> None:
    assert TolerantEqualEqualityTester() == TolerantEqualEqualityTester()


def test_tolerant_equal_equality_tester_equal_false_different_type() -> None:
    assert TolerantEqualEqualityTester() != 42


def test_tolerant_equal_equality_tester_equal_false_different_type_child() -> None:
    class Child(TolerantEqualEqualityTester): ...

    assert TolerantEqualEqualityTester() != Child()


def test_tolerant_equal_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = Mock()
    assert run_tester(TolerantEqualEqualityTester(), obj, obj, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TolerantEqualEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TolerantEqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = TolerantEqualEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = TolerantEqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TolerantEqualEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TolerantEqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = TolerantEqualEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = TolerantEqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


def test_tolerant_equal_equality_tester_objects_are_equal_true_nan(
    config: EqualityConfig,
) -> None:
    config.equal_nan = True
    assert run_tester(
        TolerantEqualEqualityTester(), MyFloat(float("nan")), MyFloat(float("nan")), config
    )


def test_tolerant_equal_equality_tester_objects_are_equal_false_nan(
    config: EqualityConfig,
) -> None:
    assert not run_tester(
        TolerantEqualEqualityTester(), MyFloat(float("nan")), MyFloat(float("nan")), config
    )


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_true_with_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = 0.5
    tester = TolerantEqualEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_true_with_tolerance_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = 0.5
    tester = TolerantEqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_false_with_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = 0.5
    tester = TolerantEqualEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_false_with_tolerance_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = 0.5
    tester = TolerantEqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_false_explaining_with_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = 0.5
    config.reports = []
    tester = TolerantEqualEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_false_explaining_with_tolerance_logs(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = 0.5
    config.reports = []
    tester = TolerantEqualEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


class NoMethods:
    """An object that implements neither ``allclose`` nor ``equal``."""


def test_tolerant_equal_equality_tester_object_without_methods(config: EqualityConfig) -> None:
    assert not run_tester(TolerantEqualEqualityTester(), NoMethods(), NoMethods(), config)
