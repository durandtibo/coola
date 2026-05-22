from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import TolerantEqualEqualityTester
from tests.unit.equality.handler.test_tolerant import MyFloat
from tests.unit.equality.utils import ExamplePair


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


################################################
#     Tests for TolerantEqualEqualityTester    #
################################################


def test_tolerant_equal_equality_tester_repr() -> None:
    assert repr(TolerantEqualEqualityTester()) == "TolerantEqualEqualityTester()"


def test_tolerant_equal_equality_tester_str() -> None:
    assert str(TolerantEqualEqualityTester()) == "TolerantEqualEqualityTester()"


def test_tolerant_equal_equality_tester_equal_true() -> None:
    assert TolerantEqualEqualityTester().equal(TolerantEqualEqualityTester())


def test_tolerant_equal_equality_tester_equal_false_different_type() -> None:
    assert not TolerantEqualEqualityTester().equal(42)


def test_tolerant_equal_equality_tester_equal_false_different_type_child() -> None:
    class Child(TolerantEqualEqualityTester): ...

    assert not TolerantEqualEqualityTester().equal(Child())


def test_tolerant_equal_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = Mock()
    assert TolerantEqualEqualityTester().objects_are_equal(obj, obj, config)


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = TolerantEqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = TolerantEqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = TolerantEqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL)
def test_tolerant_equal_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = TolerantEqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)


def test_tolerant_equal_equality_tester_objects_are_equal_true_nan(
    config: EqualityConfig,
) -> None:
    config.equal_nan = True
    assert TolerantEqualEqualityTester().objects_are_equal(
        actual=MyFloat(float("nan")), expected=MyFloat(float("nan")), config=config
    )


def test_tolerant_equal_equality_tester_objects_are_equal_false_nan(
    config: EqualityConfig,
) -> None:
    assert not TolerantEqualEqualityTester().objects_are_equal(
        actual=MyFloat(float("nan")), expected=MyFloat(float("nan")), config=config
    )


@pytest.mark.parametrize("example", TOLERANT_EQUAL_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_true_with_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.atol = 0.5
    tester = TolerantEqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_false_with_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.atol = 0.5
    tester = TolerantEqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", TOLERANT_EQUAL_NOT_EQUAL_WITH_TOLERANCE)
def test_tolerant_equal_equality_tester_objects_are_equal_false_show_difference_with_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.atol = 0.5
    config.show_difference = True
    tester = TolerantEqualEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)
