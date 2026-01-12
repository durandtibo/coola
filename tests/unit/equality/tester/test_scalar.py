from __future__ import annotations

import logging

import pytest

from coola.equality.config import EqualityConfig2
from coola.equality.tester import ScalarEqualityTester
from tests.unit.equality.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig2:
    return EqualityConfig2()


FLOAT_EQUAL = [
    pytest.param(ExamplePair(actual=4.2, expected=4.2), id="positive"),
    pytest.param(ExamplePair(actual=0.0, expected=0.0), id="zero"),
    pytest.param(ExamplePair(actual=-4.2, expected=-4.2), id="negative"),
    pytest.param(ExamplePair(actual=float("inf"), expected=float("inf")), id="infinity"),
    pytest.param(ExamplePair(actual=float("-inf"), expected=float("-inf")), id="-infinity"),
]

FLOAT_NOT_EQUAL = [
    pytest.param(
        ExamplePair(actual=4.2, expected=1.0, expected_message="numbers are not equal:"),
        id="different values",
    ),
    pytest.param(
        ExamplePair(actual=4.2, expected="meow", expected_message="objects have different types:"),
        id="different types",
    ),
]

SCALAR_EQUAL = FLOAT_EQUAL
SCALAR_NOT_EQUAL = FLOAT_NOT_EQUAL

SCALAR_EQUAL_TOLERANCE = [
    # atol
    pytest.param(ExamplePair(actual=0, expected=1, atol=1.0), id="integer 0 atol=1"),
    pytest.param(ExamplePair(actual=1, expected=0, atol=1.0), id="integer 1 atol=1"),
    pytest.param(ExamplePair(actual=1, expected=2, atol=1.0), id="integer 2 atol=1"),
    pytest.param(ExamplePair(actual=1, expected=5, atol=10.0), id="integer 1 atol=10"),
    pytest.param(ExamplePair(actual=1.0, expected=1.0001, atol=1e-3), id="float + atol=1e-3"),
    pytest.param(ExamplePair(actual=1.0, expected=0.9999, atol=1e-3), id="float - atol=1e-3"),
    pytest.param(ExamplePair(actual=True, expected=False, atol=1.0), id="bool - atol=1"),
    # rtol
    pytest.param(ExamplePair(actual=0, expected=1, rtol=1.0), id="integer 0 rtol=1"),
    pytest.param(ExamplePair(actual=1, expected=0, rtol=1.0), id="integer 1 rtol=1"),
    pytest.param(ExamplePair(actual=1, expected=2, rtol=1.0), id="integer 2 rtol=1"),
    pytest.param(ExamplePair(actual=1, expected=5, rtol=10.0), id="integer 1 rtol=10"),
    pytest.param(ExamplePair(actual=1.0, expected=1.0001, rtol=1e-3), id="float + rtol=1e-3"),
    pytest.param(ExamplePair(actual=1.0, expected=0.9999, rtol=1e-3), id="float - rtol=1e-3"),
    pytest.param(ExamplePair(actual=True, expected=False, rtol=1.0), id="bool - rtol=1"),
]


##########################################
#     Tests for ScalarEqualityTester     #
##########################################


def test_scalar_equality_tester_repr() -> None:
    assert repr(ScalarEqualityTester()).startswith("ScalarEqualityTester(")


def test_scalar_equality_tester_str() -> None:
    assert str(ScalarEqualityTester()).startswith("ScalarEqualityTester(")


def test_scalar_equality_tester_equal_true() -> None:
    assert ScalarEqualityTester().equal(ScalarEqualityTester())


def test_scalar_equality_tester_equal_false_different_type() -> None:
    assert not ScalarEqualityTester().equal(123)


def test_scalar_equality_tester_equal_false_different_type_child() -> None:
    class Child(ScalarEqualityTester): ...

    assert not ScalarEqualityTester().equal(Child())


def test_scalar_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig2,
) -> None:
    x = 4.2
    assert ScalarEqualityTester().objects_are_equal(x, x, config)


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = ScalarEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = ScalarEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = ScalarEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = ScalarEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_scalar_equality_tester_objects_are_equal_nan(
    config: EqualityConfig2, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        ScalarEqualityTester().objects_are_equal(
            actual=float("nan"),
            expected=float("nan"),
            config=config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", SCALAR_EQUAL_TOLERANCE)
def test_scalar_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig2
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert ScalarEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )
