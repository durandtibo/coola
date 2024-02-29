from __future__ import annotations

import logging

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators.scalar import (
    ScalarEqualityComparator,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from tests.unit.equality.comparators.utils import ExamplePair


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


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


##############################################
#     Tests for ScalarEqualityComparator     #
##############################################


def test_scalar_equality_comparator_str() -> None:
    assert str(ScalarEqualityComparator()).startswith("ScalarEqualityComparator(")


def test_scalar_equality_comparator__eq__true() -> None:
    assert ScalarEqualityComparator() == ScalarEqualityComparator()


def test_scalar_equality_comparator__eq__false_different_type() -> None:
    assert ScalarEqualityComparator() != 123


def test_scalar_equality_comparator_clone() -> None:
    op = ScalarEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_scalar_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = 4.2
    assert ScalarEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_scalar_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ScalarEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_scalar_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ScalarEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_scalar_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ScalarEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_scalar_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ScalarEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_scalar_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        ScalarEqualityComparator().equal(
            actual=float("nan"),
            expected=float("nan"),
            config=config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", SCALAR_EQUAL_TOLERANCE)
def test_scalar_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert ScalarEqualityComparator().equal(
        actual=example.actual, expected=example.expected, config=config
    )


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        float: ScalarEqualityComparator(),
        int: ScalarEqualityComparator(),
    }
