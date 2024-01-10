from __future__ import annotations

import logging

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators.scalar import (
    FloatEqualityComparator,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from tests.unit.equality.comparators.utils import ExamplePair


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


FLOAT_EQUAL = [
    pytest.param(ExamplePair(object1=4.2, object2=4.2), id="positive"),
    pytest.param(ExamplePair(object1=0.0, object2=0.0), id="zero"),
    pytest.param(ExamplePair(object1=-4.2, object2=-4.2), id="negative"),
    pytest.param(ExamplePair(object1=float("inf"), object2=float("inf")), id="infinity"),
    pytest.param(ExamplePair(object1=float("-inf"), object2=float("-inf")), id="-infinity"),
]


FLOAT_NOT_EQUAL = [
    pytest.param(
        ExamplePair(object1=4.2, object2=1.0, expected_message="numbers are not equal:"),
        id="different values",
    ),
    pytest.param(
        ExamplePair(object1=4.2, object2="meow", expected_message="objects have different types:"),
        id="different types",
    ),
]


#############################################
#     Tests for FloatEqualityComparator     #
#############################################


def test_float_equality_comparator_str() -> None:
    assert str(FloatEqualityComparator()).startswith("FloatEqualityComparator(")


def test_float_equality_comparator__eq__true() -> None:
    assert FloatEqualityComparator() == FloatEqualityComparator()


def test_float_equality_comparator__eq__false_different_type() -> None:
    assert FloatEqualityComparator() != 123


def test_float_equality_comparator_clone() -> None:
    op = FloatEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_float_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = 4.2
    assert FloatEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_float_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = FloatEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_float_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = FloatEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_float_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = FloatEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_float_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = FloatEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_float_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        FloatEqualityComparator().equal(
            object1=float("nan"),
            object2=float("nan"),
            config=config,
        )
        == equal_nan
    )


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        float: FloatEqualityComparator(),
    }
