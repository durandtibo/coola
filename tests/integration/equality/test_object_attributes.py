from __future__ import annotations

import functools

import pytest

from coola.equality import assert_objects_equal, compare, objects_are_allclose, objects_are_equal
from coola.equality.tester import (
    DefaultEqualityTester,
    EqualityTesterRegistry,
    ObjectAttributesEqualityTester,
    ScalarEqualityTester,
    SequenceEqualityTester,
)


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


@pytest.fixture
def registry() -> EqualityTesterRegistry:
    tester = ObjectAttributesEqualityTester()
    return EqualityTesterRegistry(
        {
            object: DefaultEqualityTester(),
            float: ScalarEqualityTester(),
            int: ScalarEqualityTester(),
            list: SequenceEqualityTester(),
            tuple: SequenceEqualityTester(),
            Point: tester,
            BaseException: tester,
            functools.partial: tester,
        }
    )


def test_plain_objects_are_different_without_the_tester() -> None:
    registry = EqualityTesterRegistry({object: DefaultEqualityTester()})
    assert not objects_are_equal(Point(1.0, 2.0), Point(1.0, 2.0), registry=registry)


def test_plain_objects_are_equal_with_the_tester(registry: EqualityTesterRegistry) -> None:
    assert objects_are_equal(Point(1.0, 2.0), Point(1.0, 2.0), registry=registry)


def test_plain_objects_are_different_if_an_attribute_differs(
    registry: EqualityTesterRegistry,
) -> None:
    assert not objects_are_equal(Point(1.0, 2.0), Point(1.0, 3.0), registry=registry)


def test_plain_objects_are_close_within_the_tolerance(registry: EqualityTesterRegistry) -> None:
    assert objects_are_allclose(Point(1.0, 2.0), Point(1.0, 2.0 + 1e-9), registry=registry)


def test_plain_objects_in_a_list(registry: EqualityTesterRegistry) -> None:
    assert objects_are_equal([Point(1.0, 2.0)], [Point(1.0, 2.0)], registry=registry)


def test_exceptions_with_the_same_args_are_equal(registry: EqualityTesterRegistry) -> None:
    assert objects_are_equal(ValueError("a"), ValueError("a"), registry=registry)


def test_exceptions_with_different_args_are_different(registry: EqualityTesterRegistry) -> None:
    assert not objects_are_equal(ValueError("a"), ValueError("b"), registry=registry)


def test_exceptions_of_different_classes_are_different(registry: EqualityTesterRegistry) -> None:
    assert not objects_are_equal(ValueError("a"), KeyError("a"), registry=registry)


def test_partials_with_the_same_arguments_are_equal(registry: EqualityTesterRegistry) -> None:
    assert objects_are_equal(
        functools.partial(print, 1), functools.partial(print, 1), registry=registry
    )


def test_partials_with_different_arguments_are_different(
    registry: EqualityTesterRegistry,
) -> None:
    assert not objects_are_equal(
        functools.partial(print, 1), functools.partial(print, 2), registry=registry
    )


def test_compare_reports_the_path_of_the_attribute(registry: EqualityTesterRegistry) -> None:
    result = compare([Point(1.0, 2.0)], [Point(1.0, 3.0)], registry=registry)
    assert result.format_path("actual") == "actual[0].y"


def test_assert_objects_equal_reports_the_path_of_the_attribute(
    registry: EqualityTesterRegistry,
) -> None:
    with pytest.raises(AssertionError, match=r"actual\[0\]\.y"):
        assert_objects_equal([Point(1.0, 2.0)], [Point(1.0, 3.0)], registry=registry)
