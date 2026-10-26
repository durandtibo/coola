from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola.equality import objects_are_allclose
from tests.integration.equality.checks.utils import EQUALITY_TESTER_FUNCTIONS, reported_messages
from tests.unit.equality.tester.test_scalar import (
    SCALAR_EQUAL,
    SCALAR_EQUAL_TOLERANCE,
    SCALAR_NOT_EQUAL,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.unit.equality.utils import ExamplePair


@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", SCALAR_EQUAL)
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
) -> None:
    assert function(example.actual, example.expected)


@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", SCALAR_EQUAL)
def test_objects_are_equal_true_does_not_log(
    function: Callable,
    example: ExamplePair,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", SCALAR_NOT_EQUAL)
def test_objects_are_equal_false(function: Callable, example: ExamplePair) -> None:
    assert not function(example.actual, example.expected)


@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", SCALAR_NOT_EQUAL)
def test_objects_are_equal_false_does_not_log(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("example", SCALAR_NOT_EQUAL)
def test_objects_are_equal_false_reports_the_difference(example: ExamplePair) -> None:
    assert reported_messages(example.actual, example.expected)[0].startswith(
        example.expected_message
    )


@pytest.mark.parametrize("example", SCALAR_EQUAL_TOLERANCE)
def test_objects_are_allclose_true_tolerance(example: ExamplePair) -> None:
    assert objects_are_allclose(
        example.actual, example.expected, atol=example.atol, rtol=example.rtol
    )


def test_objects_are_allclose_int_and_float_are_different_types() -> None:
    assert not objects_are_allclose(1, 1.0)


def test_objects_are_allclose_list_of_float_and_list_of_int_are_different_types() -> None:
    assert not objects_are_allclose([1.0], [1])
