from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola.equality import objects_are_allclose
from coola.testing.marks import xarray_available
from tests.integration.equality.checks.utils import EQUALITY_TESTER_FUNCTIONS, reported_messages
from tests.unit.equality.tester.test_xarray import (
    XARRAY_EQUAL,
    XARRAY_EQUAL_TOLERANCE,
    XARRAY_NOT_EQUAL,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.unit.equality.utils import ExamplePair


@xarray_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", XARRAY_EQUAL)
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
) -> None:
    assert function(example.actual, example.expected)


@xarray_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", XARRAY_EQUAL)
def test_objects_are_equal_true_does_not_log(
    function: Callable,
    example: ExamplePair,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        function(example.actual, example.expected)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", XARRAY_NOT_EQUAL)
def test_objects_are_equal_false(function: Callable, example: ExamplePair) -> None:
    assert not function(example.actual, example.expected)


@xarray_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", XARRAY_NOT_EQUAL)
def test_objects_are_equal_false_does_not_log(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        function(example.actual, example.expected)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_NOT_EQUAL)
def test_objects_are_equal_false_reports_the_difference(example: ExamplePair) -> None:
    assert reported_messages(example.actual, example.expected)[0].startswith(
        example.expected_message
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_EQUAL_TOLERANCE)
def test_objects_are_allclose_true_tolerance(example: ExamplePair) -> None:
    assert objects_are_allclose(
        example.actual, example.expected, atol=example.atol, rtol=example.rtol
    )
