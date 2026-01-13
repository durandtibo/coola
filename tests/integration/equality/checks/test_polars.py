from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola.equality import objects_are_allclose
from coola.testing.fixtures import polars_available
from tests.integration.equality.checks.utils import EQUALITY_TESTER_FUNCTIONS
from tests.unit.equality.tester.test_polars import (
    POLARS_EQUAL,
    POLARS_EQUAL_TOLERANCE,
    POLARS_NOT_EQUAL,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.unit.equality.utils import ExamplePair


@polars_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", POLARS_EQUAL)
@pytest.mark.parametrize("show_difference", [True, False])
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
    show_difference: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert function(example.actual, example.expected, show_difference=show_difference)
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", POLARS_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", POLARS_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)


@polars_available
@pytest.mark.parametrize("example", POLARS_EQUAL_TOLERANCE)
def test_objects_are_allclose_true_tolerance(example: ExamplePair) -> None:
    assert objects_are_allclose(
        example.actual, example.expected, atol=example.atol, rtol=example.rtol
    )
