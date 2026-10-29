from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import pytest

from coola.equality import objects_are_allclose
from coola.testing.marks import pyarrow_available
from tests.integration.equality.checks.utils import EQUALITY_TESTER_FUNCTIONS, reported_messages
from tests.unit.equality.tester.test_pyarrow import (
    PYARROW_EQUAL,
    PYARROW_EQUAL_TOLERANCE,
    PYARROW_NOT_EQUAL,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from tests.unit.equality.utils import ExamplePair


@pytest.fixture(autouse=True)
def _filter_warning() -> Generator[None]:
    warnings.filterwarnings(
        action="ignore",
        category=RuntimeWarning,
        message=".*is ignored because it is not supported.*",
    )
    yield
    warnings.resetwarnings()


@pyarrow_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
) -> None:
    assert function(example.actual, example.expected)


@pyarrow_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_objects_are_equal_true_does_not_log(
    function: Callable,
    example: ExamplePair,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        function(example.actual, example.expected)
        assert not caplog.messages


@pyarrow_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_objects_are_equal_false(function: Callable, example: ExamplePair) -> None:
    assert not function(example.actual, example.expected)


@pyarrow_available
@pytest.mark.parametrize("function", EQUALITY_TESTER_FUNCTIONS)
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_objects_are_equal_false_does_not_log(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        function(example.actual, example.expected)
        assert not caplog.messages


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_objects_are_equal_false_reports_the_difference(example: ExamplePair) -> None:
    assert reported_messages(example.actual, example.expected)[0].startswith(
        example.expected_message
    )


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL_TOLERANCE)
def test_objects_are_allclose_true_tolerance(example: ExamplePair) -> None:
    assert objects_are_allclose(
        example.actual, example.expected, atol=example.atol, rtol=example.rtol
    )
