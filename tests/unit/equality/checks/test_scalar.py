from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from tests.unit.equality.checks.test_default import COMPARATOR_FUNCTIONS
from tests.unit.equality.comparators.test_scalar import SCALAR_EQUAL, SCALAR_NOT_EQUAL

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.unit.equality.comparators.utils import ExamplePair


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", SCALAR_EQUAL)
@pytest.mark.parametrize("show_difference", [True, False])
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
    show_difference: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert function(example.object1, example.object2, show_difference)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", SCALAR_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.object1, example.object2)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", SCALAR_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.object1, example.object2, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
