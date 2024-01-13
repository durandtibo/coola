from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola import objects_are_close
from coola.testing import xarray_available
from tests.unit.equality.checks.test_default import COMPARATOR_FUNCTIONS
from tests.unit.equality.comparators.test_xarray import (
    XARRAY_EQUAL,
    XARRAY_EQUAL_TOLERANCE,
    XARRAY_NOT_EQUAL,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.unit.equality.comparators.utils import ExamplePair


@xarray_available
@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", XARRAY_EQUAL)
@pytest.mark.parametrize("show_difference", [True, False])
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
    show_difference: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert function(example.object1, example.object2, show_difference=show_difference)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", XARRAY_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.object1, example.object2)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", XARRAY_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.object1, example.object2, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_EQUAL_TOLERANCE)
def test_objects_are_close_true_tolerance(
    example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    assert objects_are_close(example.object1, example.object2, atol=example.atol, rtol=example.rtol)
