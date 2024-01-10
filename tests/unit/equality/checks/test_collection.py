from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola import objects_are_equal
from tests.unit.equality.comparators.test_collection import (
    COLLECTION_EQUAL,
    COLLECTION_NOT_EQUAL,
)

if TYPE_CHECKING:
    from tests.unit.equality.comparators.utils import ExamplePair


@pytest.mark.parametrize("example", COLLECTION_EQUAL)
@pytest.mark.parametrize("show_difference", [True, False])
def test_objects_are_equal_true(
    example: ExamplePair, show_difference: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert objects_are_equal(example.object1, example.object2, show_difference)
        assert not caplog.messages


@pytest.mark.parametrize("example", COLLECTION_NOT_EQUAL)
def test_objects_are_equal_false(example: ExamplePair, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(example.object1, example.object2)
        assert not caplog.messages


@pytest.mark.parametrize("example", COLLECTION_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(example.object1, example.object2, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
