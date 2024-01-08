from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from coola import objects_are_equal
from coola.testing import pandas_available
from tests.unit.equality.comparators.test_pandas import (
    PANDAS_DATAFRAME_EQUAL,
    PANDAS_DATAFRAME_NOT_EQUAL,
    PANDAS_SERIES_EQUAL,
    PANDAS_SERIES_NOT_EQUAL,
)

if TYPE_CHECKING:
    from tests.unit.equality.comparators.utils import ExamplePair


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL + PANDAS_DATAFRAME_EQUAL)
def test_objects_are_equal_true(example: ExamplePair, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert objects_are_equal(example.object1, example.object2)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL + PANDAS_DATAFRAME_EQUAL)
def test_objects_are_equal_true_show_difference(
    example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert objects_are_equal(example.object1, example.object2, show_difference=True)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL + PANDAS_DATAFRAME_NOT_EQUAL)
def test_objects_are_equal_false(example: ExamplePair, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(example.object1, example.object2)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL + PANDAS_DATAFRAME_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(example.object1, example.object2, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
