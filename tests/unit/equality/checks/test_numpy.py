from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola import objects_are_equal
from coola.testing import numpy_available
from coola.utils.imports import is_numpy_available
from tests.unit.equality.comparators.test_numpy import (
    NUMPY_ARRAY_EQUAL,
    NUMPY_ARRAY_NOT_EQUAL,
    NUMPY_MASKED_ARRAY_EQUAL,
    NUMPY_MASKED_ARRAY_NOT_EQUAL,
)

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


###################################
#     Tests for numpy.ndarray     #
###################################


@numpy_available
@pytest.mark.parametrize(("object1", "object2"), NUMPY_ARRAY_EQUAL + NUMPY_MASKED_ARRAY_EQUAL)
def test_objects_are_equal_true(
    object1: np.ndarray, object2: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert objects_are_equal(object1, object2)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize(("object1", "object2"), NUMPY_ARRAY_EQUAL + NUMPY_MASKED_ARRAY_EQUAL)
def test_objects_are_equal_true_show_difference(
    object1: np.ndarray, object2: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert objects_are_equal(object1, object2, show_difference=True)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2", "message"), NUMPY_ARRAY_NOT_EQUAL + NUMPY_MASKED_ARRAY_NOT_EQUAL
)
def test_objects_are_equal_false(
    object1: np.ndarray, object2: np.ndarray, message: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(object1, object2)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2", "message"), NUMPY_ARRAY_NOT_EQUAL + NUMPY_MASKED_ARRAY_NOT_EQUAL
)
def test_objects_are_equal_false_show_difference(
    object1: np.ndarray, object2: np.ndarray, message: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not objects_are_equal(object1, object2, show_difference=True)
        assert caplog.messages[-1].startswith(message)
