from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, NumpyArrayEqualHandler
from coola.equality.testers import EqualityTester
from coola.testing import numpy_available
from coola.utils import is_numpy_available
from tests.unit.equality.comparators.test_numpy import NUMPY_ARRAY_EQUAL_TOLERANCE

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if TYPE_CHECKING:
    from tests.unit.equality.comparators.utils import ExamplePair


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


############################################
#     Tests for NumpyArrayEqualHandler     #
############################################


def test_numpy_array_equal_handler_eq_true() -> None:
    assert NumpyArrayEqualHandler() == NumpyArrayEqualHandler()


def test_numpy_array_equal_handler_eq_false() -> None:
    assert NumpyArrayEqualHandler() != FalseHandler()


def test_numpy_array_equal_handler_repr() -> None:
    assert repr(NumpyArrayEqualHandler()).startswith("NumpyArrayEqualHandler(")


def test_numpy_array_equal_handler_str() -> None:
    assert str(NumpyArrayEqualHandler()).startswith("NumpyArrayEqualHandler(")


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.ones(shape=(2, 3), dtype=float)),
        (np.ones(shape=(2, 3), dtype=int), np.ones(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3, 4), dtype=bool), np.ones(shape=(2, 3, 4), dtype=bool)),
    ],
)
def test_numpy_array_equal_handler_handle_true(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert NumpyArrayEqualHandler().handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3)), np.ones(shape=(3, 2))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 1))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3, 1))),
    ],
)
def test_numpy_array_equal_handler_handle_false(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert not NumpyArrayEqualHandler().handle(object1, object2, config)


@numpy_available
def test_numpy_array_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not NumpyArrayEqualHandler().handle(
        np.array([0.0, np.nan, np.nan, 1.2]), np.array([0.0, np.nan, np.nan, 1.2]), config
    )


@numpy_available
def test_numpy_array_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert NumpyArrayEqualHandler().handle(
        np.array([0.0, np.nan, np.nan, 1.2]), np.array([0.0, np.nan, np.nan, 1.2]), config
    )


@numpy_available
def test_numpy_array_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = NumpyArrayEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(3, 2)), config=config
        )
        assert caplog.messages[0].startswith("numpy.ndarrays have different elements:")


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL_TOLERANCE)
def test_numpy_array_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert NumpyArrayEqualHandler().handle(
        object1=example.object1, object2=example.object2, config=config
    )


def test_numpy_array_equal_handler_set_next_handler() -> None:
    NumpyArrayEqualHandler().set_next_handler(FalseHandler())
