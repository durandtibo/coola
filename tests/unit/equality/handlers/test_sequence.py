from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, SequenceSameValueHandler, TrueHandler
from coola.testing import numpy_available
from coola.utils import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##############################################
#     Tests for SequenceSameValueHandler     #
##############################################


def test_sequence_same_value_handler_eq_true() -> None:
    assert SequenceSameValueHandler() == SequenceSameValueHandler()


def test_sequence_same_value_handler_eq_false() -> None:
    assert SequenceSameValueHandler() != FalseHandler()


def test_sequence_same_value_handler_str() -> None:
    assert str(SequenceSameValueHandler()).startswith("SequenceSameValueHandler(")


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([0, 1, 2], [0, 1, 2]),
        ((0, 1, 2), (0, 1, 2)),
        ([0, ("a", "b", "c"), 2], [0, ("a", "b", "c"), 2]),
        ([0, 1, 2], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [0, 1, 2]),
    ],
)
def test_sequence_same_value_handler_handle_true(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert SequenceSameValueHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.zeros(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.zeros(2))),
    ],
)
def test_sequence_same_value_handler_handle_true_numpy(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert SequenceSameValueHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([1, 2, 3], [1, 2, 4]),
        ((1, 2, 3), (1, 2, 4)),
        ([0, ("a", "b", "c"), 2], [0, ("a", "b", "d"), 2]),
    ],
)
def test_sequence_same_value_handler_handle_false(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceSameValueHandler().handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.ones(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.ones(2))),
    ],
)
def test_sequence_same_value_handler_handle_false_numpy(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceSameValueHandler(next_handler=TrueHandler()).handle(object1, object2, config)


def test_sequence_same_value_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SequenceSameValueHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1=[1, 2, 3], object2=[1, 2, 4], config=config)
        assert caplog.messages[-1].startswith("sequences have at least one different value:")


def test_sequence_same_value_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SequenceSameValueHandler()
    with pytest.raises(RuntimeError, match="The next handler is not defined"):
        handler.handle(object1=[1, 2, 3], object2=[1, 2, 3], config=config)


def test_sequence_same_value_handler_set_next_handler() -> None:
    handler = SequenceSameValueHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_sequence_same_value_handler_set_next_handler_incorrect() -> None:
    handler = SequenceSameValueHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)
