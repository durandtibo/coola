from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, SequenceSameValuesHandler, TrueHandler
from coola.equality.testers import EqualityTester
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


###############################################
#     Tests for SequenceSameValuesHandler     #
###############################################


def test_sequence_same_values_handler_eq_true() -> None:
    assert SequenceSameValuesHandler() == SequenceSameValuesHandler()


def test_sequence_same_values_handler_eq_false() -> None:
    assert SequenceSameValuesHandler() != FalseHandler()


def test_sequence_same_values_handler_repr() -> None:
    assert repr(SequenceSameValuesHandler()).startswith("SequenceSameValuesHandler(")


def test_sequence_same_values_handler_str() -> None:
    assert str(SequenceSameValuesHandler()).startswith("SequenceSameValuesHandler(")


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
def test_sequence_same_values_handler_handle_true(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert SequenceSameValuesHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.zeros(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.zeros(2))),
    ],
)
def test_sequence_same_values_handler_handle_true_numpy(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert SequenceSameValuesHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([1, 2, 3], [1, 2, 4]),
        ((1, 2, 3), (1, 2, 4)),
        ([0, ("a", "b", "c"), 2], [0, ("a", "b", "d"), 2]),
    ],
)
def test_sequence_same_values_handler_handle_false(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceSameValuesHandler().handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([np.ones((2, 3)), np.zeros(2)], [np.ones((2, 3)), np.ones(2)]),
        ((np.ones((2, 3)), np.zeros(2)), (np.ones((2, 3)), np.ones(2))),
    ],
)
def test_sequence_same_values_handler_handle_false_numpy(
    object1: Sequence, object2: Sequence, config: EqualityConfig
) -> None:
    assert not SequenceSameValuesHandler(next_handler=TrueHandler()).handle(
        object1, object2, config
    )


def test_sequence_same_values_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SequenceSameValuesHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1=[1, 2, 3], object2=[1, 2, 4], config=config)
        assert caplog.messages[-1].startswith("sequences have at least one different value:")


def test_sequence_same_values_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SequenceSameValuesHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1=[1, 2, 3], object2=[1, 2, 3], config=config)


def test_sequence_same_values_handler_set_next_handler() -> None:
    handler = SequenceSameValuesHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_sequence_same_values_handler_set_next_handler_incorrect() -> None:
    handler = SequenceSameValuesHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)
