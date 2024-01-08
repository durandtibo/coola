from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, SameDataHandler, TrueHandler
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#####################################
#     Tests for SameDataHandler     #
#####################################


def test_same_data_handler_eq_true() -> None:
    assert SameDataHandler() == SameDataHandler()


def test_same_data_handler_eq_false() -> None:
    assert SameDataHandler() != FalseHandler()


def test_same_data_handler_repr() -> None:
    assert repr(SameDataHandler()).startswith("SameDataHandler(")


def test_same_data_handler_str() -> None:
    assert str(SameDataHandler()).startswith("SameDataHandler(")


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3))),
        (np.zeros(shape=(2, 3)), np.zeros(shape=(2, 3))),
        (np.ones(shape=(2, 3), dtype=int), np.ones(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3), dtype=int)),
    ],
)
def test_same_data_handler_handle_true(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert SameDataHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3)), np.zeros(shape=(2, 3))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3, 1))),
        (np.ones(shape=(2, 3)), np.ones(shape=(3, 2))),
    ],
)
def test_same_data_handler_handle_false(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert not SameDataHandler().handle(object1, object2, config)


@numpy_available
def test_same_data_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameDataHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=np.ones(shape=(2, 3)), object2=np.zeros(shape=(2, 3)), config=config
        )
        assert caplog.messages[-1].startswith("objects have different data:")


@numpy_available
def test_same_data_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameDataHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(2, 3)), config=config)


def test_same_data_handler_set_next_handler() -> None:
    handler = SameDataHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_data_handler_set_next_handler_incorrect() -> None:
    handler = SameDataHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


@torch_available
def test_same_data_handler_handle_tensor(config: EqualityConfig) -> None:
    handler = SameDataHandler(next_handler=TrueHandler())
    assert handler.handle(torch.ones(2, 3), torch.ones(2, 3), config)
    assert not handler.handle(torch.ones(2, 3), torch.zeros(2, 3), config)


@torch_available
def test_same_data_handler_handle_padded_sequence(config: EqualityConfig) -> None:
    handler = SameDataHandler(next_handler=TrueHandler())
    assert handler.handle(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        config,
    )
