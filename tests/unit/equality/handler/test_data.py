from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, SameDataHandler, TrueHandler
from coola.testing.fixtures import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#####################################
#     Tests for SameDataHandler     #
#####################################


def test_same_data_handler_repr() -> None:
    assert repr(SameDataHandler()) == "SameDataHandler()"


def test_same_data_handler_str() -> None:
    assert str(SameDataHandler()) == "SameDataHandler()"


def test_same_data_handler_equal_true() -> None:
    assert SameDataHandler().equal(SameDataHandler())


def test_same_data_handler_equal_true_with_next_handler() -> None:
    assert SameDataHandler(FalseHandler()).equal(SameDataHandler(FalseHandler()))


def test_same_data_handler_equal_false_different_next_handler() -> None:
    assert not SameDataHandler(TrueHandler()).equal(SameDataHandler(FalseHandler()))


def test_same_data_handler_equal_false_different_next_handler_none() -> None:
    assert not SameDataHandler().equal(SameDataHandler(FalseHandler()))


def test_same_data_handler_equal_false_different_type() -> None:
    assert not SameDataHandler().equal(FalseHandler())


def test_same_data_handler_equal_false_different_type_child() -> None:
    class Child(SameDataHandler): ...

    assert not SameDataHandler().equal(Child())


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3))),
        (np.zeros(shape=(2, 3)), np.zeros(shape=(2, 3))),
        (np.ones(shape=(2, 3), dtype=int), np.ones(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3), dtype=int)),
    ],
)
def test_same_data_handler_handle_true(
    actual: np.ndarray, expected: np.ndarray, config: EqualityConfig
) -> None:
    assert SameDataHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (np.ones(shape=(2, 3)), np.zeros(shape=(2, 3))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3, 1))),
        (np.ones(shape=(2, 3)), np.ones(shape=(3, 2))),
    ],
)
def test_same_data_handler_handle_false(
    actual: np.ndarray, expected: np.ndarray, config: EqualityConfig
) -> None:
    assert not SameDataHandler().handle(actual, expected, config)


@numpy_available
def test_same_data_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameDataHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=np.ones(shape=(2, 3)), expected=np.zeros(shape=(2, 3)), config=config
        )
        assert caplog.messages[-1].startswith("objects have different data:")


@numpy_available
def test_same_data_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameDataHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual=np.ones(shape=(2, 3)), expected=np.ones(shape=(2, 3)), config=config)


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


def test_same_data_handler_set_next_handler() -> None:
    handler = SameDataHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_same_data_handler_set_next_handler_none() -> None:
    handler = SameDataHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_same_data_handler_set_next_handler_incorrect() -> None:
    handler = SameDataHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)
