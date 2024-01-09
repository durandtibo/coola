from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, SameDTypeHandler, TrueHandler
from coola.equality.testers import EqualityTester
from coola.testing import jax_available, numpy_available, torch_available
from coola.utils import is_jax_available, is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_jax_available():
    import jax.numpy as jnp
if is_torch_available():
    import torch


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


######################################
#     Tests for SameDTypeHandler     #
######################################


def test_same_dtype_handler_eq_true() -> None:
    assert SameDTypeHandler() == SameDTypeHandler()


def test_same_dtype_handler_eq_false() -> None:
    assert SameDTypeHandler() != FalseHandler()


def test_same_dtype_handler_repr() -> None:
    assert repr(SameDTypeHandler()).startswith("SameDTypeHandler(")


def test_same_dtype_handler_str() -> None:
    assert str(SameDTypeHandler()).startswith("SameDTypeHandler(")


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.zeros(shape=(2, 3), dtype=float)),
        (np.ones(shape=(2, 3), dtype=int), np.zeros(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3), dtype=bool), np.zeros(shape=(2, 3), dtype=bool)),
    ],
)
def test_same_dtype_handler_handle_true(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert SameDTypeHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.ones(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3), dtype=int), np.ones(shape=(2, 3), dtype=bool)),
        (np.ones(shape=(2, 3), dtype=bool), np.ones(shape=(2, 3), dtype=float)),
    ],
)
def test_same_dtype_handler_handle_false(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert not SameDTypeHandler().handle(object1, object2, config)


@numpy_available
def test_same_dtype_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameDTypeHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=np.ones(shape=(2, 3), dtype=float),
            object2=np.ones(shape=(2, 3), dtype=int),
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different data types:")


@numpy_available
def test_same_dtype_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameDTypeHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(2, 3)), config=config)


def test_same_dtype_handler_set_next_handler() -> None:
    handler = SameDTypeHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_dtype_handler_set_next_handler_incorrect() -> None:
    handler = SameDTypeHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


@jax_available
def test_same_dtype_handler_handle_jax(config: EqualityConfig) -> None:
    assert SameDTypeHandler(next_handler=TrueHandler()).handle(
        jnp.ones(shape=(2, 3), dtype=float), jnp.zeros(shape=(2, 3), dtype=float), config
    )


@torch_available
def test_same_dtype_handler_handle_tensor(config: EqualityConfig) -> None:
    assert SameDTypeHandler(next_handler=TrueHandler()).handle(
        torch.ones(2, 3, dtype=torch.float), torch.zeros(2, 3, dtype=torch.float), config
    )
