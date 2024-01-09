from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, SameShapeHandler, TrueHandler
from coola.equality.testers import EqualityTester
from coola.testing import (
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
)
from coola.utils import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
)
from coola.utils.imports import is_jax_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_jax_available():
    import jax.numpy as jnp
if is_pandas_available():
    import pandas
if is_polars_available():
    import polars
if is_torch_available():
    import torch


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


######################################
#     Tests for SameShapeHandler     #
######################################


def test_same_shape_handler_eq_true() -> None:
    assert SameShapeHandler() == SameShapeHandler()


def test_same_shape_handler_eq_false() -> None:
    assert SameShapeHandler() != FalseHandler()


def test_same_shape_handler_repr() -> None:
    assert repr(SameShapeHandler()).startswith("SameShapeHandler(")


def test_same_shape_handler_str() -> None:
    assert str(SameShapeHandler()).startswith("SameShapeHandler(")


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.zeros(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3), dtype=int), np.zeros(shape=(2, 3), dtype=bool)),
        (np.ones(shape=(2, 3), dtype=bool), np.zeros(shape=(2, 3), dtype=float)),
    ],
)
def test_same_shape_handler_handle_true(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert SameShapeHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3)), np.ones(shape=(3, 2))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 1))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3, 1))),
    ],
)
def test_same_shape_handler_handle_false(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert not SameShapeHandler().handle(object1, object2, config)


@numpy_available
def test_same_shape_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameShapeHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(3, 2)), config=config
        )
        assert caplog.messages[0].startswith("objects have different shapes:")


@numpy_available
def test_same_shape_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameShapeHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(2, 3)), config=config)


def test_same_shape_handler_set_next_handler() -> None:
    handler = SameShapeHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_shape_handler_set_next_handler_incorrect() -> None:
    handler = SameShapeHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


@jax_available
def test_same_shape_handler_handle_jax(config: EqualityConfig) -> None:
    assert SameShapeHandler(next_handler=TrueHandler()).handle(
        jnp.ones((2, 3)), jnp.zeros((2, 3)), config
    )


@pandas_available
def test_same_shape_handler_handle_pandas(config: EqualityConfig) -> None:
    assert SameShapeHandler(next_handler=TrueHandler()).handle(
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        config,
    )


@polars_available
def test_same_shape_handler_handle_polars(config: EqualityConfig) -> None:
    assert SameShapeHandler(next_handler=TrueHandler()).handle(
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        config,
    )


@torch_available
def test_same_shape_handler_handle_torch(config: EqualityConfig) -> None:
    assert SameShapeHandler(next_handler=TrueHandler()).handle(
        torch.ones(2, 3), torch.zeros(2, 3), config
    )
