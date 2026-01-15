from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, SameShapeHandler, TrueHandler
from coola.testing.fixtures import (
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
)
from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
)

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_jax_available():
    import jax.numpy as jnp
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_torch_available():
    import torch


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


######################################
#     Tests for SameShapeHandler     #
######################################


def test_same_shape_handler_repr() -> None:
    assert repr(SameShapeHandler()) == "SameShapeHandler()"


def test_same_shape_handler_repr_with_next_handler() -> None:
    assert repr(SameShapeHandler(FalseHandler())) == "SameShapeHandler(next_handler=FalseHandler())"


def test_same_shape_handler_str() -> None:
    assert str(SameShapeHandler()) == "SameShapeHandler()"


def test_same_shape_handler_str_with_next_handler() -> None:
    assert str(SameShapeHandler(FalseHandler())) == "SameShapeHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(SameShapeHandler(), SameShapeHandler(), id="without next handler"),
        pytest.param(
            SameShapeHandler(FalseHandler()),
            SameShapeHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_same_shape_handler_equal_true(
    handler1: SameShapeHandler, handler2: SameShapeHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            SameShapeHandler(TrueHandler()),
            SameShapeHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            SameShapeHandler(),
            SameShapeHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            SameShapeHandler(FalseHandler()),
            SameShapeHandler(),
            id="other next handler is none",
        ),
        pytest.param(SameShapeHandler(), FalseHandler(), id="different type"),
    ],
)
def test_same_shape_handler_equal_false(handler1: SameShapeHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_same_shape_handler_equal_false_different_type_child() -> None:
    class Child(SameShapeHandler): ...

    assert not SameShapeHandler().equal(Child())


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.zeros(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3), dtype=int), np.zeros(shape=(2, 3), dtype=bool)),
        (np.ones(shape=(2, 3), dtype=bool), np.zeros(shape=(2, 3), dtype=float)),
    ],
)
def test_same_shape_handler_handle_true(
    actual: np.ndarray, expected: np.ndarray, config: EqualityConfig
) -> None:
    assert SameShapeHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (np.ones(shape=(2, 3)), np.ones(shape=(3, 2))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 1))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3, 1))),
    ],
)
def test_same_shape_handler_handle_false(
    actual: np.ndarray, expected: np.ndarray, config: EqualityConfig
) -> None:
    assert not SameShapeHandler().handle(actual, expected, config)


@numpy_available
def test_same_shape_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameShapeHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=np.ones(shape=(2, 3)), expected=np.ones(shape=(3, 2)), config=config
        )
        assert caplog.messages[0].startswith("objects have different shapes:")


@numpy_available
def test_same_shape_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameShapeHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual=np.ones(shape=(2, 3)), expected=np.ones(shape=(2, 3)), config=config)


def test_same_shape_handler_set_next_handler() -> None:
    handler = SameShapeHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_same_shape_handler_set_next_handler_none() -> None:
    handler = SameShapeHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_same_shape_handler_set_next_handler_incorrect() -> None:
    handler = SameShapeHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


@jax_available
def test_same_shape_handler_handle_jax(config: EqualityConfig) -> None:
    assert SameShapeHandler(next_handler=TrueHandler()).handle(
        jnp.ones((2, 3)), jnp.zeros((2, 3)), config
    )


@pandas_available
def test_same_shape_handler_handle_pandas(config: EqualityConfig) -> None:
    assert SameShapeHandler(next_handler=TrueHandler()).handle(
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        pd.DataFrame(
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
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        pl.DataFrame(
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
