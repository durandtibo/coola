from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import (
    ArraySameDTypeHandler,
    ArraySameShapeHandler,
    FalseHandler,
    TrueHandler,
)
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


###########################################
#     Tests for ArraySameShapeHandler     #
###########################################


def test_array_same_shape_handler_eq_true() -> None:
    assert ArraySameShapeHandler() == ArraySameShapeHandler()


def test_array_same_shape_handler_eq_false() -> None:
    assert ArraySameShapeHandler() != FalseHandler()


def test_array_same_shape_handler_str() -> None:
    assert str(ArraySameShapeHandler()).startswith("ArraySameShapeHandler(")


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.zeros(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3), dtype=int), np.zeros(shape=(2, 3), dtype=bool)),
        (np.ones(shape=(2, 3), dtype=bool), np.zeros(shape=(2, 3), dtype=float)),
    ],
)
def test_array_same_shape_handler_handle_true_ndarray(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert ArraySameShapeHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3)), np.ones(shape=(3, 2))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 1))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3, 1))),
    ],
)
def test_array_same_shape_handler_handle_false_ndarray(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert not ArraySameShapeHandler().handle(object1, object2, config)


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (torch.ones(2, 3, dtype=torch.float), torch.zeros(2, 3, dtype=torch.int)),
        (torch.ones(2, 3, dtype=torch.int), torch.zeros(2, 3, dtype=torch.bool)),
        (torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.float)),
    ],
)
def test_array_same_shape_handler_handle_true_tensor(
    object1: torch.Tensor, object2: torch.Tensor, config: EqualityConfig
) -> None:
    assert ArraySameShapeHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (torch.ones(2, 3), torch.ones(3, 2)),
        (torch.ones(2, 3), torch.ones(2, 1)),
        (torch.ones(2, 3), torch.ones(2, 3, 1)),
    ],
)
def test_array_same_shape_handler_handle_false_tensor(
    object1: torch.Tensor, object2: torch.Tensor, config: EqualityConfig
) -> None:
    assert not ArraySameShapeHandler().handle(object1, object2, config)


@numpy_available
def test_array_same_shape_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = ArraySameShapeHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(3, 2)), config=config
        )
        assert caplog.messages[0].startswith("objects have different shapes:")


@numpy_available
def test_array_same_shape_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = ArraySameShapeHandler()
    with pytest.raises(RuntimeError, match="The next handler is not defined"):
        handler.handle(object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(2, 3)), config=config)


def test_array_same_shape_handler_set_next_handler() -> None:
    handler = ArraySameShapeHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_array_same_shape_handler_set_next_handler_incorrect() -> None:
    handler = ArraySameShapeHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


###########################################
#     Tests for ArraySameDTypeHandler     #
###########################################


def test_array_same_dtype_handler_eq_true() -> None:
    assert ArraySameDTypeHandler() == ArraySameDTypeHandler()


def test_array_same_dtype_handler_eq_false() -> None:
    assert ArraySameDTypeHandler() != FalseHandler()


def test_array_same_dtype_handler_str() -> None:
    assert str(ArraySameDTypeHandler()).startswith("ArraySameDTypeHandler(")


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.zeros(shape=(2, 3), dtype=float)),
        (np.ones(shape=(2, 3), dtype=int), np.zeros(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3), dtype=bool), np.zeros(shape=(2, 3), dtype=bool)),
    ],
)
def test_array_same_dtype_handler_handle_true_ndarray(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert ArraySameDTypeHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@numpy_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.ones(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3), dtype=int), np.ones(shape=(2, 3), dtype=bool)),
        (np.ones(shape=(2, 3), dtype=bool), np.ones(shape=(2, 3), dtype=float)),
    ],
)
def test_array_same_dtype_handler_handle_false_ndarray(
    object1: np.ndarray, object2: np.ndarray, config: EqualityConfig
) -> None:
    assert not ArraySameDTypeHandler().handle(object1, object2, config)


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (torch.ones(2, 3, dtype=torch.float), torch.zeros(2, 3, dtype=torch.float)),
        (torch.ones(2, 3, dtype=torch.int), torch.zeros(2, 3, dtype=torch.int)),
        (torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.bool)),
    ],
)
def test_array_same_dtype_handler_handle_true_tensor(
    object1: torch.Tensor, object2: torch.Tensor, config: EqualityConfig
) -> None:
    assert ArraySameDTypeHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (torch.ones(2, 3, dtype=torch.float), torch.ones(2, 3, dtype=torch.int)),
        (torch.ones(2, 3, dtype=torch.int), torch.ones(2, 3, dtype=torch.bool)),
        (torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 3, dtype=torch.float)),
    ],
)
def test_array_same_dtype_handler_handle_false_tensor(
    object1: torch.Tensor, object2: torch.Tensor, config: EqualityConfig
) -> None:
    assert not ArraySameDTypeHandler().handle(object1, object2, config)


@numpy_available
def test_array_same_dtype_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = ArraySameDTypeHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=np.ones(shape=(2, 3), dtype=float),
            object2=np.ones(shape=(2, 3), dtype=int),
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different data types:")


@numpy_available
def test_array_same_dtype_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = ArraySameDTypeHandler()
    with pytest.raises(RuntimeError, match="The next handler is not defined"):
        handler.handle(object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(2, 3)), config=config)


def test_array_same_dtype_handler_set_next_handler() -> None:
    handler = ArraySameDTypeHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_array_same_dtype_handler_set_next_handler_incorrect() -> None:
    handler = ArraySameDTypeHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)
