from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, TrueHandler
from coola.equality.handler.torch import (
    TorchTensorEqualHandler,
    TorchTensorSameDeviceHandler,
)
from coola.testing.fixtures import torch_available, torch_cuda_available
from coola.utils.imports import is_torch_available
from tests.unit.equality.tester.test_torch import TORCH_TENSOR_EQUAL_TOLERANCE

if is_torch_available():
    import torch
else:
    torch = Mock()

if TYPE_CHECKING:
    from tests.unit.equality.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#############################################
#     Tests for TorchTensorEqualHandler     #
#############################################


def test_torch_tensor_equal_handler_repr() -> None:
    assert repr(TorchTensorEqualHandler()) == "TorchTensorEqualHandler()"


def test_torch_tensor_equal_handler_str() -> None:
    assert str(TorchTensorEqualHandler()) == "TorchTensorEqualHandler()"


def test_torch_tensor_equal_handler_equal_true() -> None:
    assert TorchTensorEqualHandler().equal(TorchTensorEqualHandler())


def test_torch_tensor_equal_handler_equal_false_different_type() -> None:
    assert not TorchTensorEqualHandler().equal(FalseHandler())


def test_torch_tensor_equal_handler_equal_false_different_type_child() -> None:
    class Child(TorchTensorEqualHandler): ...

    assert not TorchTensorEqualHandler().equal(Child())


@torch_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (torch.ones(2, 3, dtype=torch.float), torch.ones(2, 3, dtype=torch.float)),
        (torch.ones(2, 3, dtype=torch.int), torch.ones(2, 3, dtype=torch.int)),
        (torch.ones(2, 3, 4, dtype=torch.bool), torch.ones(2, 3, 4, dtype=torch.bool)),
    ],
)
def test_torch_tensor_equal_handler_handle_true(
    actual: torch.Tensor, expected: torch.Tensor, config: EqualityConfig
) -> None:
    assert TorchTensorEqualHandler().handle(actual, expected, config)


@torch_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (torch.ones(2, 3), torch.ones(3, 2)),
        (torch.ones(2, 3), torch.ones(2, 1)),
        (torch.ones(2, 3), torch.ones(2, 3, 1)),
    ],
)
def test_torch_tensor_equal_handler_handle_false(
    actual: torch.Tensor, expected: torch.Tensor, config: EqualityConfig
) -> None:
    assert not TorchTensorEqualHandler().handle(actual, expected, config)


@torch_available
def test_torch_tensor_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not TorchTensorEqualHandler().handle(
        torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@torch_available
def test_torch_tensor_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert TorchTensorEqualHandler().handle(
        torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@torch_available
def test_torch_tensor_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = TorchTensorEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual=torch.ones(2, 3), expected=torch.ones(3, 2), config=config)
        assert caplog.messages[0].startswith("torch.Tensors have different elements:")


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL_TOLERANCE)
def test_torch_tensor_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert TorchTensorEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_torch_tensor_equal_handler_set_next_handler() -> None:
    TorchTensorEqualHandler().set_next_handler(FalseHandler())


##################################################
#     Tests for TorchTensorSameDeviceHandler     #
##################################################


def test_torch_tensor_same_device_handler_repr() -> None:
    assert repr(TorchTensorSameDeviceHandler()) == "TorchTensorSameDeviceHandler()"


def test_torch_tensor_same_device_handler_str() -> None:
    assert str(TorchTensorSameDeviceHandler()) == "TorchTensorSameDeviceHandler()"


def test_torch_tensor_same_device_handler_equal_true() -> None:
    assert TorchTensorSameDeviceHandler().equal(TorchTensorSameDeviceHandler())


def test_torch_tensor_same_device_handler_equal_false_different_type() -> None:
    assert not TorchTensorSameDeviceHandler().equal(FalseHandler())


def test_torch_tensor_same_device_handler_equal_false_different_type_child() -> None:
    class Child(TorchTensorSameDeviceHandler): ...

    assert not TorchTensorSameDeviceHandler().equal(Child())


@torch_available
def test_torch_tensor_same_device_handler_handle_true(config: EqualityConfig) -> None:
    assert TorchTensorSameDeviceHandler(next_handler=TrueHandler()).handle(
        torch.ones(2, 3), torch.zeros(2, 3), config
    )


@torch_available
@torch_cuda_available
def test_torch_tensor_same_device_handler_handle_false(config: EqualityConfig) -> None:
    assert not TorchTensorSameDeviceHandler().handle(
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("cuda:0")),
        config,
    )


@torch_available
def test_torch_tensor_same_device_handler_handle_false_mock(config: EqualityConfig) -> None:
    assert not TorchTensorSameDeviceHandler().handle(
        Mock(spec=torch.Tensor, device=torch.device("cpu")),
        Mock(spec=torch.Tensor, device=torch.device("cuda:0")),
        config,
    )


@torch_available
def test_torch_tensor_same_device_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = TorchTensorSameDeviceHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=Mock(spec=torch.Tensor, device=torch.device("cpu")),
            expected=Mock(spec=torch.Tensor, device=torch.device("cuda:0")),
            config=config,
        )
        assert caplog.messages[0].startswith("torch.Tensors have different devices:")


@torch_available
def test_torch_tensor_same_device_handler_handle_without_next_handler(
    config: EqualityConfig,
) -> None:
    handler = TorchTensorSameDeviceHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual=torch.ones(2, 3), expected=torch.ones(2, 3), config=config)


def test_torch_tensor_same_device_handler_set_next_handler() -> None:
    handler = TorchTensorSameDeviceHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_torch_tensor_same_device_handler_set_next_handler_none() -> None:
    handler = TorchTensorSameDeviceHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_torch_tensor_same_device_handler_set_next_handler_incorrect() -> None:
    handler = TorchTensorSameDeviceHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)
