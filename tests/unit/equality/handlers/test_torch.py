from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, TrueHandler
from coola.equality.handlers.torch_ import TensorEqualHandler, TensorSameDeviceHandler
from coola.testing import torch_available, torch_cuda_available
from coola.utils import is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


########################################
#     Tests for TensorEqualHandler     #
########################################


def test_tensor_equal_handler_eq_true() -> None:
    assert TensorEqualHandler() == TensorEqualHandler()


def test_tensor_equal_handler_eq_false() -> None:
    assert TensorEqualHandler() != FalseHandler()


def test_tensor_equal_handler_str() -> None:
    assert str(TensorEqualHandler()).startswith("TensorEqualHandler(")


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (torch.ones(2, 3, dtype=torch.float), torch.ones(2, 3, dtype=torch.float)),
        (torch.ones(2, 3, dtype=torch.int), torch.ones(2, 3, dtype=torch.int)),
        (torch.ones(2, 3, 4, dtype=torch.bool), torch.ones(2, 3, 4, dtype=torch.bool)),
    ],
)
def test_tensor_equal_handler_handle_true(
    object1: torch.Tensor, object2: torch.Tensor, config: EqualityConfig
) -> None:
    assert TensorEqualHandler().handle(object1, object2, config)


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (torch.ones(2, 3), torch.ones(3, 2)),
        (torch.ones(2, 3), torch.ones(2, 1)),
        (torch.ones(2, 3), torch.ones(2, 3, 1)),
    ],
)
def test_tensor_equal_handler_handle_false(
    object1: torch.Tensor, object2: torch.Tensor, config: EqualityConfig
) -> None:
    assert not TensorEqualHandler().handle(object1, object2, config)


@torch_available
def test_tensor_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not TensorEqualHandler().handle(
        torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@torch_available
def test_tensor_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert TensorEqualHandler().handle(
        torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@torch_available
def test_tensor_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = TensorEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1=torch.ones(2, 3), object2=torch.ones(3, 2), config=config)
        assert caplog.messages[0].startswith("torch.Tensors have different elements:")


def test_tensor_equal_handler_set_next_handler() -> None:
    TensorEqualHandler().set_next_handler(FalseHandler())


#############################################
#     Tests for TensorSameDeviceHandler     #
#############################################


def test_tensor_same_device_handler_eq_true() -> None:
    assert TensorSameDeviceHandler() == TensorSameDeviceHandler()


def test_tensor_same_device_handler_eq_false() -> None:
    assert TensorSameDeviceHandler() != FalseHandler()


def test_tensor_same_device_handler_str() -> None:
    assert str(TensorSameDeviceHandler()).startswith("TensorSameDeviceHandler(")


@torch_available
def test_tensor_same_device_handler_handle_true(config: EqualityConfig) -> None:
    assert TensorSameDeviceHandler(next_handler=TrueHandler()).handle(
        torch.ones(2, 3), torch.zeros(2, 3), config
    )


@torch_available
@torch_cuda_available
def test_tensor_same_device_handler_handle_false(config: EqualityConfig) -> None:
    assert not TensorSameDeviceHandler().handle(
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("cuda:0")),
        config,
    )


@torch_available
def testtensor_same_device_handler_handle_false_mock(config: EqualityConfig) -> None:
    assert not TensorSameDeviceHandler().handle(
        Mock(spec=torch.Tensor, device=torch.device("cpu")),
        Mock(spec=torch.Tensor, device=torch.device("cuda:0")),
        config,
    )


@torch_available
def test_tensor_same_device_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = TensorSameDeviceHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=Mock(spec=torch.Tensor, device=torch.device("cpu")),
            object2=Mock(spec=torch.Tensor, device=torch.device("cuda:0")),
            config=config,
        )
        assert caplog.messages[0].startswith("torch.Tensors have different devices:")


@torch_available
def test_tensor_same_device_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = TensorSameDeviceHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1=torch.ones(2, 3), object2=torch.ones(2, 3), config=config)


def test_tensor_same_device_handler_set_next_handler() -> None:
    handler = TensorSameDeviceHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_tensor_same_device_handler_set_next_handler_incorrect() -> None:
    handler = TensorSameDeviceHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)
