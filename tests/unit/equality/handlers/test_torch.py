from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, TrueHandler
from coola.equality.handlers.torch_ import (
    TorchTensorEqualHandler,
    TorchTensorSameDeviceHandler,
)
from coola.equality.testers import EqualityTester
from coola.testing import torch_available, torch_cuda_available
from coola.utils import is_torch_available
from tests.unit.equality.comparators.utils import ExamplePair

if is_torch_available():
    import torch
else:
    torch = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


TORCH_TENSOR_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(object1=torch.ones(2, 3), object2=torch.full((2, 3), 1.5), atol=1.0),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(object1=torch.ones(2, 3), object2=torch.full((2, 3), 1.05), atol=0.1),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(object1=torch.ones(2, 3), object2=torch.full((2, 3), 1.005), atol=0.01),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(object1=torch.ones(2, 3), object2=torch.full((2, 3), 1.5), rtol=1.0),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(object1=torch.ones(2, 3), object2=torch.full((2, 3), 1.05), rtol=0.1),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(object1=torch.ones(2, 3), object2=torch.full((2, 3), 1.005), rtol=0.01),
        id="rtol=0.01",
    ),
]


#############################################
#     Tests for TorchTensorEqualHandler     #
#############################################


def test_torch_tensor_equal_handler_eq_true() -> None:
    assert TorchTensorEqualHandler() == TorchTensorEqualHandler()


def test_torch_tensor_equal_handler_eq_false() -> None:
    assert TorchTensorEqualHandler() != FalseHandler()


def test_torch_tensor_equal_handler_repr() -> None:
    assert repr(TorchTensorEqualHandler()).startswith("TorchTensorEqualHandler(")


def test_torch_tensor_equal_handler_str() -> None:
    assert str(TorchTensorEqualHandler()).startswith("TorchTensorEqualHandler(")


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (torch.ones(2, 3, dtype=torch.float), torch.ones(2, 3, dtype=torch.float)),
        (torch.ones(2, 3, dtype=torch.int), torch.ones(2, 3, dtype=torch.int)),
        (torch.ones(2, 3, 4, dtype=torch.bool), torch.ones(2, 3, 4, dtype=torch.bool)),
    ],
)
def test_torch_tensor_equal_handler_handle_true(
    object1: torch.Tensor, object2: torch.Tensor, config: EqualityConfig
) -> None:
    assert TorchTensorEqualHandler().handle(object1, object2, config)


@torch_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (torch.ones(2, 3), torch.ones(3, 2)),
        (torch.ones(2, 3), torch.ones(2, 1)),
        (torch.ones(2, 3), torch.ones(2, 3, 1)),
    ],
)
def test_torch_tensor_equal_handler_handle_false(
    object1: torch.Tensor, object2: torch.Tensor, config: EqualityConfig
) -> None:
    assert not TorchTensorEqualHandler().handle(object1, object2, config)


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
        assert not handler.handle(object1=torch.ones(2, 3), object2=torch.ones(3, 2), config=config)
        assert caplog.messages[0].startswith("torch.Tensors have different elements:")


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL_TOLERANCE)
def test_torch_tensor_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert TorchTensorEqualHandler().handle(
        object1=example.object1, object2=example.object2, config=config
    )


def test_torch_tensor_equal_handler_set_next_handler() -> None:
    TorchTensorEqualHandler().set_next_handler(FalseHandler())


##################################################
#     Tests for TorchTensorSameDeviceHandler     #
##################################################


def test_torch_tensor_same_device_handler_eq_true() -> None:
    assert TorchTensorSameDeviceHandler() == TorchTensorSameDeviceHandler()


def test_torch_tensor_same_device_handler_eq_false() -> None:
    assert TorchTensorSameDeviceHandler() != FalseHandler()


def test_torch_tensor_same_device_handler_repr() -> None:
    assert repr(TorchTensorSameDeviceHandler()).startswith("TorchTensorSameDeviceHandler(")


def test_torch_tensor_same_device_handler_str() -> None:
    assert str(TorchTensorSameDeviceHandler()).startswith("TorchTensorSameDeviceHandler(")


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
            object1=Mock(spec=torch.Tensor, device=torch.device("cpu")),
            object2=Mock(spec=torch.Tensor, device=torch.device("cuda:0")),
            config=config,
        )
        assert caplog.messages[0].startswith("torch.Tensors have different devices:")


@torch_available
def test_torch_tensor_same_device_handler_handle_without_next_handler(
    config: EqualityConfig,
) -> None:
    handler = TorchTensorSameDeviceHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1=torch.ones(2, 3), object2=torch.ones(2, 3), config=config)


def test_torch_tensor_same_device_handler_set_next_handler() -> None:
    handler = TorchTensorSameDeviceHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_torch_tensor_same_device_handler_set_next_handler_incorrect() -> None:
    handler = TorchTensorSameDeviceHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)
