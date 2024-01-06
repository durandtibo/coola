from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler
from coola.equality.handlers.torch_ import TensorEqualHandler
from coola.testing import torch_available
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
