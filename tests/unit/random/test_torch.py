from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.random import TorchRandomNumberGenerator
from coola.testing import torch_available
from coola.utils import is_torch_available

if is_torch_available():
    import torch

###########################################
#     Tests for TorchRandomSeedSetter     #
###########################################


@torch_available
def test_torch_random_number_generator_repr() -> None:
    assert repr(TorchRandomNumberGenerator()).startswith("TorchRandomNumberGenerator(")


@torch_available
def test_torch_random_number_generator_str() -> None:
    assert str(TorchRandomNumberGenerator()).startswith("TorchRandomNumberGenerator(")


@torch_available
def test_torch_random_number_generator_get_rng_state() -> None:
    rng = TorchRandomNumberGenerator()
    state = rng.get_rng_state()
    assert isinstance(state, dict)
    assert len(state) == 2
    assert torch.is_tensor(state["torch"])
    assert isinstance(state["torch.cuda"], list)


@torch_available
def test_torch_random_number_generator_manual_seed() -> None:
    rng = TorchRandomNumberGenerator()
    rng.manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    rng.manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


@torch_available
@patch("torch.cuda.is_available", lambda: True)
def test_torch_random_number_generator_manual_seed_with_cuda() -> None:
    rng = TorchRandomNumberGenerator()
    with patch("torch.cuda.manual_seed_all") as mock_manual_seed_all:
        rng.manual_seed(42)
        mock_manual_seed_all.assert_called_with(42)


@torch_available
def test_torch_random_number_generator_set_rng_state() -> None:
    rng = TorchRandomNumberGenerator()
    state = rng.get_rng_state()
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    rng.set_rng_state(state)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


def test_torch_random_number_generator_no_torch() -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="`torch` package is required but not installed."),
    ):
        TorchRandomNumberGenerator()
