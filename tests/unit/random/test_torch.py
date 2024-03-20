from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.random import TorchRandomSeedSetter
from coola.testing import torch_available
from coola.utils import is_torch_available

if is_torch_available():
    import torch

###########################################
#     Tests for TorchRandomSeedSetter     #
###########################################


@torch_available
def test_torch_random_seed_setter_repr() -> None:
    assert repr(TorchRandomSeedSetter()).startswith("TorchRandomSeedSetter(")


@torch_available
def test_torch_random_seed_setter_str() -> None:
    assert str(TorchRandomSeedSetter()).startswith("TorchRandomSeedSetter(")


@torch_available
def test_torch_random_seed_setter_manual_seed() -> None:
    seed_setter = TorchRandomSeedSetter()
    seed_setter.manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    seed_setter.manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


@torch_available
@patch("torch.cuda.is_available", lambda: True)
def test_torch_random_seed_setter_manual_seed_with_cuda() -> None:
    seed_setter = TorchRandomSeedSetter()
    with patch("torch.cuda.manual_seed_all") as mock_manual_seed_all:
        seed_setter.manual_seed(42)
        mock_manual_seed_all.assert_called_with(42)


def test_torch_random_seed_setter_no_torch() -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="`torch` package is required but not installed."),
    ):
        TorchRandomSeedSetter()
