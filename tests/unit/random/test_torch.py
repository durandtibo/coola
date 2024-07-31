from __future__ import annotations

from unittest.mock import patch

import pytest

from coola import objects_are_equal
from coola.random import TorchRandomManager
from coola.random.torch_ import get_random_managers, torch_seed
from coola.testing import torch_available
from coola.utils import is_torch_available

if is_torch_available():
    import torch

########################################
#     Tests for TorchRandomManager     #
########################################


@torch_available
def test_torch_random_manager_repr() -> None:
    assert repr(TorchRandomManager()).startswith("TorchRandomManager(")


@torch_available
def test_torch_random_manager_str() -> None:
    assert str(TorchRandomManager()).startswith("TorchRandomManager(")


@torch_available
def test_torch_random_manager_eq_true() -> None:
    assert TorchRandomManager() == TorchRandomManager()


@torch_available
def test_torch_random_manager_eq_false() -> None:
    assert TorchRandomManager() != 42


@torch_available
def test_torch_random_manager_get_rng_state() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, dict)
    assert len(state) == 2
    assert torch.is_tensor(state["torch"])
    assert isinstance(state["torch.cuda"], list)


@torch_available
def test_torch_random_manager_manual_seed() -> None:
    rng = TorchRandomManager()
    rng.manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    rng.manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


@torch_available
@patch("torch.cuda.is_available", lambda: True)
def test_torch_random_manager_manual_seed_with_cuda() -> None:
    rng = TorchRandomManager()
    with patch("torch.cuda.manual_seed_all") as mock_manual_seed_all:
        rng.manual_seed(42)
        mock_manual_seed_all.assert_called_with(42)


@torch_available
def test_torch_random_manager_set_rng_state() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    rng.set_rng_state(state)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


def test_torch_random_manager_no_torch() -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match="'torch' package is required but not installed."),
    ):
        TorchRandomManager()


#########################################
#     Tests for get_random_managers     #
#########################################


@torch_available
def test_get_random_managers() -> None:
    assert objects_are_equal(get_random_managers(), {"torch": TorchRandomManager()})


def test_get_random_managers_no_torch() -> None:
    with patch("coola.random.torch_.is_torch_available", lambda: False):
        assert objects_are_equal(get_random_managers(), {})


################################
#     Tests for torch_seed     #
################################


@torch_available
def test_torch_seed_restore_random_seed() -> None:
    state = torch.get_rng_state()
    with torch_seed(42):
        torch.randn(4, 6)
    assert objects_are_equal(state, torch.get_rng_state())


@torch_available
def test_torch_seed_restore_random_seed_with_exception() -> None:
    state = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="Exception"), torch_seed(42):  # noqa: PT012
        torch.randn(4, 6)
        msg = "Exception"
        raise RuntimeError(msg)
    assert objects_are_equal(state, torch.get_rng_state())


@torch_available
def test_torch_seed_same_random_seed() -> None:
    with torch_seed(42):
        x1 = torch.randn(4, 6)
    with torch_seed(42):
        x2 = torch.randn(4, 6)
    assert x1.equal(x2)


@torch_available
def test_torch_seed_different_random_seeds() -> None:
    with torch_seed(42):
        x1 = torch.randn(4, 6)
    with torch_seed(142):
        x2 = torch.randn(4, 6)
    assert not x1.equal(x2)
