from __future__ import annotations

from coola.random import TorchRandomManager
from coola.testing.fixtures import torch_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch

########################################
#     Tests for TorchRandomManager     #
########################################


@torch_available
def test_torch_random_manager_manual_seed_seeds_torch_rand() -> None:
    manager = TorchRandomManager()
    manager.manual_seed(42)
    first = torch.rand(3)
    manager.manual_seed(42)
    assert torch.rand(3).equal(first)


@torch_available
def test_torch_random_manager_manual_seed_matches_torch_manual_seed() -> None:
    TorchRandomManager().manual_seed(42)
    first = torch.rand(3)
    torch.manual_seed(42)
    assert torch.rand(3).equal(first)

