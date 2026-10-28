from __future__ import annotations

from coola.random import TorchRandomManager
from coola.testing.marks import torch_available, torch_cuda_available, torch_mps_available
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


@torch_mps_available
def test_torch_random_manager_manual_seed_seeds_mps() -> None:
    manager = TorchRandomManager()
    manager.manual_seed(42)
    first = torch.rand(3, device="mps")
    torch.rand(5, device="mps")
    manager.manual_seed(42)
    assert torch.rand(3, device="mps").equal(first)


@torch_cuda_available
def test_torch_random_manager_manual_seed_seeds_cuda() -> None:
    manager = TorchRandomManager()
    manager.manual_seed(42)
    first = torch.rand(3, device="cuda")
    torch.rand(5, device="cuda")
    manager.manual_seed(42)
    assert torch.rand(3, device="cuda").equal(first)
