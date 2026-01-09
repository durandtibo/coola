from __future__ import annotations

from coola.random.functional import get_rng_state, manual_seed, set_rng_state
from coola.testing import torch_available
from coola.utils import is_torch_available

if is_torch_available():
    import torch

########################################
#     Tests for TorchRandomManager     #
########################################


def test_get_rng_state() -> None:
    state = get_rng_state()
    assert isinstance(state, dict)
    assert len(state) >= 1


@torch_available
def test_manual_seed() -> None:
    manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)


@torch_available
def test_set_rng_state() -> None:
    state = get_rng_state()
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    set_rng_state(state)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)
    assert not x1.equal(x2)
