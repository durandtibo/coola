from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.equality import objects_are_equal
from coola.random import TorchRandomManager, get_random_seed, get_torch_generator, validate_seed
from coola.testing.marks import torch_available
from coola.utils.imports import MissingPackageError, is_torch_available

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
def test_torch_random_manager__eq__true() -> None:
    assert TorchRandomManager() == TorchRandomManager()


@torch_available
def test_torch_random_manager__eq__false_different_type() -> None:
    assert TorchRandomManager() != 42


@torch_available
def test_torch_random_manager__eq__false_different_type_child() -> None:
    class Child(TorchRandomManager): ...

    assert TorchRandomManager() != Child()


@torch_available
def test_torch_random_manager__hash__() -> None:
    assert hash(TorchRandomManager()) == hash(TorchRandomManager())


@torch_available
def test_torch_random_manager__hash__different_type_child() -> None:
    class Child(TorchRandomManager): ...

    assert hash(TorchRandomManager()) != hash(Child())


@torch_available
@patch("torch.backends.mps.is_available", lambda: False)
def test_torch_random_manager_get_rng_state_is_a_dict() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, dict)


@torch_available
@patch("torch.backends.mps.is_available", lambda: False)
@patch("torch.cuda.is_initialized", lambda: False)
def test_torch_random_manager_get_rng_state_length() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    assert len(state) == 1


@torch_available
@patch("torch.backends.mps.is_available", lambda: False)
def test_torch_random_manager_get_rng_state_torch_state() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    assert torch.is_tensor(state["torch"])


@torch_available
@patch("torch.backends.mps.is_available", lambda: False)
@patch("torch.cuda.is_initialized", lambda: True)
@patch("torch.cuda.get_rng_state_all", list)
def test_torch_random_manager_get_rng_state_cuda_state() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    assert state["torch.cuda"] == []


@torch_available
@patch("torch.backends.mps.is_available", lambda: False)
@patch("torch.cuda.is_initialized", lambda: False)
def test_torch_random_manager_get_rng_state_cuda_not_initialized_has_no_cuda_state() -> None:
    state = TorchRandomManager().get_rng_state()
    assert "torch.cuda" not in state


@torch_available
@patch("torch.backends.mps.is_available", lambda: False)
@patch("torch.cuda.is_initialized", lambda: False)
def test_torch_random_manager_get_rng_state_cuda_not_initialized_does_not_read_cuda() -> None:
    with patch("torch.cuda.get_rng_state_all") as get_rng_state_all:
        TorchRandomManager().get_rng_state()
    get_rng_state_all.assert_not_called()


@torch_available
@patch("torch.backends.mps.is_available", lambda: False)
@patch("torch.cuda.is_initialized", lambda: False)
def test_torch_random_manager_set_rng_state_without_cuda_state_restores_the_cpu_state() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    expected = torch.rand(2)
    rng.set_rng_state(state)
    assert torch.rand(2).equal(expected)


@torch_available
def test_torch_random_manager_manual_seed_same_seed_same_values() -> None:
    rng = TorchRandomManager()
    rng.manual_seed(42)
    x1 = torch.randn(4, 6)
    torch.randn(4, 6)
    rng.manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)


@torch_available
def test_torch_random_manager_manual_seed_different_values() -> None:
    rng = TorchRandomManager()
    rng.manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    rng.manual_seed(42)
    torch.randn(4, 6)
    assert not x1.equal(x2)


@torch_available
def test_torch_random_manager_manual_seed_calls_torch_manual_seed() -> None:
    with patch("torch.manual_seed") as mock_manual_seed:
        TorchRandomManager().manual_seed(42)
    mock_manual_seed.assert_called_once_with(42)


@torch_available
@patch("torch.backends.mps.is_available", lambda: True)
def test_torch_random_manager_get_rng_state_with_mps() -> None:
    with patch("torch.mps.get_rng_state", return_value=torch.zeros(1)):
        state = TorchRandomManager().get_rng_state()
    assert objects_are_equal(state["torch.mps"], torch.zeros(1))


@torch_available
@patch("torch.backends.mps.is_available", lambda: True)
def test_torch_random_manager_set_rng_state_with_mps() -> None:
    with patch("torch.mps.set_rng_state") as mock_set_rng_state:
        TorchRandomManager().set_rng_state({"torch.mps": torch.zeros(1)})
        mock_set_rng_state.assert_called_once()


@torch_available
@patch("torch.backends.mps.is_available", lambda: False)
def test_torch_random_manager_set_rng_state_mps_not_available_raises() -> None:
    with pytest.raises(ValueError, match="MPS is not available"):
        TorchRandomManager().set_rng_state({"torch.mps": torch.zeros(1)})


@torch_available
def test_torch_random_manager_set_rng_state_partial_state() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    x1 = torch.randn(4, 6)
    rng.set_rng_state({"torch": state["torch"]})
    assert x1.equal(torch.randn(4, 6))


@torch_available
def test_torch_random_manager_set_rng_state_wrong_number_of_cuda_devices_raises() -> None:
    with (
        patch("torch.cuda.device_count", lambda: 2),
        pytest.raises(ValueError, match="CUDA generators"),
    ):
        TorchRandomManager().set_rng_state({"torch.cuda": [torch.zeros(1)]})


@torch_available
def test_torch_random_manager_set_rng_state_with_cuda() -> None:
    with (
        patch("torch.cuda.device_count", lambda: 1),
        patch("torch.cuda.set_rng_state_all") as mock_set_rng_state_all,
    ):
        TorchRandomManager().set_rng_state({"torch.cuda": [torch.zeros(1)]})
    mock_set_rng_state_all.assert_called_once()


@torch_available
def test_torch_random_manager_set_rng_state_restored_values() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    x1 = torch.randn(4, 6)
    torch.randn(4, 6)
    rng.set_rng_state(state)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)


@torch_available
def test_torch_random_manager_set_rng_state_different_values() -> None:
    rng = TorchRandomManager()
    state = rng.get_rng_state()
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    rng.set_rng_state(state)
    torch.randn(4, 6)
    assert not x1.equal(x2)


def test_torch_random_manager_no_torch() -> None:
    with (
        patch("coola.utils.imports.optional.is_torch_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'torch' package is required but not installed."),
    ):
        TorchRandomManager()


@torch_available
def test_torch_random_manager_seed_restores_the_state() -> None:
    manager = TorchRandomManager()
    state = torch.get_rng_state()
    with manager.seed(42):
        torch.rand(3)
    assert objects_are_equal(state, torch.get_rng_state())


@torch_available
def test_torch_random_manager_seed_is_reproducible() -> None:
    manager = TorchRandomManager()
    with manager.seed(42):
        first = torch.rand(3)
    with manager.seed(42):
        second = torch.rand(3)
    assert objects_are_equal(first, second)


@torch_available
def test_torch_random_manager_does_not_manage_generator_objects() -> None:
    manager = TorchRandomManager()
    generator = torch.Generator().manual_seed(0)
    expected = torch.rand(3, generator=torch.Generator().manual_seed(0))
    with manager.seed(42):
        values = torch.rand(3, generator=generator)
    assert objects_are_equal(values, expected)


#####################################
#     Tests for get_random_seed     #
#####################################


@torch_available
def test_get_random_seed_returns_int() -> None:
    assert isinstance(get_random_seed(42), int)


@torch_available
def test_get_random_seed_same_seed() -> None:
    assert get_random_seed(42) == get_random_seed(42)


@torch_available
def test_get_random_seed_different_seeds() -> None:
    assert get_random_seed(1) != get_random_seed(42)


@torch_available
def test_get_random_seed_value() -> None:
    assert get_random_seed(44) == 6176747449835261348


@torch_available
def test_get_random_seed_does_not_change_the_default_generator() -> None:
    state = torch.get_rng_state()
    get_random_seed(42)
    assert objects_are_equal(state, torch.get_rng_state())


@torch_available
def test_get_random_seed_in_range() -> None:
    assert 0 <= get_random_seed(42) < 2**63 - 1


#########################################
#     Tests for get_torch_generator     #
#########################################


@torch_available
def test_get_torch_generator_returns_generator() -> None:
    assert isinstance(get_torch_generator(42), torch.Generator)


@torch_available
def test_get_torch_generator_same_seed() -> None:
    assert torch.randn(4, 6, generator=get_torch_generator(1)).equal(
        torch.randn(4, 6, generator=get_torch_generator(1))
    )


@torch_available
def test_get_torch_generator_different_seeds() -> None:
    assert not torch.randn(4, 6, generator=get_torch_generator(1)).equal(
        torch.randn(4, 6, generator=get_torch_generator(2))
    )


@torch_available
def test_get_torch_generator_initial_seed() -> None:
    assert get_torch_generator(42).initial_seed() == 42


@torch_available
def test_get_torch_generator_default_device_is_cpu() -> None:
    assert get_torch_generator(42).device == torch.device("cpu")


@torch_available
@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_get_torch_generator_device(device: torch.device | str) -> None:
    assert get_torch_generator(42, device=device).device == torch.device("cpu")


@torch_available
def test_get_torch_generator_values() -> None:
    assert torch.allclose(
        torch.rand(2, 4, generator=get_torch_generator(42)),
        torch.tensor([[0.8823, 0.9150, 0.3829, 0.9593], [0.3904, 0.6009, 0.2566, 0.7936]]),
        atol=1e-4,
    )


@torch_available
def test_get_torch_generator_does_not_change_the_default_generator() -> None:
    state = torch.get_rng_state()
    get_torch_generator(42)
    assert objects_are_equal(state, torch.get_rng_state())


@torch_available
@pytest.mark.parametrize("seed", [-1, 2**63])
def test_torch_random_manager_manual_seed_invalid_value_raises(seed: int) -> None:
    with pytest.raises(ValueError, match=r"seed must be in the range"):
        TorchRandomManager().manual_seed(seed)


@torch_available
@pytest.mark.parametrize("seed", ["a", 1.5, True])
def test_torch_random_manager_manual_seed_invalid_type_raises(seed: object) -> None:
    with pytest.raises(TypeError, match=r"seed must be an int"):
        TorchRandomManager().manual_seed(seed)  # ty: ignore[invalid-argument-type]


@torch_available
def test_get_torch_generator_seed_is_required() -> None:
    with pytest.raises(TypeError, match="seed"):
        get_torch_generator()  # ty: ignore[missing-argument]


@torch_available
def test_get_torch_generator_device_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        get_torch_generator(42, "cpu")  # ty: ignore[too-many-positional-arguments]


@torch_available
@pytest.mark.parametrize("seed", [-1, 2**63, 2**64])
def test_get_torch_generator_invalid_value_raises(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be in the range"):
        get_torch_generator(seed)


@torch_available
def test_get_torch_generator_invalid_type_raises() -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        get_torch_generator(1.5)  # ty: ignore[invalid-argument-type]


@torch_available
def test_get_random_seed_is_a_valid_seed() -> None:
    validate_seed(get_random_seed(42))


@torch_available
def test_get_random_seed_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="seed must be in the range"):
        get_random_seed(-1)
