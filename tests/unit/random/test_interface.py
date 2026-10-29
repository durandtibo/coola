from __future__ import annotations

import contextlib
import random
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.equality import objects_are_equal
from coola.random import (
    PythonRandomManager,
    RandomManagerRegistry,
    get_default_registry,
    interface as random_interface,
    register_managers,
)
from coola.random.interface import (
    get_rng_state,
    manual_seed,
    random_seed,
    set_rng_state,
)
from coola.registry import AlreadyRegisteredError
from coola.testing.marks import (
    numpy_available,
    numpy_not_available,
    torch_available,
    torch_not_available,
)
from coola.utils.imports import is_torch_available

if TYPE_CHECKING:
    from collections.abc import Generator

if is_torch_available():
    import torch


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the default registry before and after each test to ensure
    test isolation."""
    random_interface._default_registry.reset()
    yield
    random_interface._default_registry.reset()


class CustomList(list):
    r"""Custom list subclass for testing purposes."""


#################################
#     Tests for random_seed     #
#################################


def test_random_seed_changes_the_state() -> None:
    state = random.getstate()
    with random_seed(42):
        random.uniform(0, 1)  # noqa: S311
        assert not objects_are_equal(state, random.getstate())


def test_random_seed_restore_random_seed() -> None:
    state = random.getstate()
    with random_seed(42):
        random.uniform(0, 1)  # noqa: S311
    assert objects_are_equal(state, random.getstate())


def test_random_seed_with_exception_raises_the_error() -> None:
    with pytest.raises(RuntimeError, match=r"Exception"), random_seed(42):  # noqa: PT012
        random.uniform(0, 1)  # noqa: S311
        msg = "Exception"
        raise RuntimeError(msg)


def test_random_seed_restore_random_seed_with_exception() -> None:
    state = random.getstate()
    with contextlib.suppress(RuntimeError), random_seed(42):
        random.uniform(0, 1)  # noqa: S311
        msg = "Exception"
        raise RuntimeError(msg)
    assert objects_are_equal(state, random.getstate())


def test_random_seed_same_random_seed() -> None:
    with random_seed(42):
        x1 = random.uniform(0, 1)  # noqa: S311
    with random_seed(42):
        x2 = random.uniform(0, 1)  # noqa: S311
    assert objects_are_equal(x1, x2)


def test_random_seed_different_random_seeds() -> None:
    with random_seed(42):
        x1 = random.uniform(0, 1)  # noqa: S311
    with random_seed(420):
        x2 = random.uniform(0, 1)  # noqa: S311
    assert not objects_are_equal(x1, x2)


@pytest.fixture
def seed_calls() -> tuple[list[int], list[int], Mock]:
    mock_manager = Mock()
    entered: list[int] = []
    exited: list[int] = []

    @contextmanager
    def fake_seed(seed: int) -> Generator[None]:
        entered.append(seed)
        yield
        exited.append(seed)

    mock_manager.seed = fake_seed
    return entered, exited, mock_manager


def test_random_seed_uses_custom_manager_enters_the_seed(
    seed_calls: tuple[list[int], list[int], Mock],
) -> None:
    """Test that random_seed delegates to the ``seed`` context manager
    of the provided manager."""
    entered, _, mock_manager = seed_calls
    with random_seed(42, registry=mock_manager):
        assert entered == [42]


def test_random_seed_uses_custom_manager_does_not_exit_inside_the_context(
    seed_calls: tuple[list[int], list[int], Mock],
) -> None:
    _, exited, mock_manager = seed_calls
    with random_seed(42, registry=mock_manager):
        assert exited == []


def test_random_seed_uses_custom_manager_exits_the_seed(
    seed_calls: tuple[list[int], list[int], Mock],
) -> None:
    _, exited, mock_manager = seed_calls
    with random_seed(42, registry=mock_manager):
        pass
    assert exited == [42]


###################################
#     Tests for get_rng_state     #
###################################


def test_get_rng_state_is_a_dict() -> None:
    state = get_rng_state()
    assert isinstance(state, dict)


def test_get_rng_state_not_empty() -> None:
    state = get_rng_state()
    assert len(state) >= 1


#################################
#     Tests for manual_seed     #
#################################


@torch_available
def test_manual_seed_same_seed_same_values() -> None:
    manual_seed(42)
    x1 = torch.randn(4, 6)
    torch.randn(4, 6)
    manual_seed(42)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)


@torch_available
def test_manual_seed_different_values() -> None:
    manual_seed(42)
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    manual_seed(42)
    torch.randn(4, 6)
    assert not x1.equal(x2)


def test_manual_seed_gets_the_default_registry() -> None:
    manager = Mock()
    with patch(
        "coola.random.interface.get_default_registry", Mock(return_value=manager)
    ) as get_registry:
        manual_seed(42)
    get_registry.assert_called_once()


def test_manual_seed_calls_default_registry() -> None:
    manager = Mock()
    with patch("coola.random.interface.get_default_registry", Mock(return_value=manager)):
        manual_seed(42)
    manager.manual_seed.assert_called_once_with(42)


###################################
#     Tests for set_rng_state     #
###################################


@torch_available
def test_set_rng_state_restored_values() -> None:
    state = get_rng_state()
    x1 = torch.randn(4, 6)
    torch.randn(4, 6)
    set_rng_state(state)
    x3 = torch.randn(4, 6)
    assert x1.equal(x3)


@torch_available
def test_set_rng_state_different_values() -> None:
    state = get_rng_state()
    x1 = torch.randn(4, 6)
    x2 = torch.randn(4, 6)
    set_rng_state(state)
    torch.randn(4, 6)
    assert not x1.equal(x2)


def test_set_rng_state_gets_the_default_registry() -> None:
    manager = Mock()
    with patch(
        "coola.random.interface.get_default_registry", Mock(return_value=manager)
    ) as get_registry:
        set_rng_state({"random": ("state",)})
    get_registry.assert_called_once()


def test_set_rng_state_calls_default_registry() -> None:
    manager = Mock()
    state = {"random": ("state",)}
    with patch("coola.random.interface.get_default_registry", Mock(return_value=manager)):
        set_rng_state(state)
    manager.set_rng_state.assert_called_once_with(state)


#######################################
#     Tests for register_managers     #
#######################################


def test_register_managers_calls_registry() -> None:
    """Test that register_managers successfully registers managers in
    the default registry."""
    register_managers({"custom": PythonRandomManager()})
    assert get_default_registry().has("custom")


def test_register_managers_with_exist_ok_true() -> None:
    """Test that register_managers allows re-registration when
    exist_ok=True."""
    register_managers({"custom": PythonRandomManager()})
    register_managers({"custom": PythonRandomManager()}, exist_ok=True)


def test_register_managers_with_exist_ok_false() -> None:
    """Test that register_managers raises RuntimeError when attempting
    to re-register with exist_ok=False."""
    register_managers({"custom": PythonRandomManager()})
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        register_managers({"custom": PythonRandomManager()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a RandomManagerRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, RandomManagerRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same singleton
    instance on multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


def test_get_default_registry_default() -> None:
    """Test that the 'random' manager is registered by default."""
    registry = get_default_registry()
    assert registry.has("random")


@numpy_available
def test_get_default_registry_with_numpy() -> None:
    """Test that the 'numpy' manager is registered when NumPy is
    available."""
    registry = get_default_registry()
    assert registry.has("numpy")


@numpy_not_available
def test_get_default_registry_without_numpy() -> None:
    """Test that the 'numpy' manager is not registered when NumPy is
    unavailable."""
    registry = get_default_registry()
    assert not registry.has("numpy")


@torch_available
def test_get_default_registry_with_torch() -> None:
    """Test that the 'torch' manager is registered when PyTorch is
    available."""
    registry = get_default_registry()
    assert registry.has("torch")


@torch_not_available
def test_get_default_registry_without_torch() -> None:
    """Test that the 'torch' manager is not registered when PyTorch is
    unavailable."""
    registry = get_default_registry()
    assert not registry.has("torch")


def test_get_default_registry_singleton_does_not_have_the_custom_manager() -> None:
    """Test that the registry does not have the custom manager."""
    assert not get_default_registry().has("custom")


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the singleton registry persist."""
    registry1 = get_default_registry()
    registry1.register("custom", PythonRandomManager())
    assert registry1.has("custom")


def test_get_default_registry_singleton_returns_the_same_registry() -> None:
    """Test that the registry is the same across calls."""
    registry1 = get_default_registry()
    registry1.register("custom", PythonRandomManager())
    assert registry1 is get_default_registry()


def test_get_default_registry_singleton_persists_modifications_across_calls() -> None:
    """Test that modifications persist across multiple calls to
    get_default_registry."""
    get_default_registry().register("custom", PythonRandomManager())
    assert get_default_registry().has("custom")


def test_random_seed_uses_an_empty_registry() -> None:
    state = random.getstate()
    with random_seed(42, registry=RandomManagerRegistry()):
        assert objects_are_equal(state, random.getstate())


def test_random_seed_registry_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        random_seed(42, RandomManagerRegistry())  # ty: ignore[too-many-positional-arguments]


def test_register_managers_exist_ok_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        register_managers({}, True)  # ty: ignore[too-many-positional-arguments]


####################################################
#     Tests for _register_default_managers     #
####################################################


def _default_registry_keys() -> list[str]:
    registry = RandomManagerRegistry()
    random_interface._register_default_managers(registry)
    return list(registry.keys())


def test_register_default_managers_neither_library() -> None:
    with (
        patch("coola.random.interface.is_numpy_available", lambda: False),
        patch("coola.random.interface.is_torch_available", lambda: False),
    ):
        assert _default_registry_keys() == ["random"]


@numpy_available
def test_register_default_managers_numpy_only() -> None:
    with patch("coola.random.interface.is_torch_available", lambda: False):
        assert _default_registry_keys() == ["random", "numpy"]


@torch_available
def test_register_default_managers_torch_only() -> None:
    with patch("coola.random.interface.is_numpy_available", lambda: False):
        assert _default_registry_keys() == ["random", "torch"]


@numpy_available
@torch_available
def test_register_default_managers_both_libraries() -> None:
    assert _default_registry_keys() == ["random", "numpy", "torch"]
