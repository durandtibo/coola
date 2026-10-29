from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from coola.equality import objects_are_equal
from coola.random import (
    BaseRandomManager,
    NumpyRandomManager,
    PythonRandomManager,
    TorchRandomManager,
)
from coola.testing.marks import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

if is_numpy_available():
    import numpy as np

if is_torch_available():
    import torch


def _draw_python() -> Any:
    return [random.random() for _ in range(3)]  # noqa: S311


def _draw_numpy() -> Any:
    return np.random.rand(3)  # noqa: NPY002


def _draw_torch() -> Any:
    return torch.rand(3)


# Each manager is paired with a function that draws numbers from the generator that it manages.
MANAGERS: list[tuple[type[BaseRandomManager], Callable[[], Any]]] = [
    (PythonRandomManager, _draw_python)
]
if is_numpy_available():
    MANAGERS.append((NumpyRandomManager, _draw_numpy))
if is_torch_available():
    MANAGERS.append((TorchRandomManager, _draw_torch))

MANAGER_IDS = [cls.__name__ for cls, _ in MANAGERS]


@pytest.fixture(autouse=True)
def _restore_generators() -> Generator[None]:
    states = [cls().get_rng_state() for cls, _ in MANAGERS]
    yield
    for (cls, _), state in zip(MANAGERS, states, strict=True):
        cls().set_rng_state(state)


################################################
#     Tests for the contract of the managers     #
################################################


@pytest.mark.parametrize(("cls", "draw"), MANAGERS, ids=MANAGER_IDS)
def test_manager_get_rng_state_does_not_alias_the_generator(
    cls: type[BaseRandomManager], draw: Callable[[], Any]
) -> None:
    manager = cls()
    state = manager.get_rng_state()
    snapshot = copy.deepcopy(state)
    draw()
    assert objects_are_equal(state, snapshot)


@pytest.mark.parametrize(("cls", "draw"), MANAGERS, ids=MANAGER_IDS)
def test_manager_set_rng_state_of_get_rng_state_leaves_the_draws_unchanged(
    cls: type[BaseRandomManager], draw: Callable[[], Any]
) -> None:
    manager = cls()
    manager.manual_seed(0)
    expected = draw()
    manager.manual_seed(0)
    manager.set_rng_state(manager.get_rng_state())
    assert objects_are_equal(draw(), expected)


@pytest.mark.parametrize(("cls", "draw"), MANAGERS, ids=MANAGER_IDS)
def test_manager_set_rng_state_restores_the_draws(
    cls: type[BaseRandomManager], draw: Callable[[], Any]
) -> None:
    manager = cls()
    state = manager.get_rng_state()
    expected = draw()
    manager.set_rng_state(state)
    assert objects_are_equal(draw(), expected)


@pytest.mark.parametrize(("cls", "draw"), MANAGERS, ids=MANAGER_IDS)
def test_manager_manual_seed_same_seed_same_draws(
    cls: type[BaseRandomManager], draw: Callable[[], Any]
) -> None:
    manager = cls()
    manager.manual_seed(42)
    expected = draw()
    manager.manual_seed(42)
    assert objects_are_equal(draw(), expected)


# PyTorch folds the seeds modulo 2**32 in its CPU generator, so the seeds >= 2**32 are not compared
# here (NumPy distinguishes them, see ``test_numpy.py``).
@pytest.mark.parametrize(("cls", "draw"), MANAGERS, ids=MANAGER_IDS)
@pytest.mark.parametrize(("seed", "other"), [(0, 1), (1, 2), (2**31, 2**32 - 1)])
def test_manager_manual_seed_different_seeds_different_draws(
    cls: type[BaseRandomManager], draw: Callable[[], Any], seed: int, other: int
) -> None:
    manager = cls()
    manager.manual_seed(seed)
    first = draw()
    manager.manual_seed(other)
    assert not objects_are_equal(draw(), first)


@pytest.mark.parametrize(("cls", "draw"), MANAGERS, ids=MANAGER_IDS)
@pytest.mark.parametrize("seed", [-1, 2**63, "a", 1.5, True])
def test_manager_manual_seed_invalid_seed_does_not_touch_the_generator(
    cls: type[BaseRandomManager], draw: Callable[[], Any], seed: Any
) -> None:
    manager = cls()
    manager.manual_seed(0)
    expected = draw()
    manager.manual_seed(0)
    with pytest.raises((TypeError, ValueError), match="seed must be"):
        manager.manual_seed(seed)
    assert objects_are_equal(draw(), expected)


@pytest.mark.parametrize(("cls", "draw"), MANAGERS, ids=MANAGER_IDS)
def test_manager_seed_restores_the_draws(
    cls: type[BaseRandomManager], draw: Callable[[], Any]
) -> None:
    manager = cls()
    state = manager.get_rng_state()
    expected = draw()
    manager.set_rng_state(state)
    with manager.seed(42):
        draw()
    assert objects_are_equal(draw(), expected)


@pytest.mark.parametrize("cls", [cls for cls, _ in MANAGERS], ids=MANAGER_IDS)
def test_manager_is_equal_to_another_instance(cls: type[BaseRandomManager]) -> None:
    assert cls() == cls()


@pytest.mark.parametrize("cls", [cls for cls, _ in MANAGERS], ids=MANAGER_IDS)
def test_manager_has_the_hash_of_another_instance(cls: type[BaseRandomManager]) -> None:
    assert hash(cls()) == hash(cls())


@pytest.mark.parametrize("cls", [cls for cls, _ in MANAGERS], ids=MANAGER_IDS)
def test_manager_is_not_equal_to_another_type(cls: type[BaseRandomManager]) -> None:
    assert cls() != 42


@numpy_available
def test_numpy_manager_without_numpy_raises_import_error() -> None:
    with (
        patch("coola.utils.imports.optional.is_numpy_available", lambda: False),
        pytest.raises(ImportError, match="numpy"),
    ):
        NumpyRandomManager()


@torch_available
def test_torch_manager_without_torch_raises_import_error() -> None:
    with (
        patch("coola.utils.imports.optional.is_torch_available", lambda: False),
        pytest.raises(ImportError, match="torch"),
    ):
        TorchRandomManager()
