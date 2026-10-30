from __future__ import annotations

from coola.equality import objects_are_equal
from coola.random import get_default_registry, get_rng_state, manual_seed, set_rng_state
from coola.testing.marks import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available
from tests.helpers import draw_python_random

if is_numpy_available():
    import numpy as np

if is_torch_available():
    import torch

#####################################################
#     Tests for get_rng_state and set_rng_state     #
#####################################################


def test_get_rng_state_has_the_keys_of_the_default_registry() -> None:
    assert list(get_rng_state()) == list(get_default_registry().keys())


def test_set_rng_state_restores_random() -> None:
    state = get_rng_state()
    expected = draw_python_random()
    manual_seed(42)
    set_rng_state(state)
    assert draw_python_random() == expected


@numpy_available
def test_set_rng_state_restores_numpy() -> None:
    state = get_rng_state()
    expected = np.random.rand(3)  # noqa: NPY002
    manual_seed(42)
    set_rng_state(state)
    assert objects_are_equal(np.random.rand(3), expected)  # noqa: NPY002


@torch_available
def test_set_rng_state_restores_torch() -> None:
    state = get_rng_state()
    expected = torch.rand(3)
    manual_seed(42)
    set_rng_state(state)
    assert objects_are_equal(torch.rand(3), expected)


@numpy_available
@torch_available
def test_manual_seed_seeds_the_three_libraries() -> None:
    manual_seed(42)
    first = (draw_python_random(), np.random.rand(3), torch.rand(3))  # noqa: NPY002
    manual_seed(42)
    second = (draw_python_random(), np.random.rand(3), torch.rand(3))  # noqa: NPY002
    assert objects_are_equal(first, second)


@numpy_available
def test_numpy_array_seed_is_used_for_a_large_seed() -> None:
    manual_seed(2**40)
    first = np.random.rand(3)  # noqa: NPY002
    np.random.seed([2**40 & 0xFFFFFFFF, 2**40 >> 32])  # noqa: NPY002
    assert objects_are_equal(np.random.rand(3), first)  # noqa: NPY002
