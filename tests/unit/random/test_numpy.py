from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.equality import objects_are_equal
from coola.random import NumpyRandomManager
from coola.testing.marks import numpy_available
from coola.utils.imports import MissingPackageError, is_numpy_available
from tests.helpers import draw_numpy_rand, draw_numpy_randint, draw_numpy_randn, get_numpy_state

if is_numpy_available():
    import numpy as np

########################################
#     Tests for NumpyRandomManager     #
########################################


@numpy_available
def test_numpy_random_manager_repr() -> None:
    assert repr(NumpyRandomManager()).startswith("NumpyRandomManager(")


@numpy_available
def test_numpy_random_manager_str() -> None:
    assert str(NumpyRandomManager()).startswith("NumpyRandomManager(")


@numpy_available
def test_numpy_random_manager__eq__true() -> None:
    assert NumpyRandomManager() == NumpyRandomManager()


@numpy_available
def test_numpy_random_manager__eq__false_different_type() -> None:
    assert NumpyRandomManager() != 42


@numpy_available
def test_numpy_random_manager__eq__false_different_type_child() -> None:
    class Child(NumpyRandomManager): ...

    assert NumpyRandomManager() != Child()


@numpy_available
def test_numpy_random_manager__hash__() -> None:
    assert hash(NumpyRandomManager()) == hash(NumpyRandomManager())


@numpy_available
def test_numpy_random_manager__hash__different_type_child() -> None:
    class Child(NumpyRandomManager): ...

    assert hash(NumpyRandomManager()) != hash(Child())


@numpy_available
def test_numpy_random_manager_get_rng_state() -> None:
    rng = NumpyRandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, (tuple, dict))


@numpy_available
def test_numpy_random_manager_manual_seed_same_seed_same_values() -> None:
    rng = NumpyRandomManager()
    rng.manual_seed(42)
    x1 = draw_numpy_randn(4, 6)
    draw_numpy_randn(4, 6)
    rng.manual_seed(42)
    x3 = draw_numpy_randn(4, 6)
    assert np.array_equal(x1, x3)


@numpy_available
def test_numpy_random_manager_manual_seed_different_values() -> None:
    rng = NumpyRandomManager()
    rng.manual_seed(42)
    x1 = draw_numpy_randn(4, 6)
    x2 = draw_numpy_randn(4, 6)
    rng.manual_seed(42)
    draw_numpy_randn(4, 6)
    assert not np.array_equal(x1, x2)


@numpy_available
def test_numpy_random_manager_set_rng_state_restored_values() -> None:
    rng = NumpyRandomManager()
    state = rng.get_rng_state()
    x1 = draw_numpy_randn(4, 6)
    draw_numpy_randn(4, 6)
    rng.set_rng_state(state)
    x3 = draw_numpy_randn(4, 6)
    assert np.array_equal(x1, x3)


@numpy_available
def test_numpy_random_manager_set_rng_state_different_values() -> None:
    rng = NumpyRandomManager()
    state = rng.get_rng_state()
    x1 = draw_numpy_randn(4, 6)
    x2 = draw_numpy_randn(4, 6)
    rng.set_rng_state(state)
    draw_numpy_randn(4, 6)
    assert not np.array_equal(x1, x2)


def test_numpy_random_manager_no_numpy() -> None:
    with (
        patch("coola.utils.imports.optional.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        NumpyRandomManager()


@numpy_available
def test_numpy_random_manager_seed_restores_the_state() -> None:
    manager = NumpyRandomManager()
    state = get_numpy_state()
    with manager.seed(42):
        draw_numpy_rand()
    assert objects_are_equal(state, get_numpy_state())


@numpy_available
def test_numpy_random_manager_seed_is_reproducible() -> None:
    manager = NumpyRandomManager()
    with manager.seed(42):
        first = draw_numpy_rand()
    with manager.seed(42):
        second = draw_numpy_rand()
    assert first == second


@numpy_available
@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (0, [209652396, 398764591, 924231285]),
        (42, [1608637542, 1273642419, 1935803228]),
        (2**32 - 1, [419326371, 479346978, 1771170828]),
    ],
)
def test_numpy_random_manager_manual_seed_below_2_32_keeps_the_numbers(
    seed: int, expected: list[int]
) -> None:
    NumpyRandomManager().manual_seed(seed)
    assert draw_numpy_randint(0, 2**31, size=3).tolist() == expected


@numpy_available
def test_numpy_random_manager_manual_seed_not_folded_modulo_2_32() -> None:
    manager = NumpyRandomManager()
    manager.manual_seed(1)
    first = draw_numpy_rand(3)
    manager.manual_seed(2**32 + 1)
    second = draw_numpy_rand(3)
    assert not objects_are_equal(first, second)


@numpy_available
def test_numpy_random_manager_manual_seed_large_seed_same_seed_same_values() -> None:
    manager = NumpyRandomManager()
    manager.manual_seed(2**32 + 1)
    first = draw_numpy_rand(3)
    manager.manual_seed(2**32 + 1)
    second = draw_numpy_rand(3)
    assert objects_are_equal(first, second)


@numpy_available
def test_numpy_random_manager_manual_seed_large_seeds_with_same_high_word_differ() -> None:
    manager = NumpyRandomManager()
    manager.manual_seed(2**32)
    first = draw_numpy_rand(3)
    manager.manual_seed(2**32 + 2**32)
    second = draw_numpy_rand(3)
    assert not objects_are_equal(first, second)


@numpy_available
def test_numpy_random_manager_manual_seed_max_seed() -> None:
    NumpyRandomManager().manual_seed(2**63 - 1)
    assert draw_numpy_rand(3).shape == (3,)


@numpy_available
def test_numpy_random_manager_different_seeds_give_different_numbers() -> None:
    manager = NumpyRandomManager()
    manager.manual_seed(1)
    first = draw_numpy_rand(3)
    manager.manual_seed(2)
    second = draw_numpy_rand(3)
    assert not objects_are_equal(first, second)


@numpy_available
def test_numpy_random_manager_does_not_manage_default_rng_generators() -> None:
    manager = NumpyRandomManager()
    rng = np.random.default_rng(0)
    expected = np.random.default_rng(0).random(3)
    with manager.seed(42):
        values = rng.random(3)  # the independent generator is not affected by the seed
    assert objects_are_equal(values, expected)


@numpy_available
@pytest.mark.parametrize("seed", [-1, 2**63])
def test_numpy_random_manager_manual_seed_invalid_value_raises(seed: int) -> None:
    with pytest.raises(ValueError, match=r"seed must be in the range"):
        NumpyRandomManager().manual_seed(seed)


@numpy_available
@pytest.mark.parametrize("seed", ["a", 1.5, True])
def test_numpy_random_manager_manual_seed_invalid_type_raises(seed: object) -> None:
    with pytest.raises(TypeError, match=r"seed must be an int"):
        NumpyRandomManager().manual_seed(seed)  # ty: ignore[invalid-argument-type]
