from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.equality import objects_are_equal
from coola.random import NumpyRandomManager, numpy_seed
from coola.testing.fixtures import numpy_available
from coola.utils.imports import MissingPackageError, is_numpy_available

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
def test_numpy_random_manager_manual_seed() -> None:
    rng = NumpyRandomManager()
    rng.manual_seed(42)
    x1 = np.random.randn(4, 6)  # noqa: NPY002
    x2 = np.random.randn(4, 6)  # noqa: NPY002
    rng.manual_seed(42)
    x3 = np.random.randn(4, 6)  # noqa: NPY002
    assert np.array_equal(x1, x3)
    assert not np.array_equal(x1, x2)


@numpy_available
def test_numpy_random_manager_set_rng_state() -> None:
    rng = NumpyRandomManager()
    state = rng.get_rng_state()
    x1 = np.random.randn(4, 6)  # noqa: NPY002
    x2 = np.random.randn(4, 6)  # noqa: NPY002
    rng.set_rng_state(state)
    x3 = np.random.randn(4, 6)  # noqa: NPY002
    assert np.array_equal(x1, x3)
    assert not np.array_equal(x1, x2)


def test_numpy_random_manager_no_numpy() -> None:
    with (
        patch("coola.utils.imports.numpy.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        NumpyRandomManager()


################################
#     Tests for numpy_seed     #
################################


@numpy_available
def test_numpy_seed_restore_random_seed() -> None:
    state = np.random.get_state()  # noqa: NPY002
    with numpy_seed(42):
        np.random.randn(4, 6)  # noqa: NPY002
    assert objects_are_equal(state, np.random.get_state())  # noqa: NPY002


@numpy_available
def test_numpy_seed_restore_random_seed_with_exception() -> None:
    state = np.random.get_state()  # noqa: NPY002
    with pytest.raises(RuntimeError, match=r"Exception"), numpy_seed(42):  # noqa: PT012
        np.random.randn(4, 6)  # noqa: NPY002
        msg = "Exception"
        raise RuntimeError(msg)
    assert objects_are_equal(state, np.random.get_state())  # noqa: NPY002


@numpy_available
def test_numpy_seed_same_random_seed() -> None:
    with numpy_seed(42):
        x1 = np.random.randn(4, 6)  # noqa: NPY002
    with numpy_seed(42):
        x2 = np.random.randn(4, 6)  # noqa: NPY002
    assert np.array_equal(x1, x2)


@numpy_available
def test_numpy_random_manager_seed() -> None:
    manager = NumpyRandomManager()
    state = np.random.get_state()  # noqa: NPY002
    with manager.seed(42):
        first = np.random.rand()  # noqa: NPY002
    assert objects_are_equal(state, np.random.get_state())  # noqa: NPY002
    with manager.seed(42):
        second = np.random.rand()  # noqa: NPY002
    assert first == second


@numpy_available
def test_numpy_random_manager_manual_seed_is_folded_modulo_2_32() -> None:
    manager = NumpyRandomManager()
    manager.manual_seed(1)
    first = np.random.rand(3)  # noqa: NPY002
    manager.manual_seed(2**32 + 1)
    second = np.random.rand(3)  # noqa: NPY002
    assert objects_are_equal(first, second)


@numpy_available
def test_numpy_random_manager_manual_seed_negative_seed_wraps() -> None:
    manager = NumpyRandomManager()
    manager.manual_seed(-1)
    first = np.random.rand(3)  # noqa: NPY002
    manager.manual_seed(2**32 - 1)
    second = np.random.rand(3)  # noqa: NPY002
    assert objects_are_equal(first, second)


@numpy_available
def test_numpy_random_manager_different_seeds_give_different_numbers() -> None:
    manager = NumpyRandomManager()
    manager.manual_seed(1)
    first = np.random.rand(3)  # noqa: NPY002
    manager.manual_seed(2)
    second = np.random.rand(3)  # noqa: NPY002
    assert not objects_are_equal(first, second)


@numpy_available
def test_numpy_random_manager_does_not_manage_default_rng_generators() -> None:
    manager = NumpyRandomManager()
    rng = np.random.default_rng(0)
    expected = np.random.default_rng(0).random(3)
    with manager.seed(42):
        values = rng.random(3)  # the independent generator is not affected by the seed
    assert objects_are_equal(values, expected)
