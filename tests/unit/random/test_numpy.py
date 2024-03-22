from __future__ import annotations

from unittest.mock import patch

import pytest

from coola import objects_are_equal
from coola.random import NumpyRandomManager
from coola.random.numpy_ import get_random_managers, numpy_seed
from coola.testing import numpy_available
from coola.utils import is_numpy_available

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
def test_numpy_random_manager_eq_true() -> None:
    assert NumpyRandomManager() == NumpyRandomManager()


@numpy_available
def test_numpy_random_manager_eq_false() -> None:
    assert NumpyRandomManager() != 42


@numpy_available
def test_numpy_random_manager_get_rng_state() -> None:
    rng = NumpyRandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, (tuple, dict))


@numpy_available
def test_numpy_random_manager_manual_seed() -> None:
    rng = NumpyRandomManager()
    rng.manual_seed(42)
    x1 = np.random.randn(4, 6)
    x2 = np.random.randn(4, 6)
    rng.manual_seed(42)
    x3 = np.random.randn(4, 6)
    assert np.array_equal(x1, x3)
    assert not np.array_equal(x1, x2)


@numpy_available
def test_numpy_random_manager_set_rng_state() -> None:
    rng = NumpyRandomManager()
    state = rng.get_rng_state()
    x1 = np.random.randn(4, 6)
    x2 = np.random.randn(4, 6)
    rng.set_rng_state(state)
    x3 = np.random.randn(4, 6)
    assert np.array_equal(x1, x3)
    assert not np.array_equal(x1, x2)


def test_numpy_random_manager_no_numpy() -> None:
    with (
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`numpy` package is required but not installed."),
    ):
        NumpyRandomManager()


#########################################
#     Tests for get_random_managers     #
#########################################


@numpy_available
def test_get_random_managers() -> None:
    assert objects_are_equal(get_random_managers(), {"numpy": NumpyRandomManager()})


def test_get_random_managers_no_numpy() -> None:
    with patch("coola.random.numpy_.is_numpy_available", lambda: False):
        assert objects_are_equal(get_random_managers(), {})


################################
#     Tests for numpy_seed     #
################################


@numpy_available
def test_numpy_seed_restore_random_seed() -> None:
    state = np.random.get_state()
    with numpy_seed(42):
        np.random.randn(4, 6)
    assert objects_are_equal(state, np.random.get_state())


@numpy_available
def test_numpy_seed_restore_random_seed_with_exception() -> None:
    state = np.random.get_state()
    with pytest.raises(RuntimeError, match="Exception"), numpy_seed(42):  # noqa: PT012
        np.random.randn(4, 6)
        msg = "Exception"
        raise RuntimeError(msg)
    assert objects_are_equal(state, np.random.get_state())


@numpy_available
def test_numpy_seed_same_random_seed() -> None:
    with numpy_seed(42):
        x1 = np.random.randn(4, 6)
    with numpy_seed(42):
        x2 = np.random.randn(4, 6)
    assert np.array_equal(x1, x2)
