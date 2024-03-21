from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.random import NumpyRandomManager
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
def test_numpy_random_manager_get_rng_state() -> None:
    rng = NumpyRandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, (tuple, dict))


@numpy_available
def test_numpy_random_manager_manual_seed() -> None:
    seed_setter = NumpyRandomManager()
    seed_setter.manual_seed(42)
    x1 = np.random.randn(4, 6)
    x2 = np.random.randn(4, 6)
    seed_setter.manual_seed(42)
    x3 = np.random.randn(4, 6)
    assert np.array_equal(x1, x3)
    assert not np.array_equal(x1, x2)


@numpy_available
def test_numpy_random_manager_set_rng_state() -> None:
    seed_setter = NumpyRandomManager()
    state = seed_setter.get_rng_state()
    x1 = np.random.randn(4, 6)
    x2 = np.random.randn(4, 6)
    seed_setter.set_rng_state(state)
    x3 = np.random.randn(4, 6)
    assert np.array_equal(x1, x3)
    assert not np.array_equal(x1, x2)


def test_numpy_random_manager_no_numpy() -> None:
    with (
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`numpy` package is required but not installed."),
    ):
        NumpyRandomManager()
