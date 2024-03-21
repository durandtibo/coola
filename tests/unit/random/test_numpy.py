from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.random import NumpyRandomSeedSetter
from coola.testing import numpy_available
from coola.utils import is_numpy_available

if is_numpy_available():
    import numpy as np

###########################################
#     Tests for NumpyRandomSeedSetter     #
###########################################


@numpy_available
def test_numpy_random_seed_setter_repr() -> None:
    assert repr(NumpyRandomSeedSetter()).startswith("NumpyRandomSeedSetter(")


@numpy_available
def test_numpy_random_seed_setter_str() -> None:
    assert str(NumpyRandomSeedSetter()).startswith("NumpyRandomSeedSetter(")


@numpy_available
def test_numpy_random_seed_setter_manual_seed() -> None:
    seed_setter = NumpyRandomSeedSetter()
    seed_setter.manual_seed(42)
    x1 = np.random.randn(4, 6)
    x2 = np.random.randn(4, 6)
    seed_setter.manual_seed(42)
    x3 = np.random.randn(4, 6)
    assert np.array_equal(x1, x3)
    assert not np.array_equal(x1, x2)


def test_numpy_random_seed_setter_no_numpy() -> None:
    with (
        patch("coola.utils.imports.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match="`numpy` package is required but not installed."),
    ):
        NumpyRandomSeedSetter()
