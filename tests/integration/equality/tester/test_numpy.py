from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    NumpyArrayEqualityTester,
    NumpyMaskedArrayEqualityTester,
)
from coola.testing.fixtures import numpy_available, numpy_not_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


##############################################
#     Tests for NumpyArrayEqualityTester     #
##############################################


@numpy_available
def test_numpy_array_equality_tester_with_numpy(config: EqualityConfig) -> None:
    assert NumpyArrayEqualityTester().objects_are_equal(
        np.ones((2, 3)), np.ones((2, 3)), config=config
    )


@numpy_not_available
def test_numpy_array_equality_tester_without_numpy() -> None:
    with pytest.raises(RuntimeError, match=r"'numpy' package is required but not installed."):
        NumpyArrayEqualityTester()


####################################################
#     Tests for NumpyMaskedArrayEqualityTester     #
####################################################


@numpy_available
def test_numpy_masked_array_equality_tester_with_numpy(config: EqualityConfig) -> None:
    assert NumpyMaskedArrayEqualityTester().objects_are_equal(
        np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
        np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
        config=config,
    )


@numpy_not_available
def test_numpy_masked_array_equality_tester_without_numpy() -> None:
    with pytest.raises(RuntimeError, match=r"'numpy' package is required but not installed."):
        NumpyMaskedArrayEqualityTester()
