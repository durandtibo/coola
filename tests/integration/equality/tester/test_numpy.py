from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    NumpyArrayEqualityTester,
    NumpyMaskedArrayEqualityTester,
)
from coola.testing.marks import numpy_available, numpy_not_available
from coola.utils.imports import MissingPackageError, is_numpy_available
from tests.unit.equality.utils import run_tester

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
    assert run_tester(NumpyArrayEqualityTester(), np.ones((2, 3)), np.ones((2, 3)), config)


@numpy_not_available
def test_numpy_array_equality_tester_without_numpy() -> None:
    with pytest.raises(
        MissingPackageError, match=r"'numpy' package is required but not installed."
    ):
        NumpyArrayEqualityTester()


####################################################
#     Tests for NumpyMaskedArrayEqualityTester     #
####################################################


@numpy_available
def test_numpy_masked_array_equality_tester_with_numpy(config: EqualityConfig) -> None:
    assert run_tester(
        NumpyMaskedArrayEqualityTester(),
        np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
        np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
        config,
    )


@numpy_not_available
def test_numpy_masked_array_equality_tester_without_numpy() -> None:
    with pytest.raises(
        MissingPackageError, match=r"'numpy' package is required but not installed."
    ):
        NumpyMaskedArrayEqualityTester()
