from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import PyarrowEqualityTester
from coola.testing.fixtures import pyarrow_available, pyarrow_not_available
from coola.utils.imports import is_pyarrow_available

if is_pyarrow_available():
    import pyarrow as pa


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


###########################################
#     Tests for PyarrowEqualityTester     #
###########################################


@pyarrow_available
def test_pyarrow_equality_tester_with_pyarrow(config: EqualityConfig) -> None:
    assert PyarrowEqualityTester().objects_are_equal(
        pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        config=config,
    )


@pyarrow_not_available
def test_pyarrow_equality_tester_without_pyarrow() -> None:
    with pytest.raises(RuntimeError, match=r"'pyarrow' package is required but not installed."):
        PyarrowEqualityTester()
