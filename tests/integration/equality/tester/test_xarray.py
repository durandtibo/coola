from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    XarrayDataArrayEqualityTester,
    XarrayDatasetEqualityTester,
    XarrayVariableEqualityTester,
)
from coola.testing.fixtures import xarray_available, xarray_not_available
from coola.utils.imports import is_numpy_available, is_xarray_available
from tests.unit.equality.utils import ExamplePair

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_xarray_available():
    import xarray as xr
else:
    xr = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


XARRAY_DATASET_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6))}),
            expected=xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6))}),
        ),
        id="data_vars",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(coords={"z": np.arange(6)}),
            expected=xr.Dataset(coords={"z": np.arange(6)}),
        ),
        id="coords",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(attrs={"global": "meow"}),
            expected=xr.Dataset(attrs={"global": "meow"}),
        ),
        id="attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(
                data_vars={"x": xr.DataArray(np.arange(6))},
                coords={"z": np.arange(6)},
                attrs={"global": "meow"},
            ),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.arange(6))},
                coords={"z": np.arange(6)},
                attrs={"global": "meow"},
            ),
        ),
        id="data_vars and coords and attrs",
    ),
]
XARRAY_DATASET_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.zeros(6))}),
            expected=xr.Dataset(data_vars={"x": xr.DataArray(np.ones(6))}),
            expected_message="objects have different data_vars:",
        ),
        id="different data_vars",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(coords={"z": [1, 2, 3]}),
            expected=xr.Dataset(coords={"z": [0, 1, 2]}),
            expected_message="objects have different coords:",
        ),
        id="different coords",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(attrs={"global": "meow"}),
            expected=xr.Dataset(attrs={"global": "meowwww"}),
            expected_message="objects have different attrs:",
        ),
        id="different attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
            expected=np.ones((2, 3)),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
XARRAY_DATASET_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.5))}
            ),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.05))}
            ),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.005))}
            ),
            atol=0.1,
        ),
        id="atol=0.001",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.5))}
            ),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.05))}
            ),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.005))}
            ),
            rtol=0.1,
        ),
        id="rtol=0.001",
    ),
]

XARRAY_VARIABLE_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.arange(6)),
            expected=xr.Variable(dims=["z"], data=np.arange(6)),
        ),
        id="1d",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["x", "y"], data=np.ones((2, 3))),
            expected=xr.Variable(dims=["x", "y"], data=np.ones((2, 3))),
        ),
        id="2d",
    ),
]
XARRAY_VARIABLE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(6)),
            expected=xr.Variable(dims=["z"], data=np.zeros(6)),
            expected_message="objects have different data:",
        ),
        id="different data",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["x"], data=np.arange(6)),
            expected=xr.Variable(dims=["y"], data=np.arange(6)),
            expected_message="objects have different dims:",
        ),
        id="different dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
            expected=xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
            expected_message="objects have different attrs:",
        ),
        id="different attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(6)),
            expected=np.ones(6),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
XARRAY_VARIABLE_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.5)),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.05)),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.005)),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.5)),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.05)),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.005)),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]


###################################################
#     Tests for XarrayDataArrayEqualityTester     #
###################################################


@xarray_available
def test_xarray_data_array_equality_tester_with_xarray(config: EqualityConfig) -> None:
    assert XarrayDataArrayEqualityTester().objects_are_equal(
        xr.DataArray(np.arange(6)), xr.DataArray(np.arange(6)), config=config
    )


@xarray_not_available
def test_xarray_data_array_equality_tester_without_xarray() -> None:
    with pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."):
        XarrayDataArrayEqualityTester()


#################################################
#     Tests for XarrayDatasetEqualityTester     #
#################################################


@xarray_available
def test_xarray_dataset_equality_tester_with_xarray(config: EqualityConfig) -> None:
    assert XarrayDatasetEqualityTester().objects_are_equal(
        xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6))}),
        xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6))}),
        config=config,
    )


@xarray_not_available
def test_xarray_dataset_equality_tester_without_xarray() -> None:
    with pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."):
        XarrayDatasetEqualityTester()


##################################################
#     Tests for XarrayVariableEqualityTester     #
##################################################


@xarray_available
def test_xarray_variable_equality_tester_with_xarray(config: EqualityConfig) -> None:
    assert XarrayVariableEqualityTester().objects_are_equal(
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["z"], data=np.arange(6)),
        config=config,
    )


@xarray_not_available
def test_xarray_variable_equality_tester_without_xarray() -> None:
    with pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."):
        XarrayVariableEqualityTester()
