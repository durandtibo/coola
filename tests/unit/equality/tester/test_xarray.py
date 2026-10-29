from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from coola.equality import compare
from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    XarrayDataArrayEqualityTester,
    XarrayDatasetEqualityTester,
    XarrayVariableEqualityTester,
)
from coola.equality.tolerance import Tolerance
from coola.testing.marks import xarray_available
from coola.utils.imports import MissingPackageError, is_numpy_available, is_xarray_available
from tests.unit.equality.utils import ExamplePair, explaining, messages, run_tester

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


XARRAY_DATA_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6)),
            expected=xr.DataArray(np.arange(6)),
        ),
        id="1d without dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), dims=["z"]),
            expected=xr.DataArray(np.arange(6), dims=["z"]),
        ),
        id="1d with dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.ones((2, 3))),
        ),
        id="2d without dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3)), dims=["x", "y"]),
            expected=xr.DataArray(np.ones((2, 3)), dims=["x", "y"]),
        ),
        id="2d with dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6)),
            expected=xr.DataArray(np.arange(6)),
        ),
        id="int dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6, dtype=float)),
            expected=xr.DataArray(np.arange(6, dtype=float)),
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
            expected=xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
        ),
        id="dims and coords",
    ),
]
XARRAY_DATA_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones(6)),
            expected=xr.DataArray(np.zeros(6)),
            expected_message="values are different:",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), dims=["z1"]),
            expected=xr.DataArray(np.arange(6), dims=["z2"]),
            expected_message="objects are different:",
        ),
        id="different dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), name="meow"),
            expected=xr.DataArray(np.arange(6), name="bear"),
            expected_message="objects are different:",
        ),
        id="different name",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), attrs={"global": "meow"}),
            expected=xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
            expected_message="objects are different:",
        ),
        id="different attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), coords={"z": [1, 2, 3, 4, 5, 6]}),
            expected=xr.DataArray(np.arange(6), coords={"z": [10, 20, 30, 40, 50, 60]}),
            expected_message="values are different:",
        ),
        id="different coords",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6)),
            expected=np.arange(6),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
XARRAY_DATA_ARRAY_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.5)),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.05)),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.005)),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.5)),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.05)),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.005)),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

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
            expected_message="values are different:",
        ),
        id="different data_vars",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(coords={"z": [1, 2, 3]}),
            expected=xr.Dataset(coords={"z": [0, 1, 2]}),
            expected_message="values are different:",
        ),
        id="different coords",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(attrs={"global": "meow"}),
            expected=xr.Dataset(attrs={"global": "meowwww"}),
            expected_message="objects are different:",
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
            expected_message="values are different:",
        ),
        id="different data",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["x"], data=np.arange(6)),
            expected=xr.Variable(dims=["y"], data=np.arange(6)),
            expected_message="objects are different:",
        ),
        id="different dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
            expected=xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
            expected_message="objects are different:",
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

XARRAY_EQUAL = XARRAY_DATA_ARRAY_EQUAL + XARRAY_DATASET_EQUAL + XARRAY_VARIABLE_EQUAL
XARRAY_NOT_EQUAL = (
    XARRAY_DATA_ARRAY_NOT_EQUAL + XARRAY_DATASET_NOT_EQUAL + XARRAY_VARIABLE_NOT_EQUAL
)
XARRAY_EQUAL_TOLERANCE = (
    XARRAY_DATA_ARRAY_EQUAL_TOLERANCE
    + XARRAY_DATASET_EQUAL_TOLERANCE
    + XARRAY_VARIABLE_EQUAL_TOLERANCE
)


###################################################
#     Tests for XarrayDataArrayEqualityTester     #
###################################################


@xarray_available
def test_xarray_data_array_equality_tester_repr() -> None:
    assert repr(XarrayDataArrayEqualityTester()).startswith("XarrayDataArrayEqualityTester(")


@xarray_available
def test_xarray_data_array_equality_tester_str() -> None:
    assert str(XarrayDataArrayEqualityTester()).startswith("XarrayDataArrayEqualityTester(")


@xarray_available
def test_xarray_data_array_equality_tester_equal_true() -> None:
    assert XarrayDataArrayEqualityTester() == XarrayDataArrayEqualityTester()


@xarray_available
def test_xarray_data_array_equality_tester_equal_false_different_type() -> None:
    assert XarrayDataArrayEqualityTester() != 123


@xarray_available
def test_xarray_data_array_equality_tester_equal_false_different_type_child() -> None:
    class Child(XarrayDataArrayEqualityTester): ...

    assert XarrayDataArrayEqualityTester() != Child()


@xarray_available
def test_xarray_data_array_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = xr.DataArray(np.arange(6))
    assert run_tester(XarrayDataArrayEqualityTester(), obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayDataArrayEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayDataArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayDataArrayEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayDataArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_NOT_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayDataArrayEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_NOT_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayDataArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_NOT_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayDataArrayEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_NOT_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayDataArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_data_array_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            XarrayDataArrayEqualityTester(),
            xr.DataArray(np.array([0.0, float("nan"), 2.0])),
            xr.DataArray(np.array([0.0, float("nan"), 2.0])),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL_TOLERANCE)
def test_xarray_data_array_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config = replace(config, tolerance=Tolerance(atol=example.atol, rtol=example.rtol))
    assert run_tester(XarrayDataArrayEqualityTester(), example.actual, example.expected, config)


def test_xarray_data_array_equality_tester_no_xarray() -> None:
    with (
        patch("coola.utils.imports.optional.is_xarray_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'xarray' package is required but not installed."
        ),
    ):
        XarrayDataArrayEqualityTester()


#################################################
#     Tests for XarrayDatasetEqualityTester     #
#################################################


@xarray_available
def test_xarray_dataset_equality_tester_repr() -> None:
    assert repr(XarrayDatasetEqualityTester()).startswith("XarrayDatasetEqualityTester(")


@xarray_available
def test_xarray_dataset_equality_tester_str() -> None:
    assert str(XarrayDatasetEqualityTester()).startswith("XarrayDatasetEqualityTester(")


@xarray_available
def test_xarray_dataset_equality_tester_equal_true() -> None:
    assert XarrayDatasetEqualityTester() == XarrayDatasetEqualityTester()


@xarray_available
def test_xarray_dataset_equality_tester_equal_false_different_type() -> None:
    assert XarrayDatasetEqualityTester() != 123


@xarray_available
def test_xarray_dataset_equality_tester_equal_false_different_type_child() -> None:
    class Child(XarrayDatasetEqualityTester): ...

    assert XarrayDatasetEqualityTester() != Child()


@xarray_available
def test_xarray_dataset_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])})
    assert run_tester(XarrayDatasetEqualityTester(), obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayDatasetEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayDatasetEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayDatasetEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayDatasetEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_NOT_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayDatasetEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_NOT_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayDatasetEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_NOT_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayDatasetEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_NOT_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayDatasetEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_dataset_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            XarrayDatasetEqualityTester(),
            xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
            xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL_TOLERANCE)
def test_xarray_dataset_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config = replace(config, tolerance=Tolerance(atol=example.atol, rtol=example.rtol))
    assert run_tester(XarrayDatasetEqualityTester(), example.actual, example.expected, config)


def test_xarray_dataset_equality_tester_no_xarray() -> None:
    with (
        patch("coola.utils.imports.optional.is_xarray_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'xarray' package is required but not installed."
        ),
    ):
        XarrayDatasetEqualityTester()


##################################################
#     Tests for XarrayVariableEqualityTester     #
##################################################


@xarray_available
def test_xarray_variable_equality_tester_repr() -> None:
    assert repr(XarrayVariableEqualityTester()).startswith("XarrayVariableEqualityTester(")


@xarray_available
def test_xarray_variable_equality_tester_str() -> None:
    assert str(XarrayVariableEqualityTester()).startswith("XarrayVariableEqualityTester(")


@xarray_available
def test_xarray_variable_equality_tester_equal_true() -> None:
    assert XarrayVariableEqualityTester() == XarrayVariableEqualityTester()


@xarray_available
def test_xarray_variable_equality_tester_equal_false_different_type() -> None:
    assert XarrayVariableEqualityTester() != 123


@xarray_available
def test_xarray_variable_equality_tester_equal_false_different_type_child() -> None:
    class Child(XarrayVariableEqualityTester): ...

    assert XarrayVariableEqualityTester() != Child()


@xarray_available
def test_xarray_variable_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = xr.Variable(dims=["z"], data=np.arange(6))
    assert run_tester(XarrayVariableEqualityTester(), obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayVariableEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayVariableEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayVariableEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayVariableEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_NOT_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayVariableEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_NOT_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = XarrayVariableEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_NOT_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayVariableEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_NOT_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = XarrayVariableEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_variable_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            XarrayVariableEqualityTester(),
            xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
            xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL_TOLERANCE)
def test_xarray_variable_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config = replace(config, tolerance=Tolerance(atol=example.atol, rtol=example.rtol))
    assert run_tester(XarrayVariableEqualityTester(), example.actual, example.expected, config)


def test_xarray_variable_equality_tester_no_xarray() -> None:
    with (
        patch("coola.utils.imports.optional.is_xarray_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'xarray' package is required but not installed."
        ),
    ):
        XarrayVariableEqualityTester()


@xarray_available
def test_xarray_data_array_equality_tester_non_index_coordinate(config: EqualityConfig) -> None:
    tester = XarrayDataArrayEqualityTester()
    kwargs: dict[str, Any] = {"dims": ["z"], "coords": {"label": ("z", ["a", "b", "c"])}}
    actual = xr.DataArray(np.arange(3), **kwargs)
    assert run_tester(tester, actual, xr.DataArray(np.arange(3), **kwargs), config)


@xarray_available
def test_xarray_data_array_equality_tester_non_index_coordinate_different(
    config: EqualityConfig,
) -> None:
    tester = XarrayDataArrayEqualityTester()
    kwargs: dict[str, Any] = {"dims": ["z"], "coords": {"label": ("z", ["a", "b", "c"])}}
    actual = xr.DataArray(np.arange(3), **kwargs)
    other = xr.DataArray(np.arange(3), dims=["z"], coords={"label": ("z", ["a", "b", "d"])})
    assert not run_tester(tester, actual, other, config)


@xarray_available
def test_xarray_data_array_equality_tester_missing_coordinate(config: EqualityConfig) -> None:
    tester = XarrayDataArrayEqualityTester()
    with_coord = xr.DataArray(np.arange(3), dims=["z"], coords={"z": [1, 2, 3]})
    without_coord = xr.DataArray(np.arange(3), dims=["z"])
    assert not run_tester(tester, with_coord, without_coord, config)


@xarray_available
def test_xarray_data_array_equality_tester_missing_expected_coordinate(
    config: EqualityConfig,
) -> None:
    tester = XarrayDataArrayEqualityTester()
    with_coord = xr.DataArray(np.arange(3), dims=["z"], coords={"z": [1, 2, 3]})
    without_coord = xr.DataArray(np.arange(3), dims=["z"])
    assert not run_tester(tester, without_coord, with_coord, config)


@xarray_available
def test_xarray_data_array_equality_tester_coordinate_without_tolerance() -> None:
    tester = XarrayDataArrayEqualityTester()
    actual = xr.DataArray(np.arange(3), dims=["z"], coords={"z": [1.0, 2.0, 3.0]})
    expected = xr.DataArray(np.arange(3), dims=["z"], coords={"z": [1.0, 2.0, 3.05]})
    assert not run_tester(tester, actual, expected, EqualityConfig())


@xarray_available
def test_xarray_data_array_equality_tester_coordinate_tolerance() -> None:
    tester = XarrayDataArrayEqualityTester()
    actual = xr.DataArray(np.arange(3), dims=["z"], coords={"z": [1.0, 2.0, 3.0]})
    expected = xr.DataArray(np.arange(3), dims=["z"], coords={"z": [1.0, 2.0, 3.05]})
    assert run_tester(tester, actual, expected, EqualityConfig(tolerance=Tolerance(atol=0.1)))


@xarray_available
def test_xarray_data_array_equality_tester_coordinates_do_not_recurse(
    config: EqualityConfig,
) -> None:
    # ``coords`` is a mapping of DataArrays, and each coordinate has its own coordinates
    # (including itself): comparing it would recurse forever. The tester compares
    # ``coords.variables``, which only contains Variables.
    data = xr.DataArray(np.arange(2), dims=["z"], coords={"z": [0, 1], "w": ("z", [3, 4])})
    assert run_tester(XarrayDataArrayEqualityTester(), data, data.copy(), config)


#############################################
#     Tests for the location of a difference #
#############################################


@xarray_available
def test_xarray_data_array_compare_path_of_the_data() -> None:
    assert compare(
        xr.DataArray(np.arange(3), dims=["z"]), xr.DataArray(np.array([0, 1, 5]), dims=["z"])
    ).path == ("variable", "data", 2)


@xarray_available
def test_xarray_data_array_compare_path_format() -> None:
    result = compare(
        xr.DataArray(np.arange(3), dims=["z"]), xr.DataArray(np.array([0, 1, 5]), dims=["z"])
    )
    assert result.format_path("actual") == "actual.variable.data[2]"


@xarray_available
def test_xarray_data_array_compare_path_of_the_name() -> None:
    assert compare(
        xr.DataArray(np.arange(3), dims=["z"], name="a"),
        xr.DataArray(np.arange(3), dims=["z"], name="b"),
    ).path == ("name",)


@xarray_available
def test_xarray_data_array_compare_path_of_the_coordinates() -> None:
    assert compare(
        xr.DataArray(np.arange(3), dims=["z"], coords={"z": [0, 1, 2]}),
        xr.DataArray(np.arange(3), dims=["z"], coords={"z": [0, 1, 3]}),
    ).path == ("coords.variables", "z", "data", 2)


@xarray_available
def test_xarray_dataset_compare_path_of_a_data_variable() -> None:
    assert compare(
        xr.Dataset({"x": ("z", np.arange(3))}), xr.Dataset({"x": ("z", np.array([0, 1, 5]))})
    ).path == ("data_vars", "x", "variable", "data", 2)


@xarray_available
def test_xarray_dataset_compare_path_of_the_attributes() -> None:
    assert compare(xr.Dataset(attrs={"a": 1}), xr.Dataset(attrs={"a": 2})).path == ("attrs", "a")


@xarray_available
def test_xarray_variable_compare_path_of_the_dimensions() -> None:
    assert compare(
        xr.Variable(dims=["z"], data=np.arange(3)), xr.Variable(dims=["y"], data=np.arange(3))
    ).path == ("dims", 0)


@xarray_available
def test_xarray_variable_compare_path_of_the_data() -> None:
    assert compare(
        xr.Variable(dims=["z"], data=np.arange(3)),
        xr.Variable(dims=["z"], data=np.array([0, 1, 5])),
    ).path == ("data", 2)
