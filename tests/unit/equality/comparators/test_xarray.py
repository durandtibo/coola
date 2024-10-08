from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators.xarray_ import (
    XarrayDataArrayEqualityComparator,
    XarrayDatasetEqualityComparator,
    XarrayVariableEqualityComparator,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from coola.testing import xarray_available
from coola.utils.imports import is_numpy_available, is_xarray_available
from tests.unit.equality.comparators.utils import ExamplePair

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
    return EqualityConfig(tester=EqualityTester())


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
            expected_message="objects have different variable:",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), dims=["z1"]),
            expected=xr.DataArray(np.arange(6), dims=["z2"]),
            expected_message="objects have different variable:",
        ),
        id="different dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), name="meow"),
            expected=xr.DataArray(np.arange(6), name="bear"),
            expected_message="objects have different name:",
        ),
        id="different name",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), attrs={"global": "meow"}),
            expected=xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
            expected_message="objects have different variable:",
        ),
        id="different attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), coords={"z": [1, 2, 3, 4, 5, 6]}),
            expected=xr.DataArray(np.arange(6), coords={"z": [10, 20, 30, 40, 50, 60]}),
            expected_message="objects have different _coords:",
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

XARRAY_EQUAL = XARRAY_DATA_ARRAY_EQUAL + XARRAY_DATASET_EQUAL + XARRAY_VARIABLE_EQUAL
XARRAY_NOT_EQUAL = (
    XARRAY_DATA_ARRAY_NOT_EQUAL + XARRAY_DATASET_NOT_EQUAL + XARRAY_VARIABLE_NOT_EQUAL
)
XARRAY_EQUAL_TOLERANCE = (
    XARRAY_DATA_ARRAY_EQUAL_TOLERANCE
    + XARRAY_DATASET_EQUAL_TOLERANCE
    + XARRAY_VARIABLE_EQUAL_TOLERANCE
)


#######################################################
#     Tests for XarrayDataArrayEqualityComparator     #
#######################################################


@xarray_available
def test_xarray_data_array_equality_comparator_str() -> None:
    assert str(XarrayDataArrayEqualityComparator()).startswith("XarrayDataArrayEqualityComparator(")


@xarray_available
def test_xarray_data_array_equality_comparator__eq__true() -> None:
    assert XarrayDataArrayEqualityComparator() == XarrayDataArrayEqualityComparator()


@xarray_available
def test_xarray_data_array_equality_comparator__eq__false() -> None:
    assert XarrayDataArrayEqualityComparator() != 123


@xarray_available
def test_xarray_data_array_equality_comparator_clone() -> None:
    op = XarrayDataArrayEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_xarray_data_array_equality_comparator_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    obj = xr.DataArray(np.arange(6))
    assert XarrayDataArrayEqualityComparator().equal(obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL)
def test_xarray_data_array_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL)
def test_xarray_data_array_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_NOT_EQUAL)
def test_xarray_data_array_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_NOT_EQUAL)
def test_xarray_data_array_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_data_array_equality_comparator_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        XarrayDataArrayEqualityComparator().equal(
            xr.DataArray(np.array([0.0, float("nan"), 2.0])),
            xr.DataArray(np.array([0.0, float("nan"), 2.0])),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL_TOLERANCE)
def test_xarray_data_array_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert XarrayDataArrayEqualityComparator().equal(
        actual=example.actual, expected=example.expected, config=config
    )


@xarray_available
def test_xarray_data_array_equality_comparator_no_xarray() -> None:
    with (
        patch("coola.utils.imports.is_xarray_available", lambda: False),
        pytest.raises(RuntimeError, match="'xarray' package is required but not installed."),
    ):
        XarrayDataArrayEqualityComparator()


#####################################################
#     Tests for XarrayDatasetEqualityComparator     #
#####################################################


@xarray_available
def test_xarray_dataset_equality_comparator_str() -> None:
    assert str(XarrayDatasetEqualityComparator()).startswith("XarrayDatasetEqualityComparator(")


@xarray_available
def test_xarray_dataset_equality_comparator__eq__true() -> None:
    assert XarrayDatasetEqualityComparator() == XarrayDatasetEqualityComparator()


@xarray_available
def test_xarray_dataset_equality_comparator__eq__false() -> None:
    assert XarrayDatasetEqualityComparator() != 123


@xarray_available
def test_xarray_dataset_equality_comparator_clone() -> None:
    op = XarrayDatasetEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_xarray_dataset_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])})
    assert XarrayDatasetEqualityComparator().equal(obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL)
def test_xarray_dataset_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL)
def test_xarray_dataset_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_NOT_EQUAL)
def test_xarray_dataset_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_NOT_EQUAL)
def test_xarray_dataset_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_dataset_equality_comparator_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        XarrayDatasetEqualityComparator().equal(
            xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
            xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL_TOLERANCE)
def test_xarray_dataset_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert XarrayDatasetEqualityComparator().equal(
        actual=example.actual, expected=example.expected, config=config
    )


@xarray_available
def test_xarray_dataset_equality_comparator_no_xarray() -> None:
    with (
        patch("coola.utils.imports.is_xarray_available", lambda: False),
        pytest.raises(RuntimeError, match="'xarray' package is required but not installed."),
    ):
        XarrayDatasetEqualityComparator()


######################################################
#     Tests for XarrayVariableEqualityComparator     #
######################################################


@xarray_available
def test_xarray_variable_equality_comparator_str() -> None:
    assert str(XarrayVariableEqualityComparator()).startswith("XarrayVariableEqualityComparator(")


@xarray_available
def test_xarray_variable_equality_comparator__eq__true() -> None:
    assert XarrayVariableEqualityComparator() == XarrayVariableEqualityComparator()


@xarray_available
def test_xarray_variable_equality_comparator__eq__false() -> None:
    assert XarrayVariableEqualityComparator() != 123


@xarray_available
def test_xarray_variable_equality_comparator_clone() -> None:
    op = XarrayVariableEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_xarray_variable_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = xr.Variable(dims=["z"], data=np.arange(6))
    assert XarrayVariableEqualityComparator().equal(obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL)
def test_xarray_variable_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL)
def test_xarray_variable_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_NOT_EQUAL)
def test_xarray_variable_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_NOT_EQUAL)
def test_xarray_variable_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_variable_equality_comparator_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        XarrayVariableEqualityComparator().equal(
            xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
            xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL_TOLERANCE)
def test_xarray_variable_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert XarrayVariableEqualityComparator().equal(
        actual=example.actual, expected=example.expected, config=config
    )


@xarray_available
def test_xarray_variable_equality_comparator_no_xarray() -> None:
    with (
        patch("coola.utils.imports.is_xarray_available", lambda: False),
        pytest.raises(RuntimeError, match="'xarray' package is required but not installed."),
    ):
        XarrayVariableEqualityComparator()


##########################################
#     Tests for get_mapping_equality     #
##########################################


@xarray_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        xr.DataArray: XarrayDataArrayEqualityComparator(),
        xr.Dataset: XarrayDatasetEqualityComparator(),
        xr.Variable: XarrayVariableEqualityComparator(),
    }


def test_get_type_comparator_mapping_no_xarray() -> None:
    with patch("coola.equality.comparators.xarray_.is_xarray_available", lambda: False):
        assert get_type_comparator_mapping() == {}
