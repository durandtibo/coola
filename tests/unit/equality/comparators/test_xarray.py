from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.comparators.xarray_ import (
    XarrayDataArrayEqualityComparator,
    XarrayDatasetEqualityComparator,
    XarrayVariableEqualityComparator,
    get_type_comparator_mapping,
)
from coola.testers import EqualityTester
from coola.testing import xarray_available
from coola.utils.imports import is_numpy_available, is_xarray_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_xarray_available():
    import xarray as xr
else:
    xr = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#######################################################
#     Tests for XarrayDataArrayEqualityComparator     #
#######################################################


@xarray_available
def test_objects_are_equal_data_array() -> None:
    assert objects_are_equal(
        xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"])
    )


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
def test_xarray_data_array_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(xr.DataArray(np.arange(6)), xr.DataArray(np.arange(6)), config)
        assert not caplog.messages


@xarray_available
def test_xarray_data_array_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            xr.DataArray(np.arange(6)),
            xr.DataArray(np.arange(6)),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_data_array_equality_comparator_equal_true_coords_numerical(
    config: EqualityConfig,
) -> None:
    assert XarrayDataArrayEqualityComparator().equal(
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
        config,
    )


@xarray_available
def test_xarray_data_array_equality_comparator_equal_true_coords_str(
    config: EqualityConfig,
) -> None:
    assert XarrayDataArrayEqualityComparator().equal(
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": [str(i) for i in range(6)]}),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": [str(i) for i in range(6)]}),
        config,
    )


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_variable(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6)), xr.DataArray(np.arange(6) + 1), config
        )
        assert not caplog.messages


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_variable_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6)), xr.DataArray(np.arange(6) + 1), config
        )
        assert caplog.messages[-1].startswith("objects have different variable:")


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_name(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6), name="cat"), xr.DataArray(np.arange(6), name="dog"), config
        )
        assert not caplog.messages


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_name_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6), name="cat"), xr.DataArray(np.arange(6), name="dog"), config
        )
        assert caplog.messages[-1].startswith("objects have different name:")


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_dims(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["x"]), config
        )
        assert not caplog.messages


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_dims_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["x"]), config
        )
        assert caplog.messages[-1].startswith("objects have different variable:")


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_attrs(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6), attrs={"global": "meow"}),
            xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_attrs_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6), attrs={"global": "meow"}),
            xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different variable:")


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_coords(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}),
            xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["1", "2", "3", "4", "5", "6"]}),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_coords_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}),
            xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["1", "2", "3", "4", "5", "6"]}),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different _coords:")


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.DataArray(np.arange(6)), xr.DataArray(np.arange(6) + 1), config
        )
        assert caplog.messages[-1].startswith("objects have different variable:")


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(xr.DataArray(np.arange(6)), np.arange(6), config)
        assert not caplog.messages


@xarray_available
def test_xarray_data_array_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDataArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(xr.DataArray(np.arange(6)), np.arange(6), config)
        assert caplog.messages[0].startswith("objects have different types:")


@xarray_available
def test_xarray_data_array_equality_comparator_no_xarray() -> None:
    with patch(
        "coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`xarray` package is required but not installed."):
        XarrayDataArrayEqualityComparator()


@xarray_available
def test_xarray_data_array_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not XarrayDataArrayEqualityComparator().equal(
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        config,
    )


@xarray_available
def test_xarray_data_array_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    # TODO(TIBO): update after the new version is finished  # noqa: TD003
    assert not XarrayDataArrayEqualityComparator().equal(
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        config,
    )


#####################################################
#     Tests for XarrayDatasetEqualityComparator     #
#####################################################


@xarray_available
def test_objects_are_equal_dataset() -> None:
    assert objects_are_equal(
        xr.Dataset(
            data_vars={
                "x": xr.DataArray(np.arange(6), dims=["z"]),
                "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
            },
            coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
            attrs={"global": "this is a global attribute"},
        ),
        xr.Dataset(
            data_vars={
                "x": xr.DataArray(np.arange(6), dims=["z"]),
                "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
            },
            coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
            attrs={"global": "this is a global attribute"},
        ),
    )


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
def test_xarray_dataset_equality_comparator_equal_true(config: EqualityConfig) -> None:
    assert XarrayDatasetEqualityComparator().equal(
        xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
        xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
        config,
    )


@xarray_available
def test_xarray_dataset_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
            xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_dataset_equality_comparator_equal_false_data_vars(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
            xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6) + 1, dims=["z"])}),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_dataset_equality_comparator_equal_false_data_vars_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6) + 1, dims=["z"])}),
            xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different data_vars:")


@xarray_available
def test_xarray_dataset_equality_comparator_equal_false_coords(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Dataset(data_vars={}, coords={"z": np.arange(6)}),
            xr.Dataset(data_vars={}, coords={"z": np.arange(6) + 1}),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_dataset_equality_comparator_equal_false_coords_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Dataset(data_vars={}, coords={"z": np.arange(6)}),
            xr.Dataset(data_vars={}, coords={"z": np.arange(6) + 1}),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different coords:")


@xarray_available
def test_xarray_dataset_equality_comparator_equal_false_different_attrs(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Dataset(data_vars={}, coords={}, attrs={"global": "meow"}),
            xr.Dataset(data_vars={}, coords={}, attrs={"global": "meowwww"}),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_dataset_equality_comparator_equal_false_attrs_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Dataset(data_vars={}, coords={}, attrs={"global": "meow"}),
            xr.Dataset(data_vars={}, coords={}, attrs={"global": "meowwww"}),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different attrs:")


@xarray_available
def test_xarray_dataset_equality_comparator_equal_false_different_type(
    config: EqualityConfig,
) -> None:
    assert not XarrayDatasetEqualityComparator().equal(
        xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}), np.ones((2, 3)), config
    )


@xarray_available
def test_xarray_dataset_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayDatasetEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
            np.ones((2, 3)),
            config,
        )
        assert caplog.messages[0].startswith("objects have different types:")


@xarray_available
def test_xarray_dataset_equality_comparator_no_xarray() -> None:
    with patch(
        "coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`xarray` package is required but not installed."):
        XarrayDatasetEqualityComparator()


@xarray_available
def test_xarray_dataset_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not XarrayDatasetEqualityComparator().equal(
        xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
        xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
        config,
    )


@xarray_available
def test_xarray_dataset_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    # TODO(TIBO): update after the new version is finished  # noqa: TD003
    assert not XarrayDatasetEqualityComparator().equal(
        xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
        xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
        config,
    )


######################################################
#     Tests for XarrayVariableEqualityComparator     #
######################################################


@xarray_available
def test_objects_are_equal_variable() -> None:
    assert objects_are_equal(
        xr.Variable(dims=["z"], data=np.arange(6)), xr.Variable(dims=["z"], data=np.arange(6))
    )


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
def test_xarray_variable_equality_comparator_equal_true(config: EqualityConfig) -> None:
    assert XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["z"], data=np.arange(6)),
        config,
    )


@xarray_available
def test_xarray_variable_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = xr.Variable(dims=["z"], data=np.arange(6))
    assert XarrayVariableEqualityComparator().equal(obj, obj, config)


@xarray_available
def test_xarray_variable_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["z"], data=np.arange(6)),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_variable_equality_comparator_equal_false_data(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.ones(6)),
            xr.Variable(dims=["z"], data=np.zeros(6)),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_variable_equality_comparator_equal_false_data_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.ones(6)),
            xr.Variable(dims=["z"], data=np.zeros(6)),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different data:")


@xarray_available
def test_xarray_variable_equality_comparator_equal_false_dims(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["x"], data=np.arange(6)),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_variable_equality_comparator_equal_false_dims_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["x"], data=np.arange(6)),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different dims:")


@xarray_available
def test_xarray_variable_equality_comparator_equal_false_different_attrs(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
            xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_xarray_variable_equality_comparator_equal_false_attrs_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
            xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different attrs:")


@xarray_available
def test_xarray_variable_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6)), np.arange(6), config
        )
        assert not caplog.messages


@xarray_available
def test_xarray_variable_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6)), np.arange(6), config
        )
        assert caplog.messages[0].startswith("objects have different types:")


@xarray_available
def test_xarray_variable_equality_comparator_no_xarray() -> None:
    with patch(
        "coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`xarray` package is required but not installed."):
        XarrayVariableEqualityComparator()


@xarray_available
def test_xarray_variable_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        config,
    )


@xarray_available
def test_xarray_variable_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    # TODO(TIBO): update after the new version is finished  # noqa: TD003
    assert not XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        config,
    )


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
    with patch(
        "coola.equality.comparators.xarray_.is_xarray_available", lambda *args, **kwargs: False
    ):
        assert get_type_comparator_mapping() == {}
