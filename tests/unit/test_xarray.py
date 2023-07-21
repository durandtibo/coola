from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark, raises

from coola import AllCloseTester, is_numpy_available, objects_are_allclose
from coola.equality import EqualityTester, objects_are_equal
from coola.testing import xarray_available
from coola.utils.imports import is_xarray_available
from coola.xarray_ import (
    DataArrayAllCloseOperator,
    DataArrayEqualityOperator,
    DatasetAllCloseOperator,
    DatasetEqualityOperator,
    VariableAllCloseOperator,
    VariableEqualityOperator,
)

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_xarray_available():
    import xarray as xr
else:
    xr = Mock()


@xarray_available
def test_allclose_tester_registry() -> None:
    assert isinstance(AllCloseTester.registry[xr.DataArray], DataArrayAllCloseOperator)


@xarray_available
def test_equality_tester_registry() -> None:
    assert isinstance(EqualityTester.registry[xr.DataArray], DataArrayEqualityOperator)
    assert isinstance(EqualityTester.registry[xr.Dataset], DatasetEqualityOperator)
    assert isinstance(EqualityTester.registry[xr.Variable], VariableEqualityOperator)


###############################################
#     Tests for DataArrayAllCloseOperator     #
###############################################


@xarray_available
def test_objects_are_allclose_data_array() -> None:
    assert objects_are_allclose(
        xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"])
    )


@xarray_available
def test_data_array_allclose_operator_str() -> None:
    assert str(DataArrayAllCloseOperator()).startswith("DataArrayAllCloseOperator(")


@xarray_available
def test_data_array_allclose_operator__eq__true() -> None:
    assert DataArrayAllCloseOperator() == DataArrayAllCloseOperator()


@xarray_available
def test_data_array_allclose_operator__eq__false() -> None:
    assert DataArrayAllCloseOperator() != 123


@xarray_available
def test_data_array_allclose_operator_equal_true() -> None:
    assert DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6)),
        xr.DataArray(np.arange(6)),
    )


@xarray_available
def test_data_array_allclose_operator_equal_true_same_object() -> None:
    obj = xr.DataArray(np.arange(6))
    assert DataArrayAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@xarray_available
def test_data_array_allclose_operator_equal_true_coords_numerical() -> None:
    assert DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
    )


@xarray_available
def test_data_array_allclose_operator_equal_true_coords_str() -> None:
    assert DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": [str(i) for i in range(6)]}),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": [str(i) for i in range(6)]}),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_true_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert DataArrayAllCloseOperator().allclose(
            AllCloseTester(),
            xr.DataArray(np.arange(6)),
            xr.DataArray(np.arange(6)),
            show_difference=True,
        )
        assert not caplog.messages


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_data() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6)),
        xr.DataArray(np.arange(6) + 1),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_nan() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_true_nan() -> None:
    assert DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        equal_nan=True,
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_names() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), name="cat"),
        xr.DataArray(np.arange(6), name="dog"),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_dims() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), dims=["z"]),
        xr.DataArray(np.arange(6), dims=["x"]),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_coords() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}),
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["1", "2", "3", "4", "5", "6"]}),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_attrs() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), attrs={"global": "meow"}),
        xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayAllCloseOperator().allclose(
            AllCloseTester(),
            xr.DataArray(np.arange(6)),
            xr.DataArray(np.arange(6) + 1),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_type() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6)),
        np.arange(6),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayAllCloseOperator().allclose(
            AllCloseTester(),
            xr.DataArray(np.arange(6)),
            np.arange(6),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a xarray.DataArray:")


@xarray_available
@mark.parametrize(
    "array,atol",
    (
        (xr.DataArray(np.full((2, 3), 1.5)), 1),
        (xr.DataArray(np.full((2, 3), 1.05)), 1e-1),
        (xr.DataArray(np.full((2, 3), 1.005)), 1e-2),
    ),
)
def test_data_array_allclose_operator_allclose_true_atol(array: xr.DataArray, atol: float) -> None:
    assert DataArrayAllCloseOperator().allclose(
        AllCloseTester(), xr.DataArray(np.ones((2, 3))), array, atol=atol, rtol=0
    )


@xarray_available
@mark.parametrize(
    "array,rtol",
    (
        (xr.DataArray(np.full((2, 3), 1.5)), 1),
        (xr.DataArray(np.full((2, 3), 1.05)), 1e-1),
        (xr.DataArray(np.full((2, 3), 1.005)), 1e-2),
    ),
)
def test_data_array_allclose_operator_allclose_true_rtol(array: xr.DataArray, rtol: float) -> None:
    assert DataArrayAllCloseOperator().allclose(
        AllCloseTester(), xr.DataArray(np.ones((2, 3))), array, rtol=rtol
    )


@xarray_available
def test_data_array_allclose_operator_clone() -> None:
    op = DataArrayAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_data_array_allclose_operator_no_xarray() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`xarray` package is required but not installed."):
            DataArrayAllCloseOperator()


###############################################
#     Tests for DataArrayEqualityOperator     #
###############################################


@xarray_available
def test_objects_are_equal_data_array() -> None:
    assert objects_are_equal(
        xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"])
    )


@xarray_available
def test_data_array_equality_operator_str() -> None:
    assert str(DataArrayEqualityOperator()).startswith("DataArrayEqualityOperator(")


@xarray_available
def test_data_array_equality_operator__eq__true() -> None:
    assert DataArrayEqualityOperator() == DataArrayEqualityOperator()


@xarray_available
def test_data_array_equality_operator__eq__false() -> None:
    assert DataArrayEqualityOperator() != 123


@xarray_available
def test_data_array_equality_operator_clone() -> None:
    op = DataArrayEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_data_array_equality_operator_equal_true() -> None:
    assert DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6)),
        xr.DataArray(np.arange(6)),
    )


@xarray_available
def test_data_array_equality_operator_equal_true_coords_numerical() -> None:
    assert DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
    )


@xarray_available
def test_data_array_equality_operator_equal_true_coords_str() -> None:
    assert DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": [str(i) for i in range(6)]}),
        xr.DataArray(np.arange(6), dims=["x"], coords={"x": [str(i) for i in range(6)]}),
    )


@xarray_available
def test_data_array_equality_operator_equal_true_same_object() -> None:
    obj = xr.DataArray(np.arange(6))
    assert DataArrayEqualityOperator().equal(EqualityTester(), obj, obj)


@xarray_available
def test_data_array_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert DataArrayEqualityOperator().equal(
            EqualityTester(),
            xr.DataArray(np.arange(6)),
            xr.DataArray(np.arange(6)),
            show_difference=True,
        )
        assert not caplog.messages


@xarray_available
def test_data_array_equality_operator_equal_false_different_data() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6)),
        xr.DataArray(np.arange(6) + 1),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_nan() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_names() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), name="cat"),
        xr.DataArray(np.arange(6), name="dog"),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_dims() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), dims=["z"]),
        xr.DataArray(np.arange(6), dims=["x"]),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_coords() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}),
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["1", "2", "3", "4", "5", "6"]}),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_attrs() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), attrs={"global": "meow"}),
        xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayEqualityOperator().equal(
            EqualityTester(),
            xr.DataArray(np.arange(6)),
            xr.DataArray(np.arange(6) + 1),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_equality_operator_equal_false_different_type() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6)),
        np.arange(6),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayEqualityOperator().equal(
            EqualityTester(),
            xr.DataArray(np.arange(6)),
            np.arange(6),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a xarray.DataArray:")


@xarray_available
def test_data_array_equality_operator_no_xarray() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`xarray` package is required but not installed."):
            DataArrayEqualityOperator()


#############################################
#     Tests for DatasetAllCloseOperator     #
#############################################


def create_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
            "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )


@xarray_available
def test_objects_are_allclose_dataset() -> None:
    assert objects_are_allclose(create_dataset(), create_dataset())


@xarray_available
def test_dataset_allclose_operator_str() -> None:
    assert str(DatasetAllCloseOperator()).startswith("DatasetAllCloseOperator(")


@xarray_available
def test_dataset_allclose_operator__eq__true() -> None:
    assert DatasetAllCloseOperator() == DatasetAllCloseOperator()


@xarray_available
def test_dataset_allclose_operator__eq__false() -> None:
    assert DatasetAllCloseOperator() != 123


@xarray_available
def test_dataset_allclose_operator_equal_true() -> None:
    assert DatasetAllCloseOperator().allclose(AllCloseTester(), create_dataset(), create_dataset())


@xarray_available
def test_dataset_allclose_operator_allclose_true_same_object() -> None:
    obj = create_dataset()
    assert DatasetAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@xarray_available
def test_dataset_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert DatasetAllCloseOperator().allclose(
            AllCloseTester(), create_dataset(), create_dataset(), show_difference=True
        )
        assert not caplog.messages


@xarray_available
def test_dataset_allclose_operator_allclose_false_different_data() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    assert not DatasetAllCloseOperator().allclose(AllCloseTester(), create_dataset(), ds)


@xarray_available
def test_dataset_allclose_operator_allclose_false_nan() -> None:
    assert not DatasetAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Dataset({"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
        xr.Dataset({"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
    )


@xarray_available
def test_dataset_allclose_operator_allclose_true_nan() -> None:
    assert DatasetAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Dataset({"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
        xr.Dataset({"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
        equal_nan=True,
    )


@xarray_available
def test_dataset_allclose_operator_allclose_false_different_coords() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
            "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
        },
        coords={"z": np.arange(6), "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    assert not DatasetAllCloseOperator().allclose(AllCloseTester(), create_dataset(), ds)


@xarray_available
def test_dataset_allclose_operator_allclose_false_different_attrs() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
            "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "meow"},
    )
    assert not DatasetAllCloseOperator().allclose(AllCloseTester(), create_dataset(), ds)


@xarray_available
def test_dataset_allclose_operator_allclose_false_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    with caplog.at_level(logging.INFO):
        assert not DatasetAllCloseOperator().allclose(
            AllCloseTester(), create_dataset(), ds, show_difference=True
        )
        assert caplog.messages[-1].startswith("xarray.Datasets are different")


@xarray_available
def test_dataset_allclose_operator_allclose_false_different_type() -> None:
    assert not DatasetAllCloseOperator().allclose(
        AllCloseTester(), create_dataset(), np.ones((2, 3))
    )


@xarray_available
def test_dataset_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DatasetAllCloseOperator().allclose(
            AllCloseTester(), create_dataset(), np.ones((2, 3)), show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a xarray.Dataset:")


@xarray_available
@mark.parametrize(
    "dataset,atol",
    (
        (xr.Dataset({"x": xr.DataArray(np.full((2, 3), 1.5))}), 1),
        (xr.Dataset({"x": xr.DataArray(np.full((2, 3), 1.05))}), 1e-1),
        (xr.Dataset({"x": xr.DataArray(np.full((2, 3), 1.005))}), 1e-2),
    ),
)
def test_dataset_allclose_operator_allclose_true_atol(dataset: xr.Dataset, atol: float) -> None:
    assert DatasetAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Dataset({"x": xr.DataArray(np.ones((2, 3)))}),
        dataset,
        atol=atol,
        rtol=0,
    )


@xarray_available
@mark.parametrize(
    "dataset,rtol",
    (
        (xr.Dataset({"x": xr.DataArray(np.full((2, 3), 1.5))}), 1),
        (xr.Dataset({"x": xr.DataArray(np.full((2, 3), 1.05))}), 1e-1),
        (xr.Dataset({"x": xr.DataArray(np.full((2, 3), 1.005))}), 1e-2),
    ),
)
def test_dataset_allclose_operator_allclose_true_rtol(dataset: xr.Dataset, rtol: float) -> None:
    assert DatasetAllCloseOperator().allclose(
        AllCloseTester(), xr.Dataset({"x": xr.DataArray(np.ones((2, 3)))}), dataset, rtol=rtol
    )


@xarray_available
def test_dataset_allclose_operator_clone() -> None:
    op = DatasetAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_dataset_allclose_operator_no_xarray() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`xarray` package is required but not installed."):
            DatasetAllCloseOperator()


#############################################
#     Tests for DatasetEqualityOperator     #
#############################################


@xarray_available
def test_objects_are_equal_dataset() -> None:
    assert objects_are_equal(create_dataset(), create_dataset())


@xarray_available
def test_dataset_equality_operator_str() -> None:
    assert str(DatasetEqualityOperator()).startswith("DatasetEqualityOperator(")


@xarray_available
def test_dataset_equality_operator__eq__true() -> None:
    assert DatasetEqualityOperator() == DatasetEqualityOperator()


@xarray_available
def test_dataset_equality_operator__eq__false() -> None:
    assert DatasetEqualityOperator() != 123


@xarray_available
def test_dataset_equality_operator_clone() -> None:
    op = DatasetEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_dataset_equality_operator_equal_true() -> None:
    assert DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), create_dataset())


@xarray_available
def test_dataset_equality_operator_equal_true_same_object() -> None:
    obj = create_dataset()
    assert DatasetEqualityOperator().equal(EqualityTester(), obj, obj)


@xarray_available
def test_dataset_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert DatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), create_dataset(), show_difference=True
        )
        assert not caplog.messages


@xarray_available
def test_dataset_equality_operator_equal_false_different_data() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    assert not DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_dataset_equality_operator_equal_false_nan_values() -> None:
    assert not DatasetEqualityOperator().equal(
        EqualityTester(),
        xr.Dataset({"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
        xr.Dataset({"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
    )


@xarray_available
def test_dataset_equality_operator_equal_false_different_coords() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
            "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
        },
        coords={"z": np.arange(6), "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    assert not DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_dataset_equality_operator_equal_false_different_attrs() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
            "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "meow"},
    )
    assert not DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_dataset_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture) -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    with caplog.at_level(logging.INFO):
        assert not DatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), ds, show_difference=True
        )
        assert caplog.messages[-1].startswith("xarray.Datasets are different")


@xarray_available
def test_dataset_equality_operator_equal_false_different_type() -> None:
    assert not DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), np.ones((2, 3)))


@xarray_available
def test_dataset_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), np.ones((2, 3)), show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a xarray.Dataset:")


@xarray_available
def test_dataset_equality_operator_no_xarray() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`xarray` package is required but not installed."):
            DatasetEqualityOperator()


##############################################
#     Tests for VariableAllCloseOperator     #
##############################################


@xarray_available
def test_objects_are_allclose_variable() -> None:
    assert objects_are_allclose(
        xr.Variable(dims=["z"], data=np.arange(6)), xr.Variable(dims=["z"], data=np.arange(6))
    )


@xarray_available
def test_variable_allclose_operator_str() -> None:
    assert str(VariableAllCloseOperator()).startswith("VariableAllCloseOperator(")


@xarray_available
def test_variable_allclose_operator__eq__true() -> None:
    assert VariableAllCloseOperator() == VariableAllCloseOperator()


@xarray_available
def test_variable_allclose_operator__eq__false() -> None:
    assert VariableAllCloseOperator() != 123


@xarray_available
def test_variable_allclose_operator_equal_true() -> None:
    assert VariableAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["z"], data=np.arange(6)),
    )


@xarray_available
def test_variable_allclose_operator_allclose_true_same_object() -> None:
    obj = xr.Variable(dims=["z"], data=np.arange(6))
    assert VariableAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@xarray_available
def test_variable_allclose_operator_allclose_true_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert VariableAllCloseOperator().allclose(
            AllCloseTester(),
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["z"], data=np.arange(6)),
            show_difference=True,
        )
        assert not caplog.messages


@xarray_available
def test_variable_allclose_operator_allclose_false_different_data() -> None:
    assert not VariableAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["z"], data=np.arange(6) + 1),
    )


@xarray_available
def test_variable_allclose_operator_allclose_false_nan() -> None:
    assert not VariableAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
    )


@xarray_available
def test_variable_allclose_operator_allclose_true_nan() -> None:
    assert VariableAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        equal_nan=True,
    )


@xarray_available
def test_variable_allclose_operator_allclose_false_different_dims() -> None:
    assert not VariableAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["x"], data=np.arange(6)),
    )


@xarray_available
def test_variable_allclose_operator_allclose_false_different_attrs() -> None:
    assert not VariableAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
        xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
    )


@xarray_available
def test_variable_allclose_operator_allclose_false_different_data_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not VariableAllCloseOperator().allclose(
            AllCloseTester(),
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["z"], data=np.arange(6) + 1),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("xarray.Variables are different")


@xarray_available
def test_variable_allclose_operator_allclose_false_different_type() -> None:
    assert not VariableAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Variable(dims=["z"], data=np.arange(6)),
        np.arange(6),
    )


@xarray_available
def test_variable_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not VariableAllCloseOperator().allclose(
            AllCloseTester(),
            xr.Variable(dims=["z"], data=np.arange(6)),
            np.arange(6),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a xarray.Variable:")


@xarray_available
@mark.parametrize(
    "array,atol",
    (
        (xr.Variable(dims=["x", "y"], data=np.full((2, 3), 1.5)), 1),
        (xr.Variable(dims=["x", "y"], data=np.full((2, 3), 1.05)), 1e-1),
        (xr.Variable(dims=["x", "y"], data=np.full((2, 3), 1.005)), 1e-2),
    ),
)
def test_variable_allclose_operator_allclose_true_atol(array: xr.DataArray, atol: float) -> None:
    assert VariableAllCloseOperator().allclose(
        AllCloseTester(),
        xr.Variable(dims=["x", "y"], data=np.ones((2, 3))),
        array,
        atol=atol,
        rtol=0,
    )


@xarray_available
@mark.parametrize(
    "array,rtol",
    (
        (xr.Variable(dims=["x", "y"], data=np.full((2, 3), 1.5)), 1),
        (xr.Variable(dims=["x", "y"], data=np.full((2, 3), 1.05)), 1e-1),
        (xr.Variable(dims=["x", "y"], data=np.full((2, 3), 1.005)), 1e-2),
    ),
)
def test_variable_allclose_operator_allclose_true_rtol(array: xr.Variable, rtol: float) -> None:
    assert VariableAllCloseOperator().allclose(
        AllCloseTester(), xr.Variable(dims=["x", "y"], data=np.ones((2, 3))), array, rtol=rtol
    )


@xarray_available
def test_variable_allclose_operator_clone() -> None:
    op = VariableAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_variable_allclose_operator_no_xarray() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`xarray` package is required but not installed."):
            VariableAllCloseOperator()


##############################################
#     Tests for VariableEqualityOperator     #
##############################################


@xarray_available
def test_objects_are_equal_variable() -> None:
    assert objects_are_equal(
        xr.Variable(dims=["z"], data=np.arange(6)), xr.Variable(dims=["z"], data=np.arange(6))
    )


@xarray_available
def test_variable_equality_operator_str() -> None:
    assert str(VariableEqualityOperator()).startswith("VariableEqualityOperator(")


@xarray_available
def test_variable_equality_operator__eq__true() -> None:
    assert VariableEqualityOperator() == VariableEqualityOperator()


@xarray_available
def test_variable_equality_operator__eq__false() -> None:
    assert VariableEqualityOperator() != 123


@xarray_available
def test_variable_equality_operator_clone() -> None:
    op = VariableEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_variable_equality_operator_equal_true() -> None:
    assert VariableEqualityOperator().equal(
        EqualityTester(),
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["z"], data=np.arange(6)),
    )


@xarray_available
def test_variable_equality_operator_equal_true_same_object() -> None:
    obj = xr.Variable(dims=["z"], data=np.arange(6))
    assert VariableEqualityOperator().equal(EqualityTester(), obj, obj)


@xarray_available
def test_variable_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert VariableEqualityOperator().equal(
            EqualityTester(),
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["z"], data=np.arange(6)),
            show_difference=True,
        )
        assert not caplog.messages


@xarray_available
def test_variable_equality_operator_equal_false_different_data() -> None:
    assert not VariableEqualityOperator().equal(
        EqualityTester(),
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["z"], data=np.arange(6) + 1),
    )


@xarray_available
def test_variable_equality_operator_equal_false_nan() -> None:
    assert not VariableEqualityOperator().equal(
        EqualityTester(),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
    )


@xarray_available
def test_variable_equality_operator_equal_false_different_dims() -> None:
    assert not VariableEqualityOperator().equal(
        EqualityTester(),
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["x"], data=np.arange(6)),
    )


@xarray_available
def test_variable_equality_operator_equal_false_different_attrs() -> None:
    assert not VariableEqualityOperator().equal(
        EqualityTester(),
        xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
        xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
    )


@xarray_available
def test_variable_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not VariableEqualityOperator().equal(
            EqualityTester(),
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["z"], data=np.arange(6) + 1),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("xarray.Variables are different")


@xarray_available
def test_variable_equality_operator_equal_false_different_type() -> None:
    assert not VariableEqualityOperator().equal(
        EqualityTester(),
        xr.Variable(dims=["z"], data=np.arange(6)),
        np.arange(6),
    )


@xarray_available
def test_variable_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not VariableEqualityOperator().equal(
            EqualityTester(),
            xr.Variable(dims=["z"], data=np.arange(6)),
            np.arange(6),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a xarray.Variable:")


@xarray_available
def test_variable_equality_operator_no_xarray() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`xarray` package is required but not installed."):
            VariableEqualityOperator()
