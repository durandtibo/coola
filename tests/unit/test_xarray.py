from __future__ import annotations

import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark

from coola import AllCloseTester, is_numpy_available, objects_are_allclose
from coola._xarray import (
    DataArrayAllCloseOperator,
    DataArrayEqualityOperator,
    DatasetEqualityOperator,
)
from coola.equality import EqualityTester, objects_are_equal
from coola.testing import xarray_available
from coola.utils.imports import is_xarray_available

if is_numpy_available():
    import numpy as np

if is_xarray_available():
    import xarray as xr
else:
    xr = Mock()


@xarray_available
def test_allclose_tester_registry() -> None:
    assert isinstance(AllCloseTester.registry[xr.DataArray], DataArrayAllCloseOperator)


@xarray_available
def test_objects_are_allclose_data_array() -> None:
    assert objects_are_allclose(
        xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"])
    )


@xarray_available
def test_equality_tester_registry() -> None:
    assert isinstance(EqualityTester.registry[xr.DataArray], DataArrayEqualityOperator)
    assert isinstance(EqualityTester.registry[xr.Dataset], DatasetEqualityOperator)


@xarray_available
def test_objects_are_equal_data_array() -> None:
    assert objects_are_equal(
        xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"])
    )


@xarray_available
def test_objects_are_equal_dataset() -> None:
    assert objects_are_equal(create_dataset(), create_dataset())


###############################################
#     Tests for DataArrayAllCloseOperator     #
###############################################


@xarray_available
def test_data_array_allclose_operator_str() -> None:
    assert str(DataArrayAllCloseOperator()).startswith("DataArrayAllCloseOperator(")


@xarray_available
def test_data_array_allclose_operator_equal_true() -> None:
    assert DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6)),
        xr.DataArray(np.arange(6)),
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
def test_data_array_allclose_operator_allclose_false_nan_values() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_nan_values_equal_nan() -> None:
    assert DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        equal_nan=True,
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_data_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayAllCloseOperator().allclose(
            AllCloseTester(),
            xr.DataArray(np.arange(6)),
            xr.DataArray(np.arange(6) + 1),
            show_difference=True,
        )
        assert caplog.messages[1].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_names() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), name="cat"),
        xr.DataArray(np.arange(6), name="dog"),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_names_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayAllCloseOperator().allclose(
            AllCloseTester(),
            xr.DataArray(np.arange(6), name="cat"),
            xr.DataArray(np.arange(6), name="dog"),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("Objects are different")
        assert caplog.messages[1].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_dims() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), dims=["z"]),
        xr.DataArray(np.arange(6), dims=["x"]),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_dims_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayAllCloseOperator().allclose(
            AllCloseTester(),
            xr.DataArray(np.arange(6), dims=["z"]),
            xr.DataArray(np.arange(6), dims=["x"]),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("Objects are different")
        assert caplog.messages[2].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_coords() -> None:
    print(
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}).coords
    )
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}),
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["1", "2", "3", "4", "5", "6"]}),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_coords_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayAllCloseOperator().allclose(
            AllCloseTester(),
            xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}),
            xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["1", "2", "3", "4", "5", "6"]}),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarrays are different")
        assert caplog.messages[1].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_attrs() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6), attrs={"global": "meow"}),
        xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_attrs_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayAllCloseOperator().allclose(
            AllCloseTester(),
            xr.DataArray(np.arange(6), attrs={"global": "meow"}),
            xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("Objects are different")
        assert caplog.messages[2].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_type() -> None:
    assert not DataArrayAllCloseOperator().allclose(
        AllCloseTester(),
        xr.DataArray(np.arange(6)),
        np.arange(6),
    )


@xarray_available
def test_data_array_allclose_operator_allclose_false_different_dtype_show_difference(
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


###############################################
#     Tests for DataArrayEqualityOperator     #
###############################################


@xarray_available
def test_data_array_equality_operator_str() -> None:
    assert str(DataArrayEqualityOperator()).startswith("DataArrayEqualityOperator(")


@xarray_available
def test_data_array_equality_operator_equal_true() -> None:
    assert DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6)),
        xr.DataArray(np.arange(6)),
    )


@xarray_available
def test_data_array_equality_operator_equal_true_show_difference(
    caplog: LogCaptureFixture,
) -> None:
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
def test_data_array_equality_operator_equal_false_nan_values() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
        xr.DataArray(np.array([0.0, float("nan"), 2.0])),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_data_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayEqualityOperator().equal(
            EqualityTester(),
            xr.DataArray(np.arange(6)),
            xr.DataArray(np.arange(6) + 1),
            show_difference=True,
        )
        assert caplog.messages[1].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_equality_operator_equal_false_different_names() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), name="cat"),
        xr.DataArray(np.arange(6), name="dog"),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_names_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayEqualityOperator().equal(
            EqualityTester(),
            xr.DataArray(np.arange(6), name="cat"),
            xr.DataArray(np.arange(6), name="dog"),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("Objects are different")
        assert caplog.messages[1].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_equality_operator_equal_false_different_dims() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), dims=["z"]),
        xr.DataArray(np.arange(6), dims=["x"]),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_dims_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayEqualityOperator().equal(
            EqualityTester(),
            xr.DataArray(np.arange(6), dims=["z"]),
            xr.DataArray(np.arange(6), dims=["x"]),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("Objects are different")
        assert caplog.messages[2].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_equality_operator_equal_false_different_coords() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}),
        xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["1", "2", "3", "4", "5", "6"]}),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_coords_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayEqualityOperator().equal(
            EqualityTester(),
            xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["A", "B", "C", "D", "E", "F"]}),
            xr.DataArray(np.arange(6), dims=["z"], coords={"z": ["1", "2", "3", "4", "5", "6"]}),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarrays are different")
        assert caplog.messages[1].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_equality_operator_equal_false_different_attrs() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6), attrs={"global": "meow"}),
        xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_attrs_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataArrayEqualityOperator().equal(
            EqualityTester(),
            xr.DataArray(np.arange(6), attrs={"global": "meow"}),
            xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("Objects are different")
        assert caplog.messages[2].startswith("xarray.DataArrays are different")


@xarray_available
def test_data_array_equality_operator_equal_false_different_type() -> None:
    assert not DataArrayEqualityOperator().equal(
        EqualityTester(),
        xr.DataArray(np.arange(6)),
        np.arange(6),
    )


@xarray_available
def test_data_array_equality_operator_equal_false_different_dtype_show_difference(
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


#############################################
#     Tests for DatasetEqualityOperator     #
#############################################


def create_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "x": xr.DataArray(
                np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )


@xarray_available
def test_dataset_equality_operator_str() -> None:
    assert str(DatasetEqualityOperator()).startswith("DatasetEqualityOperator(")


@xarray_available
def test_dataset_equality_operator_equal_true() -> None:
    assert DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), create_dataset())


@xarray_available
def test_dataset_equality_operator_equal_true_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert DatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), create_dataset(), show_difference=True
        )
        assert not caplog.messages


@xarray_available
def test_dataset_equality_operator_equal_false_different_data() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                np.arange(6),
                dims=["z"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    assert not DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_dataset_equality_operator_equal_false_different_data_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                np.arange(6),
                dims=["z"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    with caplog.at_level(logging.INFO):
        assert not DatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), ds, show_difference=True
        )
        assert caplog.messages[0].startswith("xarray.Datasets are different")


@xarray_available
def test_dataset_equality_operator_equal_false_different_coords() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6), "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    assert not DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_dataset_equality_operator_equal_false_different_coords_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6), "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    with caplog.at_level(logging.INFO):
        assert not DatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), ds, show_difference=True
        )
        assert caplog.messages[0].startswith("xarray.Datasets are different")


@xarray_available
def test_dataset_equality_operator_equal_false_different_attrs() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "meow"},
    )
    assert not DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_dataset_equality_operator_equal_false_different_attrs_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "meow"},
    )
    with caplog.at_level(logging.INFO):
        assert not DatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), ds, show_difference=True
        )
        assert caplog.messages[0].startswith("xarray.Datasets are different")


@xarray_available
def test_dataset_equality_operator_equal_false_different_dtype() -> None:
    assert not DatasetEqualityOperator().equal(EqualityTester(), create_dataset(), np.ones((2, 3)))


@xarray_available
def test_dataset_equality_operator_equal_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), np.ones((2, 3)), show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a xarray.Dataset:")
