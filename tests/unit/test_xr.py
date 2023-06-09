from __future__ import annotations

import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture

from coola import is_numpy_available
from coola.equality import EqualityTester, objects_are_equal
from coola.testing import xarray_available
from coola.utils.imports import is_xarray_available
from coola.xr import XarrayDatasetEqualityOperator

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_xarray_available():
    import xarray as xr
else:
    xarray = Mock()


@xarray_available
def test_equality_tester_registry_xarray() -> None:
    assert isinstance(EqualityTester.registry[xr.Dataset], XarrayDatasetEqualityOperator)


@xarray_available
def test_objects_are_equal_xarray() -> None:
    assert objects_are_equal(create_dataset(), create_dataset())


#############################################
#     Tests for NDArrayEqualityOperator     #
#############################################


def create_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "x": xr.DataArray(
                data=np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                data=np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )


@xarray_available
def test_xarray_dataset_equality_operator_str() -> None:
    assert str(XarrayDatasetEqualityOperator()).startswith("XarrayDatasetEqualityOperator(")


@xarray_available
def test_xarray_dataset_equality_operator_equal_true() -> None:
    assert XarrayDatasetEqualityOperator().equal(
        EqualityTester(), create_dataset(), create_dataset()
    )


@xarray_available
def test_xarray_dataset_equality_operator_equal_true_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert XarrayDatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), create_dataset()
        )
        assert not caplog.messages


@xarray_available
def test_xarray_dataset_equality_operator_equal_false_different_data() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                data=np.arange(6),
                dims=["z"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    assert not XarrayDatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_xarray_dataset_equality_operator_equal_false_different_data_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                data=np.arange(6),
                dims=["z"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    with caplog.at_level(logging.INFO):
        assert not XarrayDatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), ds, show_difference=True
        )
        assert caplog.messages[0].startswith("xarray.ndarrays are different")


@xarray_available
def test_xarray_dataset_equality_operator_equal_false_different_coords() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                data=np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                data=np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6), "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    assert not XarrayDatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_xarray_dataset_equality_operator_equal_false_different_coords_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                data=np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                data=np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6), "t": ["t1", "t2", "t3"]},
        attrs={"global": "this is a global attribute"},
    )
    with caplog.at_level(logging.INFO):
        assert not XarrayDatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), ds, show_difference=True
        )
        assert caplog.messages[0].startswith("xarray.ndarrays are different")


@xarray_available
def test_xarray_dataset_equality_operator_equal_false_different_attrs() -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                data=np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                data=np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "meow"},
    )
    assert not XarrayDatasetEqualityOperator().equal(EqualityTester(), create_dataset(), ds)


@xarray_available
def test_xarray_dataset_equality_operator_equal_false_different_attrs_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                data=np.arange(6),
                dims=["z"],
            ),
            "y": xr.DataArray(
                data=np.ones((6, 3)),
                dims=["z", "t"],
            ),
        },
        coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
        attrs={"global": "meow"},
    )
    with caplog.at_level(logging.INFO):
        assert not XarrayDatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), ds, show_difference=True
        )
        assert caplog.messages[0].startswith("xarray.ndarrays are different")


@xarray_available
def test_xarray_dataset_equality_operator_equal_false_different_dtype() -> None:
    assert not XarrayDatasetEqualityOperator().equal(
        EqualityTester(), create_dataset(), np.ones((2, 3))
    )


@xarray_available
def test_xarray_dataset_equality_operator_equal_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not XarrayDatasetEqualityOperator().equal(
            EqualityTester(), create_dataset(), np.ones((2, 3)), show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a xarray.Dataset:")
