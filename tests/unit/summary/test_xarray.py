from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    SummaryConfig,
    XarrayDataArraySummarizer,
    XarrayDatasetSummarizer,
    XarrayVariableSummarizer,
)
from coola.testing.marks import xarray_available
from coola.utils.imports import MissingPackageError, is_xarray_available

if is_xarray_available():
    import numpy as np
    import xarray as xr
else:  # pragma: no cover
    np = Mock()
    xr = Mock()


SUMMARIZERS = [
    XarrayDataArraySummarizer,
    XarrayDatasetSummarizer,
    XarrayVariableSummarizer,
]

CONFIG = SummaryConfig()
SHOW_DATA = SummaryConfig(show_data=True)


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.optional.is_xarray_available", lambda: False),
        pytest.raises(MissingPackageError, match="'xarray' package is required"),
    ):
        summarizer_cls()


@xarray_available
def test_xarray_dataarray_summarizer_summarize_dims() -> None:
    array = xr.DataArray(np.ones((2, 3)), dims=("x", "y"), name="a")
    result = XarrayDataArraySummarizer().summarize(array, CONFIG, compact=False)
    assert result.endswith(" | dims=('x', 'y') | shape=(2, 3) | dtype=float64 | name='a'")


@xarray_available
def test_xarray_dataarray_summarizer_summarize_class_name() -> None:
    array = xr.DataArray(np.ones((2, 3)), dims=("x", "y"), name="a")
    result = XarrayDataArraySummarizer().summarize(array, CONFIG, compact=False)
    assert "DataArray" in result.split(" | ")[0]


@xarray_available
def test_xarray_dataarray_summarizer_summarize_no_name() -> None:
    result = XarrayDataArraySummarizer().summarize(xr.DataArray(np.ones(2)), CONFIG, compact=False)
    assert result.endswith(" | dims=('dim_0',) | shape=(2,) | dtype=float64 | name=None")


@xarray_available
def test_xarray_dataarray_summarizer_summarize_show_data() -> None:
    array = xr.DataArray(np.ones(2))
    assert XarrayDataArraySummarizer().summarize(array, SHOW_DATA, compact=False) == repr(array)


@xarray_available
def test_xarray_dataset_summarizer_summarize_dims() -> None:
    dataset = xr.Dataset({"a": (("x",), np.ones(3)), "b": (("x",), np.zeros(3, dtype=np.int32))})
    result = XarrayDatasetSummarizer().summarize(dataset, CONFIG, compact=False)
    assert result.endswith(" | dims={'x': 3} | data_vars=a: float64, b: int32")


@xarray_available
def test_xarray_dataset_summarizer_summarize_class_name() -> None:
    dataset = xr.Dataset({"a": (("x",), np.ones(3)), "b": (("x",), np.zeros(3, dtype=np.int32))})
    result = XarrayDatasetSummarizer().summarize(dataset, CONFIG, compact=False)
    assert "Dataset" in result.split(" | ")[0]


@xarray_available
def test_xarray_dataset_summarizer_summarize_many_variables() -> None:
    dataset = xr.Dataset({f"v{i}": (("x",), np.ones(2)) for i in range(13)})
    result = XarrayDatasetSummarizer().summarize(dataset, CONFIG, compact=False)
    assert result.endswith("v4: float64, ... (+8 more)")


@xarray_available
def test_xarray_dataset_summarizer_summarize_empty() -> None:
    result = XarrayDatasetSummarizer().summarize(xr.Dataset(), CONFIG, compact=False)
    assert result.endswith(" | dims={} | data_vars=")


@xarray_available
def test_xarray_dataset_summarizer_summarize_show_data() -> None:
    dataset = xr.Dataset({"a": (("x",), np.ones(2))})
    assert XarrayDatasetSummarizer().summarize(dataset, SHOW_DATA, compact=False) == repr(dataset)


@xarray_available
def test_xarray_variable_summarizer_summarize_dims() -> None:
    variable = xr.Variable(("x", "y"), np.ones((2, 3)))
    result = XarrayVariableSummarizer().summarize(variable, CONFIG, compact=False)
    assert result.endswith(" | dims=('x', 'y') | shape=(2, 3) | dtype=float64")


@xarray_available
def test_xarray_variable_summarizer_summarize_class_name() -> None:
    variable = xr.Variable(("x", "y"), np.ones((2, 3)))
    result = XarrayVariableSummarizer().summarize(variable, CONFIG, compact=False)
    assert "Variable" in result.split(" | ")[0]


@xarray_available
def test_xarray_variable_summarizer_summarize_show_data() -> None:
    variable = xr.Variable(("x",), np.ones(2))
    assert XarrayVariableSummarizer().summarize(variable, SHOW_DATA, compact=False) == repr(
        variable
    )
