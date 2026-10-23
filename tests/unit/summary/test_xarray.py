from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    SummarizerRegistry,
    XarrayDataArraySummarizer,
    XarrayDatasetSummarizer,
    XarrayVariableSummarizer,
)
from coola.testing.fixtures import xarray_available
from coola.utils.imports import MissingPackageError, is_xarray_available

if is_xarray_available():
    import numpy as np
    import xarray as xr
else:  # pragma: no cover
    np = Mock()
    xr = Mock()


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


SUMMARIZERS = [
    XarrayDataArraySummarizer,
    XarrayDatasetSummarizer,
    XarrayVariableSummarizer,
]


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_default(summarizer_cls: type) -> None:
    assert summarizer_cls()._show_data is False


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True)._show_data is True


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_default(summarizer_cls: type) -> None:
    assert repr(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_show_data(summarizer_cls: type) -> None:
    assert repr(summarizer_cls(show_data=True)) == f"{summarizer_cls.__name__}(show_data=True)"


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_str(summarizer_cls: type) -> None:
    assert str(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_default(summarizer_cls: type) -> None:
    assert summarizer_cls().equal(summarizer_cls())


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True).equal(summarizer_cls(show_data=True))


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_show_data(summarizer_cls: type) -> None:
    assert not summarizer_cls(show_data=False).equal(summarizer_cls(show_data=True))


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_type(summarizer_cls: type) -> None:
    assert not summarizer_cls().equal(42)


@xarray_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_child_class(summarizer_cls: type) -> None:
    class Child(summarizer_cls): ...

    assert not summarizer_cls().equal(Child())


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.xarray.is_xarray_available", lambda: False),
        pytest.raises(MissingPackageError, match="'xarray' package is required"),
    ):
        summarizer_cls()


@xarray_available
def test_xarray_dataarray_summarizer_summarize_dims(registry: SummarizerRegistry) -> None:
    array = xr.DataArray(np.ones((2, 3)), dims=("x", "y"), name="a")
    result = XarrayDataArraySummarizer().summarize(array, registry)
    assert result.endswith(" | dims=('x', 'y') | shape=(2, 3) | dtype=float64 | name='a'")


@xarray_available
def test_xarray_dataarray_summarizer_summarize_class_name(registry: SummarizerRegistry) -> None:
    array = xr.DataArray(np.ones((2, 3)), dims=("x", "y"), name="a")
    result = XarrayDataArraySummarizer().summarize(array, registry)
    assert "DataArray" in result.split(" | ")[0]


@xarray_available
def test_xarray_dataarray_summarizer_summarize_no_name(registry: SummarizerRegistry) -> None:
    result = XarrayDataArraySummarizer().summarize(xr.DataArray(np.ones(2)), registry)
    assert result.endswith(" | dims=('dim_0',) | shape=(2,) | dtype=float64 | name=None")


@xarray_available
def test_xarray_dataarray_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    array = xr.DataArray(np.ones(2))
    assert XarrayDataArraySummarizer(show_data=True).summarize(array, registry) == repr(array)


@xarray_available
def test_xarray_dataset_summarizer_summarize_dims(registry: SummarizerRegistry) -> None:
    dataset = xr.Dataset({"a": (("x",), np.ones(3)), "b": (("x",), np.zeros(3, dtype=np.int32))})
    result = XarrayDatasetSummarizer().summarize(dataset, registry)
    assert result.endswith(" | dims={'x': 3} | data_vars=a: float64, b: int32")


@xarray_available
def test_xarray_dataset_summarizer_summarize_class_name(registry: SummarizerRegistry) -> None:
    dataset = xr.Dataset({"a": (("x",), np.ones(3)), "b": (("x",), np.zeros(3, dtype=np.int32))})
    result = XarrayDatasetSummarizer().summarize(dataset, registry)
    assert "Dataset" in result.split(" | ")[0]


@xarray_available
def test_xarray_dataset_summarizer_summarize_many_variables(registry: SummarizerRegistry) -> None:
    dataset = xr.Dataset({f"v{i}": (("x",), np.ones(2)) for i in range(13)})
    result = XarrayDatasetSummarizer().summarize(dataset, registry)
    assert result.endswith("v9: float64, ... (+3 more)")


@xarray_available
def test_xarray_dataset_summarizer_summarize_empty(registry: SummarizerRegistry) -> None:
    result = XarrayDatasetSummarizer().summarize(xr.Dataset(), registry)
    assert result.endswith(" | dims={} | data_vars=")


@xarray_available
def test_xarray_dataset_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    dataset = xr.Dataset({"a": (("x",), np.ones(2))})
    assert XarrayDatasetSummarizer(show_data=True).summarize(dataset, registry) == repr(dataset)


@xarray_available
def test_xarray_variable_summarizer_summarize_dims(registry: SummarizerRegistry) -> None:
    variable = xr.Variable(("x", "y"), np.ones((2, 3)))
    result = XarrayVariableSummarizer().summarize(variable, registry)
    assert result.endswith(" | dims=('x', 'y') | shape=(2, 3) | dtype=float64")


@xarray_available
def test_xarray_variable_summarizer_summarize_class_name(registry: SummarizerRegistry) -> None:
    variable = xr.Variable(("x", "y"), np.ones((2, 3)))
    result = XarrayVariableSummarizer().summarize(variable, registry)
    assert "Variable" in result.split(" | ")[0]


@xarray_available
def test_xarray_variable_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    variable = xr.Variable(("x",), np.ones(2))
    assert XarrayVariableSummarizer(show_data=True).summarize(variable, registry) == repr(variable)


@xarray_available
def test_dataset_summarizer_max_fields(registry: SummarizerRegistry) -> None:
    dataset = xr.Dataset({"a": ("x", [1]), "b": ("x", [2.0]), "c": ("x", [3])})
    out = XarrayDatasetSummarizer(max_fields=1).summarize(dataset, registry)
    assert out.endswith("data_vars=a: int64, ... (+2 more)")
