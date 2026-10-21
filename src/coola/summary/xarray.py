r"""Implement summarizers for xarray objects.

The summarizers show metadata (dimensions, shape, dtype) instead of the
values by default.
"""

from __future__ import annotations

__all__ = ["XarrayDataArraySummarizer", "XarrayDatasetSummarizer", "XarrayVariableSummarizer"]

from typing import TYPE_CHECKING

from coola.summary.metadata import BaseFieldsSummarizer, BaseMetadataSummarizer
from coola.utils.imports import check_xarray, is_xarray_available

if TYPE_CHECKING or is_xarray_available():
    import xarray as xr
else:  # pragma: no cover
    from coola.utils.fallback.xarray import xarray as xr


class XarrayDataArraySummarizer(BaseMetadataSummarizer[xr.DataArray]):
    r"""Implement a summarizer for ``xarray.DataArray`` objects.

    By default, it displays the metadata (type, dimensions, shape,
    dtype, name) rather than the values.

    Args:
        show_data: If ``True``, returns ``repr(dataarray)``, displaying
            the values. If ``False`` (default), returns only the
            metadata.

    Raises:
        MissingPackageError: If xarray is not installed.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.summary import SummarizerRegistry, XarrayDataArraySummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = XarrayDataArraySummarizer()
        >>> array = xr.DataArray(np.ones((2, 3)), dims=("x", "y"), name="a")
        >>> print(summarizer.summarize(array, registry))  # doctest: +ELLIPSIS
        <class '...'> | dims=('x', 'y') | shape=(2, 3) | dtype=float64 | name='a'

        ```
    """

    def _check_dependencies(self) -> None:
        check_xarray()

    def _describe(self, data: xr.DataArray) -> list[str]:
        return [
            f"dims={data.dims}",
            f"shape={data.shape}",
            f"dtype={data.dtype}",
            f"name={data.name!r}",
        ]


class XarrayDatasetSummarizer(BaseFieldsSummarizer[xr.Dataset]):
    r"""Implement a summarizer for ``xarray.Dataset`` objects.

    By default, it displays the metadata (type, dimensions with their
    size, data variables with their dtype) rather than the values. At
    most 10 data variables are listed.

    Args:
        show_data: If ``True``, returns ``repr(dataset)``, displaying
            the values. If ``False`` (default), returns only the
            metadata.

    Raises:
        MissingPackageError: If xarray is not installed.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.summary import SummarizerRegistry, XarrayDatasetSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = XarrayDatasetSummarizer()
        >>> dataset = xr.Dataset({"a": (("x",), np.ones(3)), "b": (("x",), np.zeros(3))})
        >>> print(summarizer.summarize(dataset, registry))  # doctest: +ELLIPSIS
        <class '...'> | dims={'x': 3} | data_vars=a: float64, b: float64

        ```
    """

    def _check_dependencies(self) -> None:
        check_xarray()

    def _describe(self, data: xr.Dataset) -> list[str]:
        variables = [(name, variable.dtype) for name, variable in data.data_vars.items()]
        return [f"dims={dict(data.sizes)}", f"data_vars={self._format_fields(variables)}"]


class XarrayVariableSummarizer(BaseMetadataSummarizer[xr.Variable]):
    r"""Implement a summarizer for ``xarray.Variable`` objects.

    By default, it displays the metadata (type, dimensions, shape,
    dtype) rather than the values.

    Args:
        show_data: If ``True``, returns ``repr(variable)``, displaying
            the values. If ``False`` (default), returns only the
            metadata.

    Raises:
        MissingPackageError: If xarray is not installed.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.summary import SummarizerRegistry, XarrayVariableSummarizer
        >>> registry = SummarizerRegistry()
        >>> summarizer = XarrayVariableSummarizer()
        >>> variable = xr.Variable(("x", "y"), np.ones((2, 3)))
        >>> print(summarizer.summarize(variable, registry))  # doctest: +ELLIPSIS
        <class '...'> | dims=('x', 'y') | shape=(2, 3) | dtype=float64

        ```
    """

    def _check_dependencies(self) -> None:
        check_xarray()

    def _describe(self, data: xr.Variable) -> list[str]:
        return [f"dims={data.dims}", f"shape={data.shape}", f"dtype={data.dtype}"]
