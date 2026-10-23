r"""Implement equality testers for xarray objects.

This module provides equality testers for xarray.DataArray,
xarray.Dataset, and xarray.Variable with recursive comparison of data,
coordinates, and attributes.
"""

from __future__ import annotations

__all__ = [
    "XarrayDataArrayEqualityTester",
    "XarrayDatasetEqualityTester",
    "XarrayVariableEqualityTester",
]

from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    SameAttributeHandler,
    SameDataHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester
from coola.utils.imports import check_xarray, is_xarray_available

if TYPE_CHECKING or is_xarray_available():
    import xarray as xr
else:  # pragma: no cover
    from coola.utils.fallback.xarray import xarray as xr

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler
    from coola.equality.tester.base import BaseEqualityTester


class XarrayDataArrayEqualityTester(BaseChainEqualityTester[xr.DataArray]):
    r"""Implement an equality tester for ``xarray.DataArray``.

    This tester compares xarray DataArrays by checking their variable, name,
    and coordinates. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are xarray DataArrays
    3. SameAttributeHandler("variable"): Compare the underlying Variable
    4. SameAttributeHandler("name"): Compare the DataArray names
    5. SameAttributeHandler("coords.variables"): Compare the coordinate variables
    6. TrueHandler: Return True if all checks pass

    Example:
        Basic DataArray comparison:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import XarrayDataArrayEqualityTester
        >>> config = EqualityConfig()
        >>> tester = XarrayDataArrayEqualityTester()
        >>> tester.objects_are_equal(
        ...     xr.DataArray(np.arange(6), dims=["z"]),
        ...     xr.DataArray(np.arange(6), dims=["z"]),
        ...     config,
        ... )
        True
        >>> tester.objects_are_equal(
        ...     xr.DataArray(np.ones(6), dims=["z"]),
        ...     xr.DataArray(np.zeros(6), dims=["z"]),
        ...     config,
        ... )
        False

        ```
    """

    def _check_dependencies(self) -> None:
        check_xarray()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameAttributeHandler(name="variable"),
            SameAttributeHandler(name="name"),
            SameAttributeHandler(name="coords.variables"),
            TrueHandler(),
        )


class XarrayDatasetEqualityTester(BaseChainEqualityTester[xr.Dataset]):
    r"""Implement an equality tester for ``xarray.Dataset``.

    This tester compares xarray Datasets by checking their data variables,
    coordinates, and attributes. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are xarray Datasets
    3. SameAttributeHandler("data_vars"): Compare data variable dictionaries
    4. SameAttributeHandler("coords"): Compare coordinate dictionaries
    5. SameAttributeHandler("attrs"): Compare attribute dictionaries
    6. TrueHandler: Return True if all checks pass

    Example:
        Basic Dataset comparison:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import XarrayDatasetEqualityTester
        >>> config = EqualityConfig()
        >>> tester = XarrayDatasetEqualityTester()
        >>> tester.objects_are_equal(
        ...     xr.Dataset({"x": xr.DataArray(np.arange(6), dims=["z"])}),
        ...     xr.Dataset({"x": xr.DataArray(np.arange(6), dims=["z"])}),
        ...     config,
        ... )
        True
        >>> tester.objects_are_equal(
        ...     xr.Dataset({"x": xr.DataArray(np.zeros(6), dims=["z"])}),
        ...     xr.Dataset({"x": xr.DataArray(np.ones(6), dims=["z"])}),
        ...     config,
        ... )
        False

        ```
    """

    def _check_dependencies(self) -> None:
        check_xarray()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameAttributeHandler(name="data_vars"),
            SameAttributeHandler(name="coords"),
            SameAttributeHandler(name="attrs"),
            TrueHandler(),
        )


class XarrayVariableEqualityTester(BaseChainEqualityTester[xr.Variable]):
    r"""Implement an equality tester for ``xarray.Variable``.

    This tester compares xarray Variables by checking their data, dimensions,
    and attributes. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are xarray Variables
    3. SameDataHandler: Compare the underlying data arrays
    4. SameAttributeHandler("dims"): Compare the dimension names
    5. SameAttributeHandler("attrs"): Compare attribute dictionaries
    6. TrueHandler: Return True if all checks pass

    Example:
        Basic Variable comparison:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import XarrayVariableEqualityTester
        >>> config = EqualityConfig()
        >>> tester = XarrayVariableEqualityTester()
        >>> tester.objects_are_equal(
        ...     xr.Variable(dims=["z"], data=np.arange(6)),
        ...     xr.Variable(dims=["z"], data=np.arange(6)),
        ...     config,
        ... )
        True
        >>> tester.objects_are_equal(
        ...     xr.Variable(dims=["z"], data=np.zeros(6)),
        ...     xr.Variable(dims=["z"], data=np.ones(6)),
        ...     config,
        ... )
        False

        ```
    """

    def _check_dependencies(self) -> None:
        check_xarray()

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(
            SameObjectHandler(),
            SameTypeHandler(),
            SameDataHandler(),
            SameAttributeHandler(name="dims"),
            SameAttributeHandler(name="attrs"),
            TrueHandler(),
        )


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the xarray objects.

    Returns:
        The equality testers of the xarray objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the xarray objects.
    """
    if not is_xarray_available():
        return {}

    return {
        xr.DataArray: XarrayDataArrayEqualityTester(),
        xr.Dataset: XarrayDatasetEqualityTester(),
        xr.Variable: XarrayVariableEqualityTester(),
    }
