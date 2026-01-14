r"""Implement equality testers for xarray objects.

This module provides equality testers for xarray.DataArray, xarray.Dataset,
and xarray.Variable with recursive comparison of data, coordinates, and attributes.
"""

from __future__ import annotations

__all__ = [
    "XarrayDataArrayEqualityTester",
    "XarrayDatasetEqualityTester",
    "XarrayVariableEqualityTester",
]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    SameAttributeHandler,
    SameDataHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_xarray, is_xarray_available

if TYPE_CHECKING or is_xarray_available():
    import xarray as xr
else:  # pragma: no cover
    from coola.utils.fallback.xarray import xarray as xr

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class XarrayDataArrayEqualityTester(BaseEqualityTester[xr.DataArray]):
    r"""Implement an equality tester for ``xarray.DataArray``.

    This tester compares xarray DataArrays by checking their variable, name,
    and coordinates. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are xarray DataArrays
    3. SameAttributeHandler("variable"): Compare the underlying Variable
    4. SameAttributeHandler("name"): Compare the DataArray names
    5. SameAttributeHandler("_coords"): Compare the coordinate mappings
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

    def __init__(self) -> None:
        check_xarray()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameAttributeHandler(name="variable")).chain(
            SameAttributeHandler(name="name")
        ).chain(SameAttributeHandler(name="_coords")).chain(TrueHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: xr.DataArray,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class XarrayDatasetEqualityTester(BaseEqualityTester[xr.Dataset]):
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

    def __init__(self) -> None:
        check_xarray()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameAttributeHandler(name="data_vars")).chain(
            SameAttributeHandler(name="coords")
        ).chain(SameAttributeHandler(name="attrs")).chain(TrueHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: xr.Dataset,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class XarrayVariableEqualityTester(BaseEqualityTester[xr.Variable]):
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

    def __init__(self) -> None:
        check_xarray()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDataHandler()).chain(
            SameAttributeHandler(name="dims")
        ).chain(SameAttributeHandler(name="attrs")).chain(TrueHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: xr.Variable,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
