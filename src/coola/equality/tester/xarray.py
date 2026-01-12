r"""Implement an equality tester for ``xarray`` objects."""

from __future__ import annotations

__all__ = [
    "XarrayDataArrayEqualityTester",
    "XarrayDatasetEqualityTester",
    "XarrayVariableEqualityTester",
]

import logging
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
    from coola.equality.config import EqualityConfig2

logger: logging.Logger = logging.getLogger(__name__)


class XarrayDataArrayEqualityTester(BaseEqualityTester[xr.DataArray]):
    r"""Implement an equality tester for ``xarray.DataArray``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import XarrayDataArrayEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = XarrayDataArrayEqualityTester()
        >>> tester.equal(
        ...     xr.DataArray(np.arange(6), dims=["z"]),
        ...     xr.DataArray(np.arange(6), dims=["z"]),
        ...     config,
        ... )
        True
        >>> tester.equal(
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
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class XarrayDatasetEqualityTester(BaseEqualityTester[xr.Dataset]):
    r"""Implement an equality tester for ``xarray.Dataset``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import XarrayDatasetEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = XarrayDatasetEqualityTester()
        >>> tester.equal(
        ...     xr.Dataset({"x": xr.DataArray(np.arange(6), dims=["z"])}),
        ...     xr.Dataset({"x": xr.DataArray(np.arange(6), dims=["z"])}),
        ...     config,
        ... )
        True
        >>> tester.equal(
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
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class XarrayVariableEqualityTester(BaseEqualityTester[xr.Variable]):
    r"""Implement an equality tester for ``xarray.Variable``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.tester import XarrayVariableEqualityTester
        >>> config = EqualityConfig2()
        >>> tester = XarrayVariableEqualityTester()
        >>> tester.equal(
        ...     xr.Variable(dims=["z"], data=np.arange(6)),
        ...     xr.Variable(dims=["z"], data=np.arange(6)),
        ...     config,
        ... )
        True
        >>> tester.equal(
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
        config: EqualityConfig2,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
