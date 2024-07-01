r"""Implement an equality comparator for ``xarray`` objects."""

from __future__ import annotations

__all__ = [
    "XarrayDataArrayEqualityComparator",
    "XarrayDatasetEqualityComparator",
    "XarrayVariableEqualityComparator",
    "get_type_comparator_mapping",
]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    SameAttributeHandler,
    SameDataHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.utils import check_xarray, is_xarray_available

if is_xarray_available():
    import xarray as xr
else:  # pragma: no cover
    xr = Mock()

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class XarrayDataArrayEqualityComparator(BaseEqualityComparator[xr.DataArray]):
    r"""Implement an equality comparator for ``xarray.DataArray``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> import xarray as xr
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import XarrayDataArrayEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = XarrayDataArrayEqualityComparator()
    >>> comparator.equal(
    ...     xr.DataArray(np.arange(6), dims=["z"]),
    ...     xr.DataArray(np.arange(6), dims=["z"]),
    ...     config,
    ... )
    True
    >>> comparator.equal(
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> XarrayDataArrayEqualityComparator:
        return self.__class__()

    def equal(self, actual: xr.DataArray, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


class XarrayDatasetEqualityComparator(BaseEqualityComparator[xr.Dataset]):
    r"""Implement an equality comparator for ``xarray.Dataset``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> import xarray as xr
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import XarrayDatasetEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = XarrayDatasetEqualityComparator()
    >>> comparator.equal(
    ...     xr.Dataset({"x": xr.DataArray(np.arange(6), dims=["z"])}),
    ...     xr.Dataset({"x": xr.DataArray(np.arange(6), dims=["z"])}),
    ...     config,
    ... )
    True
    >>> comparator.equal(
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> XarrayDatasetEqualityComparator:
        return self.__class__()

    def equal(self, actual: xr.Dataset, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


class XarrayVariableEqualityComparator(BaseEqualityComparator[xr.Variable]):
    r"""Implement an equality comparator for ``xarray.Variable``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> import xarray as xr
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import XarrayVariableEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = XarrayVariableEqualityComparator()
    >>> comparator.equal(
    ...     xr.Variable(dims=["z"], data=np.arange(6)),
    ...     xr.Variable(dims=["z"], data=np.arange(6)),
    ...     config,
    ... )
    True
    >>> comparator.equal(
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> XarrayVariableEqualityComparator:
        return self.__class__()

    def equal(self, actual: xr.Variable, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if ``xarray`` is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.xarray_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'xarray...DataArray'>: XarrayDataArrayEqualityComparator(),
     <class 'xarray...Dataset'>: XarrayDatasetEqualityComparator(),
     <class 'xarray...Variable'>: XarrayVariableEqualityComparator()}

    ```
    """
    if not is_xarray_available():
        return {}
    return {
        xr.DataArray: XarrayDataArrayEqualityComparator(),
        xr.Dataset: XarrayDatasetEqualityComparator(),
        xr.Variable: XarrayVariableEqualityComparator(),
    }
