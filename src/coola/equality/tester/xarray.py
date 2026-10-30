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

from operator import attrgetter
from typing import TYPE_CHECKING, Any

from coola.equality._helpers.utils import AttributeName
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_xarray, is_xarray_available

if TYPE_CHECKING or is_xarray_available():
    import xarray as xr
else:  # pragma: no cover
    from coola.utils.fallback.xarray import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine


def _attributes(
    actual: object, expected: object, names: tuple[str, ...]
) -> Iterator[tuple[object, object, object]]:
    r"""Return the children of two objects: the values of their
    attributes.

    A dotted name (``"coords.variables"``) is read through the
    attributes, and it is one part of the path.

    Args:
        actual: The actual object.
        expected: The expected object.
        names: The names of the attributes to compare.

    Yields:
        The triplet ``(actual value, expected value, path part)`` of
            each attribute.
    """
    for name in names:
        getter = attrgetter(name)
        yield getter(actual), getter(expected), AttributeName(name)


class XarrayDataArrayEqualityTester(BaseEqualityTester[xr.DataArray]):
    r"""Implement an equality tester for ``xarray.DataArray``.

    This tester compares the variable (data, dimensions and attributes),
    the name and the coordinate variables of two DataArrays. The
    location of a difference is the name of the attribute.

    Raises:
        MissingPackageError: If ``xarray`` is not installed.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> engine.run(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"]))
        True
        >>> engine.run(xr.DataArray(np.ones(6), dims=["z"]), xr.DataArray(np.zeros(6), dims=["z"]))
        False

        ```
    """

    def __init__(self) -> None:
        check_xarray()

    def objects_are_equal(
        self, actual: xr.DataArray, expected: xr.DataArray, _ctx: Engine
    ) -> Iterator[tuple[object, object, object]]:
        return _attributes(actual, expected, ("variable", "name", "coords.variables"))


class XarrayDatasetEqualityTester(BaseEqualityTester[xr.Dataset]):
    r"""Implement an equality tester for ``xarray.Dataset``.

    This tester compares the data variables, the coordinates and the
    attributes of two Datasets. The location of a difference is the
    name of the attribute.

    Raises:
        MissingPackageError: If ``xarray`` is not installed.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> engine.run(
        ...     xr.Dataset({"x": ("z", np.arange(6))}), xr.Dataset({"x": ("z", np.arange(6))})
        ... )
        True
        >>> engine.run(
        ...     xr.Dataset({"x": ("z", np.ones(6))}), xr.Dataset({"x": ("z", np.zeros(6))})
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_xarray()

    def objects_are_equal(
        self, actual: xr.Dataset, expected: xr.Dataset, _ctx: Engine
    ) -> Iterator[tuple[object, object, object]]:
        return _attributes(actual, expected, ("data_vars", "coords", "attrs"))


class XarrayVariableEqualityTester(BaseEqualityTester[xr.Variable]):
    r"""Implement an equality tester for ``xarray.Variable``.

    This tester compares the data, the dimension names and the
    attributes of two Variables. The location of a difference is the
    name of the attribute.

    Raises:
        MissingPackageError: If ``xarray`` is not installed.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> engine.run(
        ...     xr.Variable(dims=["z"], data=np.arange(6)),
        ...     xr.Variable(dims=["z"], data=np.arange(6)),
        ... )
        True
        >>> engine.run(
        ...     xr.Variable(dims=["z"], data=np.zeros(6)), xr.Variable(dims=["z"], data=np.ones(6))
        ... )
        False

        ```
    """

    def __init__(self) -> None:
        check_xarray()

    def objects_are_equal(
        self, actual: xr.Variable, expected: xr.Variable, _ctx: Engine
    ) -> Iterator[tuple[object, object, object]]:
        return _attributes(actual, expected, ("data", "dims", "attrs"))


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
