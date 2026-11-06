r"""Define the transformers of the xarray objects."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.jsonable.transformers import ConvertTransformer
from coola.utils.imports import is_xarray_available

if TYPE_CHECKING or is_xarray_available():
    import xarray as xr
else:  # pragma: no cover
    from coola.utils.fallback.xarray import xarray as xr

if TYPE_CHECKING:
    from coola.recursive.base import Transformer


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the xarray objects.

    A ``DataArray``, a ``Dataset`` and a ``Variable`` give the
    dictionary of ``to_dict(data="list")``: the dimensions, the
    attributes, the coordinates and the data as nested lists.

    Returns:
        The transformers of the xarray objects and their associated
            types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the xarray objects.

    Example:
        ```pycon
        >>> from coola.jsonable.xarray import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['DataArray', 'Dataset', 'Variable']

        ```
    """
    if not is_xarray_available():
        return {}
    to_dict: ConvertTransformer[Any] = ConvertTransformer(lambda data: data.to_dict(data="list"))
    return {xr.DataArray: to_dict, xr.Dataset: to_dict, xr.Variable: to_dict}
