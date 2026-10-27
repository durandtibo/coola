r"""Provide deterministic hashing for xarray objects."""

from __future__ import annotations

__all__ = ["XarrayHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node
from coola.utils.imports import is_xarray_available

if TYPE_CHECKING or is_xarray_available():
    import xarray as xr
else:  # pragma: no cover
    from coola.utils.fallback.xarray import xarray as xr

if TYPE_CHECKING:
    from coola.hashing.base import Hasher


class XarrayHasher(InlineDisplayMixin, BaseHasher[Any]):
    r"""Hasher for ``xarray.DataArray``, ``xarray.Dataset`` and
    ``xarray.Variable`` objects.

    The hash depends on the dimensions, the values, the coordinates and
    the attributes (and on the name of a ``DataArray``). The values are
    hashed as NumPy arrays.

    Args:
        tag: The type tag mixed into the hash, e.g. ``"DataArray"``.

    Example:
        ```pycon
        >>> import xarray as xr
        >>> from coola.hashing import XarrayHasher
        >>> hasher = XarrayHasher("DataArray")
        >>> hasher
        XarrayHasher(tag='DataArray')
        >>> hasher.encode(xr.DataArray([1, 2], dims=["x"])).tag
        'xarray.DataArray'

        ```
    """

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def _display_fields(self) -> dict[str, Any]:
        return {"tag": self._tag}

    def encode(self, data: Any) -> Encoding:
        r"""Describe an xarray object.

        Args:
            data: The ``Variable``, ``DataArray`` or ``Dataset`` to
                describe.

        Returns:
            The ``Node`` of the parts of the object: the dimensions, the
            values and the attributes for a variable; the name, the
            variable and the coordinates for a data array; the
            variables, the coordinates and the attributes for a dataset.
        """
        if self._tag == "Variable":
            children = [list(data.dims), data.values, dict(data.attrs)]
        elif self._tag == "DataArray":
            children = [
                data.name,
                data.variable,
                {k: v.variable for k, v in data.coords.items()},
            ]
        else:
            children = [
                {k: v.variable for k, v in data.data_vars.items()},
                {k: v.variable for k, v in data.coords.items()},
                dict(data.attrs),
            ]
        return Node(f"xarray.{self._tag}", children)


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the xarray objects.

    Returns:
        The hashers of the xarray objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the xarray objects.
    """
    if not is_xarray_available():
        return {}
    return {
        xr.Variable: XarrayHasher("Variable"),
        xr.DataArray: XarrayHasher("DataArray"),
        xr.Dataset: XarrayHasher("Dataset"),
    }
