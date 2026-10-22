r"""Provide deterministic hashing for xarray objects."""

from __future__ import annotations

__all__ = ["XarrayHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


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
        >>> from coola.hashing import XarrayHasher, get_default_registry
        >>> hasher = XarrayHasher("DataArray")
        >>> hasher
        XarrayHasher(tag='DataArray')
        >>> len(hasher.hash(xr.DataArray([1, 2], dims=["x"]), registry=get_default_registry()))
        64

        ```
    """

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"tag": self._tag}

    def hash(
        self,
        data: Any,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        def sub(obj: Any) -> str:
            return registry.hash(obj, length=length, ignore_unhashable=ignore_unhashable)

        tag = f"xarray.{self._tag}"
        if self._tag == "Variable":
            parts = [sub(list(data.dims)), sub(data.values), sub(dict(data.attrs))]
        elif self._tag == "DataArray":
            parts = [
                sub(data.name),
                sub(data.variable),
                sub({k: v.variable for k, v in data.coords.items()}),
            ]
        else:
            parts = [
                sub({k: v.variable for k, v in data.data_vars.items()}),
                sub({k: v.variable for k, v in data.coords.items()}),
                sub(dict(data.attrs)),
            ]
        return hash_tagged(tag, "\0".join(parts), length=length)
