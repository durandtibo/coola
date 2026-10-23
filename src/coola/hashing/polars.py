r"""Provide deterministic hashing for polars objects."""

from __future__ import annotations

__all__ = ["PolarsDataFrameHasher", "PolarsSeriesHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged
from coola.utils.imports import is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl

if TYPE_CHECKING:
    from coola.hashing.base import Hasher
    from coola.hashing.registry import HasherRegistry


def _hash_series(
    series: pl.Series, registry: HasherRegistry, length: int, ignore_unhashable: bool
) -> str:
    r"""Hash the name, the dtype and the values of a series."""
    if series.dtype.is_numeric() and series.null_count() == 0:
        values = series.to_numpy()  # fast path, no Python objects
    else:
        values = series.to_list()
    parts = [
        registry.hash(series.name, length=length, ignore_unhashable=ignore_unhashable),
        str(series.dtype),
        registry.hash(values, length=length, ignore_unhashable=ignore_unhashable),
    ]
    return hash_tagged("polars.Series", "\0".join(parts), length=length)


class PolarsSeriesHasher(InlineDisplayMixin, BaseHasher["pl.Series"]):
    r"""Hasher for ``polars.Series`` objects.

    The hash depends on the name, the dtype and the values, ``null``
    included.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.hashing import PolarsSeriesHasher, get_default_registry
        >>> hasher = PolarsSeriesHasher()
        >>> hasher
        PolarsSeriesHasher()
        >>> len(hasher.hash(pl.Series("a", [1, 2, 3]), registry=get_default_registry()))
        64

        ```
    """

    def hash(
        self,
        data: pl.Series,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        return _hash_series(data, registry, length, ignore_unhashable)


class PolarsDataFrameHasher(InlineDisplayMixin, BaseHasher["pl.DataFrame"]):
    r"""Hasher for ``polars.DataFrame`` objects.

    The hash depends on the column names, the dtypes and the values.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.hashing import PolarsDataFrameHasher, get_default_registry
        >>> hasher = PolarsDataFrameHasher()
        >>> hasher
        PolarsDataFrameHasher()
        >>> len(hasher.hash(pl.DataFrame({"a": [1, 2]}), registry=get_default_registry()))
        64

        ```
    """

    def hash(
        self,
        data: pl.DataFrame,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        parts = [_hash_series(s, registry, length, ignore_unhashable) for s in data.get_columns()]
        return hash_tagged("polars.DataFrame", "\0".join(parts), length=length)


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the polars objects.

    Returns:
        The hashers of the polars objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the polars objects.
    """
    if not is_polars_available():
        return {}
    return {pl.Series: PolarsSeriesHasher(), pl.DataFrame: PolarsDataFrameHasher()}
