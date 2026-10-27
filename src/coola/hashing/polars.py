r"""Provide deterministic hashing for polars objects."""

from __future__ import annotations

__all__ = ["PolarsDataFrameHasher", "PolarsSeriesHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node
from coola.utils.imports import is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl

if TYPE_CHECKING:
    from coola.hashing.base import Hasher


class PolarsSeriesHasher(InlineDisplayMixin, BaseHasher["pl.Series"]):
    r"""Hasher for ``polars.Series`` objects.

    The hash depends on the name, the dtype and the values, ``null``
    included.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.hashing import PolarsSeriesHasher
        >>> hasher = PolarsSeriesHasher()
        >>> hasher
        PolarsSeriesHasher()
        >>> hasher.encode(pl.Series("a", [1, 2, 3])).tag
        'polars.Series'

        ```
    """

    def encode(self, data: pl.Series) -> Encoding:
        r"""Describe a polars series.

        Args:
            data: The series to describe.

        Returns:
            The ``Node`` of the name, the dtype and the values.
        """
        if data.dtype.is_numeric() and data.null_count() == 0:
            values = data.to_numpy()  # fast path, no Python objects
        else:
            values = data.to_list()
        return Node("polars.Series", [data.name, str(data.dtype), values])


class PolarsDataFrameHasher(InlineDisplayMixin, BaseHasher["pl.DataFrame"]):
    r"""Hasher for ``polars.DataFrame`` objects.

    The hash depends on the column names, the dtypes and the values.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.hashing import PolarsDataFrameHasher
        >>> hasher = PolarsDataFrameHasher()
        >>> hasher
        PolarsDataFrameHasher()
        >>> hasher.encode(pl.DataFrame({"a": [1, 2]})).tag
        'polars.DataFrame'

        ```
    """

    def encode(self, data: pl.DataFrame) -> Encoding:
        r"""Describe a polars data frame.

        Args:
            data: The data frame to describe.

        Returns:
            The ``Node`` of the columns.
        """
        return Node("polars.DataFrame", data.get_columns())


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
