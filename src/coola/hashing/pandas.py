r"""Provide deterministic hashing for pandas objects."""

from __future__ import annotations

__all__ = ["PandasDataFrameHasher", "PandasSeriesHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged
from coola.utils.imports import is_pandas_available

if TYPE_CHECKING or is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

if TYPE_CHECKING:
    from coola.hashing.base import Hasher
    from coola.hashing.registry import HasherRegistry


def _hash_values(
    values: Any, registry: HasherRegistry, length: int, ignore_unhashable: bool
) -> str:
    return registry.hash(values, length=length, ignore_unhashable=ignore_unhashable)


def _hash_dtype(dtype: Any, registry: HasherRegistry, length: int, ignore_unhashable: bool) -> str:
    r"""Hash a dtype: ``str(dtype)`` is ``"category"`` for every categorical
    dtype, so the categories and the ``ordered`` flag are added."""
    if isinstance(dtype, pd.CategoricalDtype):
        categories = _hash_values(dtype.categories.to_numpy(), registry, length, ignore_unhashable)
        return f"{dtype}\0{dtype.ordered}\0{categories}"
    return str(dtype)


def _hash_index(
    index: pd.Index, registry: HasherRegistry, length: int, ignore_unhashable: bool
) -> str:
    r"""Hash an index: its names, its dtype(s) and its values.

    The values of a ``MultiIndex`` are hashed level by level, not as an
    array of tuples.
    """
    levels = (
        [index.get_level_values(i) for i in range(index.nlevels)]
        if isinstance(index, pd.MultiIndex)
        else [index]
    )
    parts = [_hash_values(list(index.names), registry, length, ignore_unhashable)]
    for level in levels:
        parts.append(_hash_dtype(level.dtype, registry, length, ignore_unhashable))
        parts.append(_hash_values(level.to_numpy(), registry, length, ignore_unhashable))
    return "\0".join(parts)


class PandasSeriesHasher(InlineDisplayMixin, BaseHasher["pd.Series"]):
    r"""Hasher for ``pandas.Series`` objects.

    The hash depends on the name, the dtype, the index (its names, its
    dtype and its values) and the values. For a categorical dtype, the
    categories and the ``ordered`` flag are part of the dtype.
    The values are hashed as a NumPy array, so the elements of an
    ``object`` column are hashed through the registry. A missing value
    that has no registered hasher (e.g. ``pandas.NA``) raises an error.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.hashing import HasherRegistry, PandasSeriesHasher, get_default_registry
        >>> hasher = PandasSeriesHasher()
        >>> hasher
        PandasSeriesHasher()
        >>> len(hasher.hash(pd.Series([1, 2, 3]), registry=get_default_registry()))
        64

        ```
    """

    def hash(
        self,
        data: pd.Series,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        parts = [
            _hash_values(data.name, registry, length, ignore_unhashable),
            _hash_dtype(data.dtype, registry, length, ignore_unhashable),
            _hash_index(data.index, registry, length, ignore_unhashable),
            _hash_values(data.to_numpy(), registry, length, ignore_unhashable),
        ]
        return hash_tagged("pandas.Series", "\0".join(parts), length=length)


class PandasDataFrameHasher(InlineDisplayMixin, BaseHasher["pd.DataFrame"]):
    r"""Hasher for ``pandas.DataFrame`` objects.

    The hash depends on the column labels (and the name of the columns
    index), the index (its names, its dtype and its values) and, for
    each column, its dtype and its values (see ``PandasSeriesHasher``).

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.hashing import HasherRegistry, PandasDataFrameHasher, get_default_registry
        >>> hasher = PandasDataFrameHasher()
        >>> hasher
        PandasDataFrameHasher()
        >>> len(hasher.hash(pd.DataFrame({"a": [1, 2]}), registry=get_default_registry()))
        64

        ```
    """

    def hash(
        self,
        data: pd.DataFrame,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        parts = [
            _hash_index(data.columns, registry, length, ignore_unhashable),
            _hash_index(data.index, registry, length, ignore_unhashable),
        ]
        for i in range(data.shape[1]):
            column = data.iloc[:, i]
            parts.append(_hash_dtype(column.dtype, registry, length, ignore_unhashable))
            parts.append(_hash_values(column.to_numpy(), registry, length, ignore_unhashable))
        return hash_tagged("pandas.DataFrame", "\0".join(parts), length=length)


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the pandas objects.

    Returns:
        The hashers of the pandas objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the pandas objects.
    """
    if not is_pandas_available():
        return {}
    return {pd.Series: PandasSeriesHasher(), pd.DataFrame: PandasDataFrameHasher()}
