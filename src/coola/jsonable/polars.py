r"""Define the transformers of the polars objects."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.jsonable.transformers import ConvertTransformer
from coola.utils.imports import is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl

if TYPE_CHECKING:
    from coola.recursive.base import Transformer


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the polars objects.

    A series gives a list, and a missing value gives ``None``. A data
    frame has no single JSON shape, so it is not converted: register a
    transformer that chooses the shape, for example
    ``ConvertTransformer(lambda df: df.to_dicts())``.

    Returns:
        The transformers of the polars objects and their associated
            types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the polars objects.

    Example:
        ```pycon
        >>> from coola.jsonable.polars import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['Series']

        ```
    """
    if not is_polars_available():
        return {}
    return {pl.Series: ConvertTransformer(lambda series: series.to_list())}
