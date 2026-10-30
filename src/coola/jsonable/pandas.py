r"""Define the transformers of the pandas objects."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.jsonable.transformers import ConvertTransformer
from coola.utils.imports import is_pandas_available

if TYPE_CHECKING or is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

if TYPE_CHECKING:
    from coola.recursive.base import Transformer


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the pandas objects.

    A series and an index give a list. The missing values ``pd.NaT``
    (a ``datetime`` subclass) and ``pd.NA`` give ``None``. A data frame
    has no single JSON shape, so it is not converted: register a
    transformer that chooses the shape, for example
    ``ConvertTransformer(lambda df: df.to_dict(orient="records"))``.

    Returns:
        The transformers of the pandas objects and their associated
            types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the pandas objects.

    Example:
        ```pycon
        >>> from coola.jsonable.pandas import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['Series', 'Index', 'NaTType', 'NAType']

        ```
    """
    if not is_pandas_available():
        return {}
    to_none = ConvertTransformer(lambda _: None)
    return {
        pd.Series: ConvertTransformer(lambda series: series.tolist()),
        pd.Index: ConvertTransformer(lambda index: index.tolist()),
        type(pd.NaT): to_none,
        type(pd.NA): to_none,
    }
