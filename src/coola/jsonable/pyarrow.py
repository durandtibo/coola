r"""Define the transformers of the pyarrow objects."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

from typing import TYPE_CHECKING, Any, cast

from coola.jsonable.transformers import ConvertTransformer
from coola.utils.imports import is_pyarrow_available

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa

if TYPE_CHECKING:
    from coola.recursive.base import Transformer


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the pyarrow objects.

    An array and a chunked array give a list, and a ``null`` gives
    ``None``. A table has no single JSON shape, so it is not converted:
    register a transformer that chooses the shape, for example
    ``ConvertTransformer(lambda table: table.to_pylist())``.

    Returns:
        The transformers of the pyarrow objects and their associated
            types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the pyarrow objects.

    Example:
        ```pycon
        >>> from coola.jsonable.pyarrow import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['Array', 'ChunkedArray']

        ```
    """
    if not is_pyarrow_available():
        return {}
    pyarrow = cast("Any", pa)
    to_list = ConvertTransformer(lambda array: array.to_pylist())
    return {pyarrow.Array: to_list, pyarrow.ChunkedArray: to_list}
