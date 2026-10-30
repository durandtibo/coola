r"""Define the transformers of the NumPy objects."""

from __future__ import annotations

__all__ = ["NDArrayTransformer", "get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.jsonable.transformers import ConvertTransformer
from coola.recursive import BaseTransformer
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.recursive.base import Expansion, Transformer


class NDArrayTransformer(InlineDisplayMixin, BaseTransformer["np.ndarray"]):
    r"""Transformer for NumPy arrays that rebuilds nested lists.

    The array gives the nested lists of ``tolist``. The elements of an
    array of booleans, integers or floats (of at most 64 bits) are
    Python ``bool``, ``int`` and ``float``, so the list is already JSON
    data and the engine does not visit its elements: checking each of
    the millions of elements would cost more than ``tolist`` itself. The
    other arrays (complex numbers, objects, dates, long doubles) give a
    list that is converted in turn, so what ``json`` rejects is reported.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.jsonable.numpy import NDArrayTransformer
        >>> children, rebuild = NDArrayTransformer().expand(np.arange(3))
        >>> children
        ()
        >>> rebuild([])
        [0, 1, 2]

        ```
    """

    def __init__(self) -> None:
        self._convert = ConvertTransformer(lambda array: array.tolist())

    def expand(self, data: np.ndarray) -> Expansion | None:
        dtype = data.dtype
        # ``longdouble`` has the kind "f" but ``tolist`` keeps its NumPy scalars
        if dtype.kind in "biu" or dtype.char in "efd":
            return (), lambda _: data.tolist()
        return self._convert.expand(data)


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the NumPy objects.

    An array gives the nested lists of ``tolist`` and a scalar gives its
    Python value with ``item``. The result is converted in turn (except
    the elements of the arrays of booleans, integers and floats, see
    ``NDArrayTransformer``), so a
    ``datetime64`` gives an ISO string, a ``timedelta64`` gives a number
    of seconds, and complex numbers and objects are rejected. A
    ``datetime64`` or a ``timedelta64`` with a nanosecond unit gives an
    ``int``, which is kept: cast to a coarser unit first.

    Returns:
        The transformers of the NumPy objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the NumPy objects.

    Example:
        ```pycon
        >>> from coola.jsonable.numpy import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['ndarray', 'generic']

        ```
    """
    if not is_numpy_available():
        return {}
    return {
        np.ndarray: NDArrayTransformer(),
        np.generic: ConvertTransformer(lambda scalar: scalar.item()),
    }
