r"""Define the transformers of the NumPy objects."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.jsonable.transformers import ConvertTransformer
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.recursive.base import Transformer


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the NumPy objects.

    An array gives the nested lists of ``tolist`` and a scalar gives its
    Python value with ``item``. The result is converted in turn, so a
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
        np.ndarray: ConvertTransformer(lambda array: array.tolist()),
        np.generic: ConvertTransformer(lambda scalar: scalar.item()),
    }
