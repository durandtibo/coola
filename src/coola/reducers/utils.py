from __future__ import annotations

__all__ = ["auto_reducer"]

from coola.reducers import BaseReducer, BasicReducer, NumpyReducer, TorchReducer
from coola.utils import is_numpy_available, is_torch_available


def auto_reducer() -> BaseReducer:
    r"""Finds the "best" reducer to used based on the installed packages.

    The "best" reducer is found by using the following rules:
        - If ``torch`` is available, use ``TorchReducer``
        - If ``numpy`` is available, use ``NumpyReducer``
        - Otherwise, use ``BasicReducer``

    Returns:
    -------
        ``BaseReducer``: The "best" reducer.

    Example usage:

    .. code-block:: pycon

        >>> from coola.reducers import auto_reducer
        >>> reducer = auto_reducer()
    """
    if is_torch_available():
        return TorchReducer()
    if is_numpy_available():
        return NumpyReducer()
    return BasicReducer()
