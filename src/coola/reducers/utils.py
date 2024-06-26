r"""Implement utility functions for reducers."""

from __future__ import annotations

__all__ = ["auto_reducer"]

from coola.reducers import BaseReducer, NativeReducer, NumpyReducer, TorchReducer
from coola.utils import is_numpy_available, is_torch_available


def auto_reducer() -> BaseReducer:
    r"""Find the "best" reducer to used based on the installed packages.

    The "best" reducer is found by using the following rules:
        - If ``torch`` is available, use ``TorchReducer``
        - If ``numpy`` is available, use ``NumpyReducer``
        - Otherwise, use ``BasicReducer``

    Returns:
        The "best" reducer.

    Example usage:

    ```pycon

    >>> from coola.reducers import auto_reducer
    >>> reducer = auto_reducer()

    ```
    """
    if is_torch_available():
        return TorchReducer()
    if is_numpy_available():
        return NumpyReducer()
    return NativeReducer()
