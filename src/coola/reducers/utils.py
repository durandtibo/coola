r"""Implement utility functions for reducers."""

from __future__ import annotations

__all__ = ["auto_reducer"]

from typing import TYPE_CHECKING

from coola.reducers import BaseReducer, NativeReducer, NumpyReducer, TorchReducer
from coola.utils import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    from collections.abc import Sequence


def auto_reducer() -> BaseReducer[Sequence[int | float]]:
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
