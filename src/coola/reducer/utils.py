r"""Implement utility functions for reducer."""

from __future__ import annotations

__all__ = ["auto_reducer"]

import pkgutil
from typing import TYPE_CHECKING

from coola.reducer.native import NativeReducer
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING:
    from coola.reducer.base import BaseReducer


def auto_reducer() -> BaseReducer:
    r"""Find the "best" reducer to used based on the installed packages.

    The "best" reducer is the fastest one that is installed for plain
    Python sequences. It is found by using the following rules:
        - If ``numpy`` is available, use ``NumpyReducer``
        - Otherwise, use ``NativeReducer``

    ``TorchReducer`` is never chosen implicitly: for a list, it is
    slower than ``NumpyReducer`` (the values are converted to a tensor
    first) and importing ``torch`` is slow. Create a ``TorchReducer``
    explicitly to reduce tensors that live on a device. The rules do not
    look at the values. ``NumpyReducer`` fails for an integer that does
    not fit in 64 bits: use ``NativeReducer`` for arbitrary precision
    integers.

    Returns:
        The "best" reducer.

    Example:
        ```pycon
        >>> from coola.reducer import auto_reducer
        >>> reducer = auto_reducer()

        ```
    """
    if is_numpy_available():
        return pkgutil.resolve_name("coola.reducer.numpy:NumpyReducer")()
    return NativeReducer()
