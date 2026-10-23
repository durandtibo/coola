r"""Implement utility functions for reducer."""

from __future__ import annotations

__all__ = ["auto_reducer"]

from typing import TYPE_CHECKING

from coola.reducer import BaseReducer, NativeReducer
from coola.utils.imports import import_object, is_numpy_available, is_torch_available

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.reducer import BaseReducer


def auto_reducer() -> BaseReducer[Sequence[float]]:
    r"""Find the "best" reducer to used based on the installed packages.

    The "best" reducer is found by using the following rules:
        - If ``torch`` is available, use ``TorchReducer``
        - If ``numpy`` is available, use ``NumpyReducer``
        - Otherwise, use ``NativeReducer``

    The rules do not look at the values, so ``TorchReducer`` is also
    chosen for a plain list. It converts the values to a tensor: it is
    slower than ``NativeReducer`` for a short list, and it fails for an
    integer that does not fit in 64 bits. ``NumpyReducer`` has the same
    limit (and converts it to a float). Use ``NativeReducer`` for
    arbitrary precision integers.

    Returns:
        The "best" reducer.

    Example:
        ```pycon
        >>> from coola.reducer import auto_reducer
        >>> reducer = auto_reducer()

        ```
    """
    if is_torch_available():
        return import_object("coola.reducer.torch:TorchReducer")()
    if is_numpy_available():
        return import_object("coola.reducer.numpy:NumpyReducer")()
    return NativeReducer()
