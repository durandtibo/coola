r"""Reduction utilities for computing statistics over numeric sequences.

A reducer turns a one-dimensional sequence of real numbers into a
statistic (``max``, ``min``, ``mean``, ``median``, ``std``,
``quantile``) or into a sorted list. ``NativeReducer``, ``NumpyReducer``
and ``TorchReducer`` have the same methods, the same errors and the same
result types (a Python ``int`` or ``float``), so the backend can be
swapped. The statistics raise ``EmptySequenceError`` on an empty input,
``NaN`` propagates whatever its position, the input is never modified,
and importing this package does not import NumPy or PyTorch.

Not guaranteed: bit-identical floating point results across backends,
sums that exceed the range of a ``float``, and number types other than
``int`` and ``float``.
"""

__all__ = [
    "BaseReducer",
    "EmptySequenceError",
    "NativeReducer",
    "NumpyReducer",
    "TorchReducer",
    "auto_reducer",
]

from typing import TYPE_CHECKING

from coola.reducer.base import BaseReducer, EmptySequenceError
from coola.reducer.native import NativeReducer
from coola.reducer.utils import auto_reducer
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.reducer.numpy import NumpyReducer
    from coola.reducer.torch import TorchReducer


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "NumpyReducer": "coola.reducer.numpy",
    "TorchReducer": "coola.reducer.torch",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
