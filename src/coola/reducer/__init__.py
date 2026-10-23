r"""Reduction utilities for computing statistics over numeric
sequences."""

__all__ = [
    "BaseBasicReducer",
    "BaseReducer",
    "EmptySequenceError",
    "NativeReducer",
    "NumpyReducer",
    "TorchReducer",
    "auto_reducer",
]

from typing import TYPE_CHECKING

from coola.reducer.base import BaseBasicReducer, BaseReducer, EmptySequenceError
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
