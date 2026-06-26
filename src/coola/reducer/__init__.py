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

from coola.reducer.base import BaseBasicReducer, BaseReducer, EmptySequenceError
from coola.reducer.native import NativeReducer
from coola.reducer.numpy import NumpyReducer
from coola.reducer.torch import TorchReducer
from coola.reducer.utils import auto_reducer
