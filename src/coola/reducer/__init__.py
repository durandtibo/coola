r"""Contain the reducer implementations."""

__all__ = [
    "BaseBasicReducer",
    "BaseReducer",
    "EmptySequenceError",
    "NativeReducer",
    "NumpyReducer",
    "ReducerRegistry",
    "TorchReducer",
    "auto_reducer",
]

from coola.reducer.base import BaseBasicReducer, BaseReducer, EmptySequenceError
from coola.reducer.native import NativeReducer
from coola.reducer.numpy import NumpyReducer
from coola.reducer.registry import ReducerRegistry
from coola.reducer.torch import TorchReducer
from coola.reducer.utils import auto_reducer
