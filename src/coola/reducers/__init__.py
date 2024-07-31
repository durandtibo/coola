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

from coola.reducers.base import BaseBasicReducer, BaseReducer, EmptySequenceError
from coola.reducers.native import NativeReducer
from coola.reducers.numpy_ import NumpyReducer
from coola.reducers.registry import ReducerRegistry
from coola.reducers.torch_ import TorchReducer
from coola.reducers.utils import auto_reducer
