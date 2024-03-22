r"""Contain functionalities to manage randomness."""

from __future__ import annotations

__all__ = [
    "BaseRandomManager",
    "NumpyRandomManager",
    "RandomManager",
    "RandomRandomManager",
    "TorchRandomManager",
    "get_random_managers",
    "numpy_seed",
    "random_seed",
    "register_random_managers",
    "torch_seed",
]

from coola.random.base import BaseRandomManager
from coola.random.default import RandomManager, random_seed, register_random_managers
from coola.random.numpy_ import NumpyRandomManager, numpy_seed
from coola.random.random_ import RandomRandomManager
from coola.random.torch_ import TorchRandomManager, torch_seed
from coola.random.utils import get_random_managers

register_random_managers()
