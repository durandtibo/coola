r"""Contain functionalities to manage randomness."""

from __future__ import annotations

__all__ = [
    "BaseRandomManager",
    "NumpyRandomManager",
    "RandomManagerRegistry",
    "RandomRandomManager",
    "TorchRandomManager",
    "get_default_registry",
    "get_rng_state",
    "manual_seed",
    "numpy_seed",
    "random_seed",
    "register_managers",
    "set_rng_state",
    "torch_seed",
]

from coola.random.base import BaseRandomManager
from coola.random.interface import (
    get_default_registry,
    get_rng_state,
    manual_seed,
    random_seed,
    register_managers,
    set_rng_state,
)
from coola.random.numpy import NumpyRandomManager, numpy_seed
from coola.random.random import RandomRandomManager
from coola.random.registry import RandomManagerRegistry
from coola.random.torch import TorchRandomManager, torch_seed
