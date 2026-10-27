r"""Utilities to control and restore random number generator states."""

from __future__ import annotations

__all__ = [
    "BaseRandomManager",
    "NumpyRandomManager",
    "PythonRandomManager",
    "RandomManagerRegistry",
    "TorchRandomManager",
    "get_default_random_manager_registry",
    "get_default_registry",
    "get_random_seed",
    "get_rng_state",
    "get_torch_generator",
    "manual_seed",
    "random_seed",
    "register_managers",
    "set_rng_state",
    "validate_seed",
]

from typing import TYPE_CHECKING

from coola.random.base import BaseRandomManager
from coola.random.interface import (
    get_default_random_manager_registry,
    get_default_registry,
    get_rng_state,
    manual_seed,
    random_seed,
    register_managers,
    set_rng_state,
)
from coola.random.python import PythonRandomManager
from coola.random.registry import RandomManagerRegistry
from coola.random.seed import validate_seed
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.random.numpy import NumpyRandomManager
    from coola.random.torch import TorchRandomManager, get_random_seed, get_torch_generator


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "NumpyRandomManager": "coola.random.numpy",
    "TorchRandomManager": "coola.random.torch",
    "get_random_seed": "coola.random.torch",
    "get_torch_generator": "coola.random.torch",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
