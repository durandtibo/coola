r"""Provide functions to round numerical values.

The values can be native numbers, NumPy arrays, PyTorch tensors, or
pandas, polars and pyarrow Series, arrays and tables. A rounder can be
registered for another type with ``register_rounders``.
"""

from __future__ import annotations

__all__ = [
    "RoundMode",
    "Rounder",
    "UnsupportedTypeError",
    "get_default_registry",
    "register_rounders",
    "round_values",
]

from coola.rounding.common import Rounder, RoundMode, UnsupportedTypeError
from coola.rounding.interface import get_default_registry, register_rounders, round_values
