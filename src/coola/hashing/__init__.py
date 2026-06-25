r"""Provide deterministic hashing functions."""

from __future__ import annotations

__all__ = ["BaseHasher", "HasherRegistry", "hash_string"]

from coola.hashing.base import BaseHasher
from coola.hashing.registry import HasherRegistry
from coola.hashing.string import hash_string
