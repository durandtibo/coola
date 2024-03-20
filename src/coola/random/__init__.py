r"""Contain functionalities to manage randomness."""

from __future__ import annotations

__all__ = ["BaseRandomSeedSetter", "TorchRandomSeedSetter"]

from coola.random.base import BaseRandomSeedSetter
from coola.random.torch_ import TorchRandomSeedSetter
