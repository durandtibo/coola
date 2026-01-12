r"""Contain the equality testers to check if two objects are equal or
not."""

from __future__ import annotations

__all__ = [
    "BaseEqualityTester", "EqualityTesterRegistry"
]

from coola.equality.tester.base import BaseEqualityTester
from coola.equality.tester.registry import EqualityTesterRegistry
