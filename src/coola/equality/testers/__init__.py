r"""Contain the testers to check if two objects are equal or not."""

from __future__ import annotations

__all__ = ["BaseEqualityTester", "EqualityTester", "LocalEqualityTester"]

from coola.equality.testers.base import BaseEqualityTester
from coola.equality.testers.default import EqualityTester, LocalEqualityTester
