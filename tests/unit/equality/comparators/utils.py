r"""Define some utility functions/classes."""

from __future__ import annotations

__all__ = ["ExamplePair"]

from dataclasses import dataclass
from typing import Any


@dataclass
class ExamplePair:
    object1: Any
    object2: Any
    expected_message: str | None = None
