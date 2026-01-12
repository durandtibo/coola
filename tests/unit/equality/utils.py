r"""Define some utility functions/classes."""

from __future__ import annotations

__all__ = ["ExamplePair"]

from dataclasses import dataclass
from typing import Any


@dataclass
class ExamplePair:
    actual: Any
    expected: Any
    expected_message: str | None = None
    atol: float = 0.0
    rtol: float = 0.0
