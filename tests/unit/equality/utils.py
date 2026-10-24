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


class BadStr:
    r"""An object that is compared by value and whose ``__str__``
    raises."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BadStr) and self.value == other.value

    def __str__(self) -> str:
        msg = "broken __str__"
        raise ZeroDivisionError(msg)
