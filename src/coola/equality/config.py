r"""Define the equality configuration."""

from __future__ import annotations

__all__ = ["EqualityConfig"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coola.equality.testers import BaseEqualityTester


@dataclass
class EqualityConfig:
    r"""Define the config to control the comparison rules."""

    tester: BaseEqualityTester
    equal_nan: bool = False
    atol: float = 0.0
    rtol: float = 0.0
    show_difference: bool = False
