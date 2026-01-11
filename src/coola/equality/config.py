r"""Define the equality configuration."""

from __future__ import annotations

__all__ = ["EqualityConfig"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coola.equality.testers import BaseEqualityTester


@dataclass
class EqualityConfig:
    r"""Define the config to control the comparison rules.

    Args:
        tester: The equality tester to use for comparisons.
        equal_nan: If ``True``, NaN values will be considered equal.
            Defaults to ``False``.
        atol: The absolute tolerance parameter for floating-point
            comparisons. Defaults to 0.0.
        rtol: The relative tolerance parameter for floating-point
            comparisons. Defaults to 0.0.
        show_difference: If ``True``, shows differences between
            non-equal objects. Defaults to ``False``.

    Example:
        ```pycon
        >>> from coola.equality import EqualityConfig
        >>> from coola.equality.testers import EqualityTester
        >>> config = EqualityConfig(tester=EqualityTester())
        >>> config
        EqualityConfig(tester=EqualityTester(...), equal_nan=False, atol=0.0, rtol=0.0, show_difference=False)

        ```
    """

    tester: BaseEqualityTester
    equal_nan: bool = False
    atol: float = 0.0
    rtol: float = 0.0
    show_difference: bool = False
