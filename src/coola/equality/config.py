r"""Define the equality configuration."""

from __future__ import annotations

__all__ = ["EqualityConfig"]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coola.equality.tester.registry import EqualityTesterRegistry


def create_default_registry() -> EqualityTesterRegistry:
    # local import to avoid circular imports.
    from coola.equality.tester.interface import get_default_registry  # noqa: PLC0415

    return get_default_registry()


@dataclass
class EqualityConfig:
    r"""Define the config to control the comparison rules.

    Args:
        registry: The registry with the equality tester to use.
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
        >>> from coola.equality.config import EqualityConfig
        >>> config = EqualityConfig()
        >>> config
        EqualityConfig(registry=EqualityTesterRegistry(...), equal_nan=False, atol=0.0, rtol=0.0, show_difference=False)

        ```
    """

    registry: EqualityTesterRegistry = field(default_factory=create_default_registry)
    equal_nan: bool = False
    atol: float = 0.0
    rtol: float = 0.0
    show_difference: bool = False
