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

    Note:
        This class is **not thread-safe**. Each comparison should create
        its own config instance. Do not share config instances between
        threads as the internal depth counter is not protected by locks.

    Args:
        registry: The registry with the equality tester to use.
        equal_nan: If ``True``, NaN values will be considered equal.
            Defaults to ``False``.
        atol: The absolute tolerance parameter for floating-point
            comparisons. Must be non-negative. Defaults to 0.0.
        rtol: The relative tolerance parameter for floating-point
            comparisons. Must be non-negative. Defaults to 0.0.
        show_difference: If ``True``, shows differences between
            non-equal objects. Defaults to ``False``.
        max_depth: Maximum recursion depth for nested object
            comparisons. Must be positive. Defaults to 1000.
            Set to a lower value to protect against stack overflow
            with extremely deeply nested structures.

    Raises:
        ValueError: if ``atol`` or ``rtol`` is negative, or if
            ``max_depth`` is not positive.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> config = EqualityConfig()
        >>> config
        EqualityConfig(registry=EqualityTesterRegistry(...), equal_nan=False, atol=0.0, rtol=0.0, show_difference=False, max_depth=1000)

        ```
    """

    registry: EqualityTesterRegistry = field(default_factory=create_default_registry)
    equal_nan: bool = False
    atol: float = 0.0
    rtol: float = 0.0
    show_difference: bool = False
    max_depth: int = 1000
    _current_depth: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.atol < 0:
            msg = f"atol must be non-negative, but got {self.atol}"
            raise ValueError(msg)
        if self.rtol < 0:
            msg = f"rtol must be non-negative, but got {self.rtol}"
            raise ValueError(msg)
        if self.max_depth <= 0:
            msg = f"max_depth must be positive, but got {self.max_depth}"
            raise ValueError(msg)
