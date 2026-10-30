r"""Define the equality configuration."""

from __future__ import annotations

__all__ = ["EqualityConfig"]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from coola.equality.tolerance import Tolerance
from coola.validation import validate_gt, validate_int

if TYPE_CHECKING:
    from coola.equality.tester.registry import EqualityTesterRegistry


def create_default_registry() -> EqualityTesterRegistry:
    """Create the default equality tester registry.

    This helper is used as a ``dataclasses.field`` default factory to avoid
    importing the registry interface at module import time.

    Returns:
        The default equality tester registry.
    """
    # local import to avoid circular imports.
    from coola.equality.tester.interface import get_default_registry  # noqa: PLC0415

    return get_default_registry()


@dataclass(frozen=True, slots=True)
class EqualityConfig:
    r"""Define the config to control the comparison rules.

    The config is immutable: it can be shared between comparisons and
    threads. The state of one comparison lives in the ``Engine``.

    Args:
        registry: The registry with the equality tester to use.
        tolerance: The tolerance of the numbers. Defaults to no
            tolerance (exact comparison).
        equal_nan: If ``True``, NaN values will be considered equal.
            Defaults to ``False``.
        max_depth: The nesting limit: the maximum nesting level of the
            compared objects. A deeper structure raises
            ``MaxDepthExceededError`` (it is not a display depth like
            the ``max_depth`` of ``summarize``). Must be positive.
            See the "Common options" section of the equality user
            guide. Defaults to 100.

    Raises:
        TypeError: if ``max_depth`` is not an ``int`` (a ``bool`` is
            rejected).
        ValueError: if ``max_depth`` is not positive.

    Example:
        ```pycon
        >>> from coola.equality import Tolerance
        >>> from coola.equality.config import EqualityConfig
        >>> config = EqualityConfig(tolerance=Tolerance(atol=1e-6))
        >>> config.atol
        1e-06
        >>> config.rtol
        0.0

        ```
    """

    registry: EqualityTesterRegistry = field(default_factory=create_default_registry)
    tolerance: Tolerance = field(default_factory=Tolerance)
    equal_nan: bool = False
    max_depth: int = 100

    def __post_init__(self) -> None:
        r"""Validate the configuration parameters after initialization.

        Raises:
            TypeError: If ``max_depth`` is not an int.
            ValueError: If ``max_depth`` is not positive.
        """
        validate_int(self.max_depth, name="max_depth")
        validate_gt(self.max_depth, 0, name="max_depth")

    @property
    def atol(self) -> float:
        r"""The absolute tolerance."""
        return self.tolerance.atol

    @property
    def rtol(self) -> float:
        r"""The relative tolerance."""
        return self.tolerance.rtol
