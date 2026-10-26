r"""Define the equality configuration."""

from __future__ import annotations

__all__ = ["EqualityConfig"]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from coola.validation import (
    validate_finite,
    validate_int,
    validate_non_negative,
    validate_number,
    validate_positive,
)

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


@dataclass(slots=True)
class EqualityConfig:
    r"""Define the config to control the comparison rules.

    Args:
        registry: The registry with the equality tester to use.
        equal_nan: If ``True``, NaN values will be considered equal.
            Defaults to ``False``.
        atol: The absolute tolerance parameter for floating-point
            comparisons. Must be non-negative. Defaults to 0.0.
        rtol: The relative tolerance parameter for floating-point
            comparisons. Must be non-negative. Defaults to 0.0.
        max_depth: The nesting limit: the maximum nesting level of the
            compared objects. A deeper structure raises
            ``MaxDepthExceededError`` (it is not a display depth like
            the ``max_depth`` of ``summarize``). Must be positive.
            See the "Common options" section of the equality user
            guide. Defaults to 100.
        reports: ``None`` (the default) if the comparison does not
            explain the difference: nothing is formatted. Otherwise a
            list that receives, innermost first, the ``(message, path)``
            of the difference found by the engine; ``compare`` and
            ``assert_*`` use it.

    Raises:
        TypeError: if ``atol`` or ``rtol`` is not a real number (a
            ``bool`` is rejected), or if ``max_depth`` is not an ``int``
            (a ``bool`` is rejected).
        ValueError: if ``atol`` or ``rtol`` is negative or not finite, or if
            ``max_depth`` is not positive.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> config = EqualityConfig()
        >>> config
        EqualityConfig(registry=EqualityTesterRegistry(...), equal_nan=False, atol=0.0, rtol=0.0, max_depth=100)

        ```
    """

    registry: EqualityTesterRegistry = field(default_factory=create_default_registry)
    equal_nan: bool = False
    atol: float = 0.0
    rtol: float = 0.0
    max_depth: int = 100
    reports: list[tuple[str, tuple[object, ...]]] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        validate_number(self.atol, name="atol")
        validate_number(self.rtol, name="rtol")
        validate_int(self.max_depth, name="max_depth")
        validate_non_negative(self.atol, name="atol")
        validate_non_negative(self.rtol, name="rtol")
        validate_positive(self.max_depth, name="max_depth")
        validate_finite(self.atol, name="atol")
        validate_finite(self.rtol, name="rtol")

    @property
    def explaining(self) -> bool:
        r"""Indicate if the comparison explains the difference.

        Returns:
            ``True`` if ``reports`` is a list, i.e. the testers must
            build the description of the difference.
        """
        return self.reports is not None
