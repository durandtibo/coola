r"""Define the equality configuration."""

from __future__ import annotations

__all__ = ["EqualityConfig"]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from coola.validation import validate_finite, validate_non_negative, validate_positive

if TYPE_CHECKING:
    from collections.abc import Callable

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
            comparisons. Must be positive. Defaults to 100.
            Set to a lower value to protect against stack overflow
            with extremely deeply nested structures.
        reporter: An optional callable that receives the differences
            found while ``show_difference`` is ``True``. It is called
            with the message and the location of the difference in the
            compared objects (relative to the current nesting level).
            When it is set, the differences are sent to it instead of
            being logged. Defaults to ``None`` (log the differences).

    Raises:
        ValueError: if ``atol`` or ``rtol`` is negative or not finite, or if
            ``max_depth`` is not positive.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> config = EqualityConfig()
        >>> config
        EqualityConfig(registry=EqualityTesterRegistry(...), equal_nan=False, atol=0.0, rtol=0.0, show_difference=False, max_depth=100)

        ```
    """

    registry: EqualityTesterRegistry = field(default_factory=create_default_registry)
    equal_nan: bool = False
    atol: float = 0.0
    rtol: float = 0.0
    show_difference: bool = False
    # With the default Python recursion limit (1000 frames) and about 6 frames per
    # nesting level, the interpreter limit is reached at roughly 150 levels, so the
    # default is below it and the clear ``MaxDepthExceededError`` comes first (the
    # cycles do not need it, they are detected with ``_in_progress``).
    max_depth: int = 100
    reporter: Callable[[str, tuple[object, ...]], None] | None = field(
        default=None, repr=False, compare=False
    )
    # Internal state of one comparison: a config must not be shared by
    # concurrent comparisons (each ``objects_are_*`` call creates its own).
    _current_depth: int = field(default=0, init=False, repr=False, compare=False)
    # The names of the ignored settings (``equal_nan``, ``atol``, ``rtol``) that were
    # already reported in a warning, so a comparison warns once per setting and not for
    # each object (see ``warn_ignored_setting``).
    _warned: set[str] = field(default_factory=set, init=False, repr=False, compare=False)
    # ``(id(actual), id(expected))`` of the pairs being compared, used to detect
    # the cycles (see ``track_comparison``). The ids are safe to use as keys
    # because the pairs are alive during the whole comparison.
    _in_progress: set[tuple[int, int]] = field(
        default_factory=set, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        validate_non_negative(self.atol, name="atol")
        validate_non_negative(self.rtol, name="rtol")
        validate_positive(self.max_depth, name="max_depth")
        validate_finite(self.atol, name="atol")
        validate_finite(self.rtol, name="rtol")

    @property
    def depth(self) -> int:
        r"""The current recursion depth counter."""
        return self._current_depth

    def increment_depth(self) -> None:
        """Increment the current depth counter."""
        self._current_depth += 1

    def decrement_depth(self) -> None:
        """Decrement the current depth counter."""
        self._current_depth -= 1
