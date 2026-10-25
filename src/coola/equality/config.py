r"""Define the equality configuration."""

from __future__ import annotations

__all__ = ["EqualityConfig"]

from contextlib import contextmanager
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
    from collections.abc import Generator

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
        max_depth: The nesting limit: the maximum nesting level of the
            compared objects. A deeper structure raises
            ``MaxDepthExceededError`` (it is not a display depth like
            the ``max_depth`` of ``summarize``). Must be positive.
            See the "Common options" section of the equality user
            guide. Defaults to 100.
        reports: ``None`` (the default) if the comparison does not
            explain the difference: nothing is formatted. Otherwise a
            list that receives, innermost first, the ``(message, path)``
            of the differences found by the testers (see
            ``report_difference``); ``compare`` and ``assert_*`` use it.

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
    # With the default Python recursion limit (1000 frames) and about 6 frames per
    # nesting level, the interpreter limit is reached at roughly 150 levels, so the
    # default is below it and the clear ``MaxDepthExceededError`` comes first (the
    # cycles do not need it, they are detected with ``_in_progress``).
    max_depth: int = 100
    reports: list[tuple[str, tuple[object, ...]]] | None = field(
        default=None, repr=False, compare=False
    )
    # Internal state of one comparison: a config must not be shared by
    # concurrent comparisons (each ``objects_are_*`` call creates its own).
    _current_depth: int = field(default=0, init=False, repr=False, compare=False)
    # The names of the ignored settings (``equal_nan``, ``atol``, ``rtol``) that were
    # already reported in a warning, so a comparison warns once per setting and not for
    # each object (see ``_warn_ignored`` in ``coola.equality.handler.pyarrow``).
    _warned: set[str] = field(default_factory=set, init=False, repr=False, compare=False)
    # ``(id(actual), id(expected))`` of the pairs being compared, used to detect
    # the cycles (see ``track_comparison``). The ids are safe to use as keys
    # because the pairs are alive during the whole comparison.
    _in_progress: set[tuple[int, int]] = field(
        default_factory=set, init=False, repr=False, compare=False
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

    @property
    def depth(self) -> int:
        r"""The current recursion depth counter.

        Returns:
            The current depth, which is ``0`` at the root of the
            comparison.
        """
        return self._current_depth

    @depth.setter
    def depth(self, value: int) -> None:
        self._current_depth = value

    def increment_depth(self) -> None:
        """Increment the current depth counter.

        The handlers that recurse call it before they compare the
        children, and ``decrement_depth`` after.
        """
        self._current_depth += 1

    def decrement_depth(self) -> None:
        """Decrement the current depth counter.

        It is the inverse of ``increment_depth``.
        """
        self._current_depth -= 1

    def mark_warned(self, name: str) -> bool:
        """Record that a warning was emitted for a setting.

        The handlers use it to warn once per setting and per
        comparison, and not for each object of a nested structure.

        Args:
            name: The name of the setting the warning is about.

        Returns:
            ``True`` if this is the first call with this name, so the
                handler should warn, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.equality.config import EqualityConfig
            >>> config = EqualityConfig()
            >>> config.mark_warned("atol")
            True
            >>> config.mark_warned("atol")
            False
            >>> config.mark_warned("rtol")
            True

            ```
        """
        if name in self._warned:
            return False
        self._warned.add(name)
        return True

    @contextmanager
    def visiting(self, actual: object, expected: object) -> Generator[bool]:
        """Context manager to detect the cycles in the objects to
        compare.

        It records that ``actual`` and ``expected`` are being compared.
        If this pair is already being compared higher in the call
        stack, the objects contain a cycle. See ``track_comparison``.

        Args:
            actual: The actual object.
            expected: The expected object.

        Yields:
            ``True`` if the pair is already being compared, otherwise
                ``False``.

        Example:
            ```pycon
            >>> from coola.equality.config import EqualityConfig
            >>> config = EqualityConfig()
            >>> x, y = [], []
            >>> with config.visiting(x, y) as in_progress:
            ...     print(in_progress)
            ...     with config.visiting(x, y) as in_progress:
            ...         print(in_progress)
            ...
            False
            True

            ```
        """
        key = (id(actual), id(expected))
        if key in self._in_progress:
            yield True
            return
        self._in_progress.add(key)
        try:
            yield False
        finally:
            self._in_progress.discard(key)
