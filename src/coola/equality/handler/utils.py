r"""Contain utilities for handlers."""

from __future__ import annotations

__all__ = [
    "AttributeName",
    "MaxDepthExceededError",
    "check_recursion_depth",
    "create_chain",
    "handlers_are_equal",
    "report_difference",
    "track_comparison",
]

from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import logging
    from collections.abc import Generator

    from coola.equality.config import EqualityConfig
    from coola.equality.handler.base import BaseEqualityHandler


class AttributeName(str):
    r"""Name of an attribute in the location of a difference.

    It is a ``str`` that marks a path component as an attribute (a
    dataclass field, for example) rather than a key or an index, so a
    path is formatted as ``['a'][2].x`` and not as ``['a'][2]['x']``.

    Example:
        ```pycon
        >>> from coola.equality.handler.utils import AttributeName
        >>> AttributeName("x") == "x"
        True

        ```
    """

    __slots__ = ()


class MaxDepthExceededError(RecursionError):
    r"""Raised when the nesting of the compared objects exceeds the
    ``max_depth`` of the equality configuration.

    It derives from ``RecursionError`` for backward compatibility. It
    allows telling this error apart from the interpreter's own stack
    overflow, which is also a ``RecursionError``.
    """


def report_difference(
    logger: logging.Logger,
    config: EqualityConfig,
    msg: str,
    *args: object,
    extra: dict[str, object] | None = None,
) -> None:
    r"""Report a difference found while comparing two objects.

    If the configuration has a ``reporter``, the formatted message is
    sent to it together with the location of the difference, and
    nothing is logged. Otherwise, the message is logged at the ``INFO``
    level with ``logger``.

    Handlers should call this function (instead of ``logger.info``) when
    ``config.show_difference`` is ``True``.

    Args:
        logger: The logger used when there is no reporter.
        config: The equality configuration.
        msg: The message, optionally with ``%`` placeholders.
        *args: The values for the placeholders of ``msg``.
        extra: Extra information about the difference. The
            ``"coola_path"`` entry is a tuple with the location of the
            difference in the compared objects, relative to the current
            level of nesting. The components that are attributes are
            ``AttributeName`` instances.

    Example:
        ```pycon
        >>> import logging
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import report_difference
        >>> reports = []
        >>> config = EqualityConfig(reporter=lambda msg, path: reports.append((msg, path)))
        >>> report_difference(
        ...     logging.getLogger(), config, "different %s", "values", extra={"coola_path": ("a",)}
        ... )
        >>> reports
        [('different values', ('a',))]

        ```
    """
    if config.reporter is None:
        logger.info(msg, *args, extra=extra)
        return
    path = cast("tuple[object, ...]", (extra or {}).get("coola_path", ()))
    config.reporter(msg % args if args else msg, path)


def create_chain(*handlers: BaseEqualityHandler) -> BaseEqualityHandler:
    r"""Create a chain of handlers and return the first handler.

    Args:
        handlers: Handlers to chain.

    Returns:
        The first handler of the chain.

    Raises:
        ValueError: If no handler is provided.

    Example:
        ```pycon
        >>> from coola.equality.handler import (
        ...     create_chain,
        ...     SameObjectHandler,
        ...     SameTypeHandler,
        ...     ObjectEqualHandler,
        ... )
        >>> handler = create_chain(SameObjectHandler(), SameTypeHandler(), ObjectEqualHandler())
        >>> print(handler.visualize_chain())
        (0): SameObjectHandler()
        (1): SameTypeHandler()
        (2): ObjectEqualHandler()

        ```
    """
    if not handlers:
        msg = "At least one handler is required to create a chain."
        raise ValueError(msg)
    handler = handlers[0]
    handler.chain_all(*handlers[1:])
    return handler


def handlers_are_equal(
    handler1: BaseEqualityHandler | None, handler2: BaseEqualityHandler | None
) -> bool:
    r"""Indicate whether two handlers are equal.

    Args:
        handler1: The first handler.
        handler2: The second handler.

    Returns:
        ``True`` if both handlers are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.equality.handler import SameObjectHandler, FalseHandler, handlers_are_equal
        >>> handlers_are_equal(SameObjectHandler(), SameObjectHandler())
        True
        >>> handlers_are_equal(SameObjectHandler(), FalseHandler())
        False
        >>> handlers_are_equal(None, SameObjectHandler())
        False
        >>> handlers_are_equal(SameObjectHandler(), None)
        False
        >>> handlers_are_equal(None, None)
        True

        ```
    """
    if handler1 is None:
        return handler1 == handler2
    return handler1.equal(handler2)


@contextmanager
def check_recursion_depth(config: EqualityConfig) -> Generator[None]:
    r"""Context manager to track and enforce recursion depth limits.

    This context manager increments the recursion depth counter on entry
    and decrements it on exit (even if an exception occurs). It raises
    a RecursionError if the maximum depth is exceeded.

    Args:
        config: The equality configuration containing depth settings.

    Yields:
        ``None``.

    Raises:
        MaxDepthExceededError: if the current depth exceeds max_depth.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import check_recursion_depth
        >>> config = EqualityConfig(max_depth=5)
        >>> with check_recursion_depth(config):
        ...     print(config.depth)
        ...
        1

        ```
    """
    if config.depth >= config.max_depth:
        msg = (
            f"Maximum recursion depth ({config.max_depth}) exceeded. "
            f"Consider increasing max_depth parameter or simplifying the data structure."
        )
        raise MaxDepthExceededError(msg)
    config.increment_depth()
    try:
        yield
    finally:
        config.decrement_depth()


@contextmanager
def track_comparison(config: EqualityConfig, actual: object, expected: object) -> Generator[bool]:
    r"""Context manager to detect the cycles in the objects to compare.

    It records that ``actual`` and ``expected`` are being compared. If
    this pair is already being compared higher in the call stack, the
    objects contain a cycle (for example, a list that contains
    itself). The pair is then not recorded again and ``True`` is
    yielded: the handler should consider the pair as equal, because
    the cycle only repeats the comparison that is already in progress.

    Args:
        config: The equality configuration that stores the pairs.
        actual: The actual object.
        expected: The expected object.

    Yields:
        ``True`` if the pair is already being compared, otherwise
            ``False``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import track_comparison
        >>> config = EqualityConfig()
        >>> x, y = [], []
        >>> with track_comparison(config, x, y) as in_progress:
        ...     print(in_progress)
        ...     with track_comparison(config, x, y) as in_progress:
        ...         print(in_progress)
        ...
        False
        True

        ```
    """
    with config.visiting(actual, expected) as in_progress:
        yield in_progress
