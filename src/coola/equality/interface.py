r"""Define the public interface to compare nested data for equality."""

from __future__ import annotations

__all__ = ["objects_are_allclose", "objects_are_equal", "run_comparison"]

from typing import TYPE_CHECKING

from coola.equality.config import EqualityConfig
from coola.equality.handler.utils import MaxDepthExceededError
from coola.equality.tester.interface import get_default_registry
from coola.utils.recursion import raise_recursion_limit_error

if TYPE_CHECKING:
    from coola.equality.tester.registry import EqualityTesterRegistry


def run_comparison(
    registry: EqualityTesterRegistry, actual: object, expected: object, config: EqualityConfig
) -> bool:
    """Run the comparison and make interpreter stack overflows
    actionable.

    It is the code path shared by all the entry points
    (``objects_are_equal``, ``objects_are_allclose`` and ``compare``).
    Use it to run a comparison with an ``EqualityConfig`` that you
    built, for example with a ``reporter``.

    Args:
        registry: The registry that finds the tester of ``actual``.
        actual: The actual object.
        expected: The expected object.
        config: The settings of the comparison. A config holds the state
            of one comparison, so do not share it between comparisons.

    Returns:
        ``True`` if the objects are equal, otherwise ``False``.

    Raises:
        MaxDepthExceededError: If the nesting of the objects exceeds
            ``config.max_depth``.
        RecursionError: If the interpreter recursion limit is reached
            before ``config.max_depth``.

    The comparison recurses through several Python frames per nesting
    level, so the interpreter limit (``sys.getrecursionlimit()``) is
    usually reached well before ``config.max_depth``.
    """
    # Shared by all the entry points (including ``compare``) so they raise the same,
    # actionable error. The cycles do not reach this error, they are handled by
    # ``track_comparison``.
    try:
        return registry.objects_are_equal(actual, expected, config)
    except MaxDepthExceededError:
        raise  # already explains how to fix it
    except RecursionError as exc:
        raise_recursion_limit_error("comparing", exc)


def objects_are_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    show_difference: bool = False,
    max_depth: int = 100,
    registry: EqualityTesterRegistry | None = None,
) -> bool:
    r"""Indicate if two objects are equal within a tolerance.

    Note:
        The types are compared first, so ``1`` and ``1.0`` are different
        (as are ``[1]`` and ``[1.0]``), even with a tolerance. A ``bool``
        is an ``int``, so ``True`` and ``False`` are close with
        ``atol=1``, unlike ``numpy.bool_``.

        An object is always equal to itself, even if it contains ``NaN``
        and ``equal_nan=False``: ``[n]`` equals ``[n]`` for
        ``n = float("nan")``, but not ``[float("nan")]``. Two distinct
        objects that contain ``NaN`` are equal only with
        ``equal_nan=True``.

    Args:
        actual: The actual object.
        expected: The expected object.
        rtol: The relative tolerance parameter. Must be non-negative.
        atol: The absolute tolerance parameter. Must be non-negative.
        equal_nan: If ``True``, treat two ``NaN`` values as equal.
        show_difference: If ``True``, log a readable description when the
            comparison fails, at the ``INFO`` level on the
            ``coola.equality`` logger.
        max_depth: A limit that raises, not a display depth: the maximum
            recursion depth for nested comparisons (a deeper structure
            raises ``MaxDepthExceededError``, unlike the ``max_depth`` of
            ``summarize``, which truncates the output).
            Must be positive. Defaults to 100.
        registry: Registry used to resolve type-specific equality testers.
            If ``None``, the default registry is used.

    Returns:
        ``True`` if the two objects are equal within tolerances, otherwise
        ``False``.

    Raises:
        ValueError: If ``rtol`` or ``atol`` is negative.
        RecursionError: If recursion depth exceeds ``max_depth``.

    Example:
        ```pycon
        >>> from coola.equality import objects_are_allclose
        >>> objects_are_allclose([1.0, 2.0], [1.0 + 1e-8, 2.0])
        True
        >>> objects_are_allclose([1.0, 2.0], [1.0, 2.1], atol=1e-3, rtol=0.0)
        False
        >>> objects_are_allclose([float("nan")], [float("nan")], equal_nan=False)
        False
        >>> objects_are_allclose([float("nan")], [float("nan")], equal_nan=True)
        True

        ```
    """
    if registry is None:
        registry = get_default_registry()
    config = EqualityConfig(
        registry=registry,
        show_difference=show_difference,
        equal_nan=equal_nan,
        atol=atol,
        rtol=rtol,
        max_depth=max_depth,
    )
    return run_comparison(registry, actual, expected, config)


def objects_are_equal(
    actual: object,
    expected: object,
    *,
    equal_nan: bool = False,
    show_difference: bool = False,
    max_depth: int = 100,
    registry: EqualityTesterRegistry | None = None,
) -> bool:
    r"""Indicate if two objects are equal or not.

    Note:
        The types are compared first, so ``1`` and ``1.0`` are different
        (as are ``[1]`` and ``[1.0]``), even with a tolerance. A ``bool``
        is an ``int``, so ``True`` and ``False`` are close with
        ``atol=1``, unlike ``numpy.bool_``.

        An object is always equal to itself, even if it contains ``NaN``
        and ``equal_nan=False``: ``[n]`` equals ``[n]`` for
        ``n = float("nan")``, but not ``[float("nan")]``. Two distinct
        objects that contain ``NaN`` are equal only with
        ``equal_nan=True``.

    Args:
        actual: The actual object.
        expected: The expected object.
        equal_nan: If ``True``, treat two ``NaN`` values as equal.
        show_difference: If ``True``, log a readable description when the
            comparison fails, at the ``INFO`` level on the
            ``coola.equality`` logger.
        max_depth: A limit that raises, not a display depth: the maximum
            recursion depth for nested comparisons (a deeper structure
            raises ``MaxDepthExceededError``, unlike the ``max_depth`` of
            ``summarize``, which truncates the output).
            Must be positive. Defaults to 100.
        registry: Registry used to resolve type-specific equality testers.
            If ``None``, the default registry is used.

    Returns:
        ``True`` if the two objects are exactly equal, otherwise ``False``.

    Raises:
        RecursionError: If recursion depth exceeds ``max_depth``.

    Example:
        ```pycon
        >>> from coola.equality import objects_are_equal
        >>> objects_are_equal([1, {"a": 2}], [1, {"a": 2}])
        True
        >>> objects_are_equal([1, {"a": 2}], [1, {"a": 3}])
        False

        ```
    """
    if registry is None:
        registry = get_default_registry()
    config = EqualityConfig(
        registry=registry,
        show_difference=show_difference,
        equal_nan=equal_nan,
        max_depth=max_depth,
    )
    return run_comparison(registry, actual, expected, config)
