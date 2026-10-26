r"""Define the public interface to compare nested data for equality."""

from __future__ import annotations

__all__ = ["objects_are_allclose", "objects_are_equal"]

from typing import TYPE_CHECKING

from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester.interface import get_default_registry
from coola.equality.tolerance import Tolerance

if TYPE_CHECKING:
    from coola.equality.tester.registry import EqualityTesterRegistry


def objects_are_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    max_depth: int = 100,
    registry: EqualityTesterRegistry | None = None,
) -> bool:
    r"""Indicate if two objects are equal within a tolerance.

    Note:
        The types are compared first, so ``1`` and ``1.0`` are different
        (as are ``[1]`` and ``[1.0]``), even with a tolerance. A boolean
        is a label, not a quantity: ``True`` and ``False`` are never
        close, even with ``atol=1`` (also for ``numpy.bool_`` and for
        boolean arrays).

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
        max_depth: The nesting limit: the maximum nesting level of the
            compared objects. A deeper structure raises
            ``MaxDepthExceededError`` (it is not a display depth like
            the ``max_depth`` of ``summarize``). Must be positive.
            See the "Common options" section of the equality user
            guide. Defaults to 100.
        registry: Registry used to resolve type-specific equality testers.
            If ``None``, the default registry is used.

    Returns:
        ``True`` if the two objects are equal within tolerances, otherwise
        ``False``.

    Raises:
        TypeError: If ``rtol`` or ``atol`` is not a real number, or if
            ``max_depth`` is not an ``int`` (a ``bool`` is rejected).
        ValueError: If ``rtol`` or ``atol`` is negative, or if
            ``max_depth`` is not positive.
        MaxDepthExceededError: If the nesting of the objects exceeds
            ``max_depth``.

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
        tolerance=Tolerance(rtol=rtol, atol=atol),
        equal_nan=equal_nan,
        max_depth=max_depth,
    )
    return Engine(config).run(actual, expected)


def objects_are_equal(
    actual: object,
    expected: object,
    *,
    equal_nan: bool = False,
    max_depth: int = 100,
    registry: EqualityTesterRegistry | None = None,
) -> bool:
    r"""Indicate if two objects are equal or not.

    Note:
        The types are compared first, so ``1`` and ``1.0`` are different
        (as are ``[1]`` and ``[1.0]``). There is no tolerance: use
        ``objects_are_allclose`` to compare numbers within a tolerance.

        An object is always equal to itself, even if it contains ``NaN``
        and ``equal_nan=False``: ``[n]`` equals ``[n]`` for
        ``n = float("nan")``, but not ``[float("nan")]``. Two distinct
        objects that contain ``NaN`` are equal only with
        ``equal_nan=True``.

    Args:
        actual: The actual object.
        expected: The expected object.
        equal_nan: If ``True``, treat two ``NaN`` values as equal.
        max_depth: The nesting limit: the maximum nesting level of the
            compared objects. A deeper structure raises
            ``MaxDepthExceededError`` (it is not a display depth like
            the ``max_depth`` of ``summarize``). Must be positive.
            See the "Common options" section of the equality user
            guide. Defaults to 100.
        registry: Registry used to resolve type-specific equality testers.
            If ``None``, the default registry is used.

    Returns:
        ``True`` if the two objects are exactly equal, otherwise ``False``.

    Raises:
        TypeError: If ``max_depth`` is not an ``int`` (a ``bool`` is
            rejected).
        ValueError: If ``max_depth`` is not positive.
        MaxDepthExceededError: If the nesting of the objects exceeds
            ``max_depth``.

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
    config = EqualityConfig(registry=registry, equal_nan=equal_nan, max_depth=max_depth)
    return Engine(config).run(actual, expected)
