r"""Implement a structured comparison result and an assertion helper."""

from __future__ import annotations

__all__ = [
    "ComparisonResult",
    "assert_objects_allclose",
    "assert_objects_equal",
    "compare",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from coola.equality._helpers.utils import AttributeName
from coola.equality.config import EqualityConfig
from coola.equality.engine import Element, Engine
from coola.equality.tester.interface import get_default_registry
from coola.equality.tolerance import Tolerance

if TYPE_CHECKING:
    from coola.equality.tester.registry import EqualityTesterRegistry


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    r"""Describe the outcome of comparing two objects.

    ``bool(result)`` is ``True`` if the objects are equal, so the result
    can be used as a drop-in replacement for the ``bool`` returned by
    ``objects_are_equal``.

    Attributes:
        equal: ``True`` if the objects are equal.
        path: The location of the first difference, e.g. ``("a", 2)``
            for ``actual["a"][2]``. The attributes (the fields of a
            dataclass) are ``AttributeName`` instances and are
            formatted as ``.name``, and the elements of a set are
            ``Element`` instances formatted as ``{value}``. Empty if equal, or if the
            difference is at the root.
        reason: A description of the first difference found, or
            ``None`` if the objects are equal.

    Example:
        ```pycon
        >>> from coola.equality import compare
        >>> result = compare({"a": [1, 2, 3]}, {"a": [1, 2, 4]})
        >>> bool(result)
        False
        >>> result.path
        ('a', 2)
        >>> result.format_path()
        "['a'][2]"

        ```
    """

    equal: bool
    path: tuple[object, ...] = ()
    reason: str | None = None

    def __bool__(self) -> bool:
        r"""Return the verdict.

        Returns:
            ``True`` if the objects are equal, otherwise ``False``.
        """
        return self.equal

    def format_path(self, root: str = "") -> str:
        r"""Format the path as an indexing expression.

        Args:
            root: The name prepended to the path.

        Returns:
            The formatted path, e.g. ``data['a'][2]``.
        """
        return root + "".join(_format_part(key) for key in self.path)

    def format_message(self, root: str = "actual") -> str:
        r"""Format a human-readable description of the result.

        Args:
            root: The name used for the root of the path.

        Returns:
            The message.
        """
        if self.equal:
            return "objects are equal"
        lines = [f"objects are not equal at {self.format_path(root)}"]
        if self.reason:
            lines.append(self.reason)
        return "\n".join(lines)


def _format_part(key: object) -> str:
    r"""Format one part of a path.

    Args:
        key: The part: an attribute name, a set element or an index/key.

    Returns:
        ``.name`` for an attribute, ``{element}`` for a set element,
            otherwise ``[key]``.
    """
    if isinstance(key, AttributeName):
        return f".{key}"
    if isinstance(key, Element):
        return f"{{{key.value!r}}}"
    return f"[{key!r}]"


def compare(
    actual: object,
    expected: object,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    equal_nan: bool = False,
    max_depth: int = 100,
    registry: EqualityTesterRegistry | None = None,
) -> ComparisonResult:
    r"""Compare two objects and explain the first difference.

    Exact equality is used by default. Set ``rtol`` or ``atol`` to
    compare within a tolerance: ``rtol=1e-5`` and ``atol=1e-8`` are the
    defaults of ``objects_are_allclose``. Comparison stops at the first
    difference, and only this difference is explained.

    Note:
        The difference is the first failure recorded by the engine
        (``ctx.fail`` of a tester).
        Nothing is logged, and no global state is modified, so this
        function can be called concurrently from several threads.

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
        registry: Registry used to resolve type-specific equality
            testers. If ``None``, the default registry is used.

    Returns:
        The comparison result.

    Raises:
        ValueError: If ``rtol`` or ``atol`` is negative, or if
            ``max_depth`` is not positive.
        MaxDepthExceededError: If the nesting of the objects exceeds
            ``max_depth``.

    Example:
        ```pycon
        >>> from coola.equality import compare
        >>> compare([1, 2], [1, 2]).equal
        True
        >>> result = compare({"a": 1}, {"a": 2})
        >>> result.path
        ('a',)
        >>> result.equal
        False
        >>> compare(1.0, 1.0 + 1e-9).equal
        False
        >>> compare(1.0, 1.0 + 1e-9, rtol=1e-5, atol=1e-8).equal
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
    engine = Engine(config, explaining=True)
    if engine.run(actual, expected):
        return ComparisonResult(True)
    path, reason = engine.failure or ((), None)
    return ComparisonResult(False, path, reason)


def assert_objects_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    max_depth: int = 100,
    registry: EqualityTesterRegistry | None = None,
    root: str = "actual",
) -> None:
    r"""Assert that two objects are equal within a tolerance.

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
        registry: Registry used to resolve type-specific equality
            testers. If ``None``, the default registry is used.
        root: The name used for the root of the path in the message.

    Raises:
        AssertionError: If the objects are not equal within the
            tolerances. The message contains the path to the first
            difference.
        ValueError: If ``rtol`` or ``atol`` is negative, or if
            ``max_depth`` is not positive.
        MaxDepthExceededError: If the nesting of the objects exceeds
            ``max_depth``.

    Example:
        ```pycon
        >>> from coola.equality import assert_objects_allclose
        >>> assert_objects_allclose({"a": [1.0, 2.0]}, {"a": [1.0, 2.0 + 1e-9]})
        >>> assert_objects_allclose([1.0], [1.5], atol=0.1, root="data")
        Traceback (most recent call last):
            ...
        AssertionError: objects are not equal at data[0]
        numbers are different:
          actual   : 1.0
          expected : 1.5

        ```
    """
    result = compare(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        max_depth=max_depth,
        registry=registry,
    )
    if not result:
        raise AssertionError(result.format_message(root))


def assert_objects_equal(
    actual: object,
    expected: object,
    *,
    equal_nan: bool = False,
    max_depth: int = 100,
    registry: EqualityTesterRegistry | None = None,
    root: str = "actual",
) -> None:
    r"""Assert that two objects are equal.

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
        registry: Registry used to resolve type-specific equality
            testers. If ``None``, the default registry is used.
        root: The name used for the root of the path in the message.

    Raises:
        AssertionError: If the objects are not equal. The message
            contains the path to the first difference.
        ValueError: If ``max_depth`` is not positive.
        MaxDepthExceededError: If the nesting of the objects exceeds
            ``max_depth``.

    Example:
        ```pycon
        >>> from coola.equality import assert_objects_equal
        >>> assert_objects_equal({"a": [1, 2]}, {"a": [1, 2]})
        >>> assert_objects_equal({"a": [1, 2]}, {"a": [1, 3]}, root="data")
        Traceback (most recent call last):
            ...
        AssertionError: objects are not equal at data['a'][1]
        numbers are different:
          actual   : 2
          expected : 3

        ```
    """
    result = compare(
        actual,
        expected,
        equal_nan=equal_nan,
        max_depth=max_depth,
        registry=registry,
    )
    if not result:
        raise AssertionError(result.format_message(root))
