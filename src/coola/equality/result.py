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

from coola.equality.config import EqualityConfig
from coola.equality.handler.utils import AttributeName
from coola.equality.interface import _compare
from coola.equality.tester.interface import get_default_registry
from coola.utils.safe import safe_repr

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
            formatted as ``.name``. Empty if equal, or if the
            difference is at the root.
        reason: A description of the first difference found, or
            ``None`` if the objects are equal.
        actual: The ``repr`` of the actual object if the objects are
            different (a placeholder if its ``__repr__`` raises),
            otherwise an empty string. The ``repr`` is not
            computed for equal objects because it can be expensive for
            large objects.
        expected: The ``repr`` of the expected object if the objects
            are different, otherwise an empty string.

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
    actual: str = ""
    expected: str = ""

    def __bool__(self) -> bool:
        return self.equal

    def format_path(self, root: str = "") -> str:
        r"""Format the path as an indexing expression.

        Args:
            root: The name prepended to the path.

        Returns:
            The formatted path, e.g. ``data['a'][2]``.
        """
        return root + "".join(
            f".{key}" if isinstance(key, AttributeName) else f"[{key!r}]" for key in self.path
        )

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


def compare(
    actual: object,
    expected: object,
    *,
    equal_nan: bool = False,
    atol: float = 0.0,
    rtol: float = 0.0,
    max_depth: int = 100,
    registry: EqualityTesterRegistry | None = None,
) -> ComparisonResult:
    r"""Compare two objects and explain the first difference.

    Exact equality is used by default. Set ``atol`` and/or ``rtol`` to
    compare within a tolerance (as ``objects_are_allclose`` does).
    Comparison stops at the first difference.

    Note:
        The difference is collected from the reports the equality
        handlers send to the ``reporter`` of the comparison
        configuration (see ``report_difference``). Nothing is logged,
        and no global state is modified, so this function can be called
        concurrently from several threads. Custom handlers that write
        to a logger directly instead of calling ``report_difference``
        are not captured, and the result then has no explanation.

    Args:
        actual: The actual object.
        expected: The expected object.
        equal_nan: If ``True``, treat two ``NaN`` values as equal.
        atol: The absolute tolerance. Must be non-negative.
        rtol: The relative tolerance. Must be non-negative.
        max_depth: Maximum recursion depth for nested comparisons.
        registry: Registry used to resolve type-specific equality
            testers. If ``None``, the default registry is used.

    Returns:
        The comparison result.

    Raises:
        ValueError: If ``atol`` or ``rtol`` is negative.

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

        ```
    """
    if registry is None:
        registry = get_default_registry()
    config = EqualityConfig(
        registry=registry,
        show_difference=True,
        equal_nan=equal_nan,
        atol=atol,
        rtol=rtol,
        max_depth=max_depth,
    )
    reports: list[tuple[str, tuple[object, ...]]] = []
    config.reporter = lambda message, path: reports.append((message, path))
    equal = _compare(registry, actual, expected, config)
    if equal:
        # The ``repr`` of the objects is not computed for equal objects, because it
        # is expensive for large objects and ``compare`` is used for each passing
        # assertion in the tests (see ``assert_objects_equal``).
        return ComparisonResult(True)
    # Handlers report innermost differences first: the outermost path
    # component is reported last.
    path: list[object] = []
    for _, location in reversed(reports):
        path.extend(location)
    reason = reports[0][0] if reports else None
    return ComparisonResult(
        False, tuple(path), reason, actual=safe_repr(actual), expected=safe_repr(expected)
    )


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
        max_depth: Maximum recursion depth for nested comparisons.
        registry: Registry used to resolve type-specific equality
            testers. If ``None``, the default registry is used.
        root: The name used for the root of the path in the message.

    Raises:
        AssertionError: If the objects are not equal within the
            tolerances. The message contains the path to the first
            difference.
        ValueError: If ``rtol`` or ``atol`` is negative.

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
        equal_nan=equal_nan,
        atol=atol,
        rtol=rtol,
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
        max_depth: Maximum recursion depth for nested comparisons.
        registry: Registry used to resolve type-specific equality
            testers. If ``None``, the default registry is used.
        root: The name used for the root of the path in the message.

    Raises:
        AssertionError: If the objects are not equal. The message
            contains the path to the first difference.

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
