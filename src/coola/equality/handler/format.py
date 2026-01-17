r"""Implement utilities for formatting equality differences."""

from __future__ import annotations

__all__ = [
    "format_mapping_difference",
    "format_sequence_difference",
    "format_shape_difference",
    "format_type_difference",
    "format_value_difference",
]

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


def _format_value(value: Any, max_length: int = 100) -> str:
    """Format a value for display, truncating if too long.

    Args:
        value: The value to format.
        max_length: Maximum length for the formatted string.

    Returns:
        Formatted value string.
    """
    formatted = str(value)
    if len(formatted) > max_length:
        return formatted[: max_length - 3] + "..."
    return formatted


def format_mapping_difference(
    actual: Mapping[Any, Any],
    expected: Mapping[Any, Any],
    *,
    missing_keys: set[Any] | None = None,
    additional_keys: set[Any] | None = None,
    different_value_key: Any | None = None,
) -> str:
    r"""Format a user-friendly difference message for mappings.

    Args:
        actual: The actual mapping.
        expected: The expected mapping.
        missing_keys: Keys present in actual but not in expected.
        additional_keys: Keys present in expected but not in actual.
        different_value_key: A key with different values in both mappings.

    Returns:
        A formatted difference message.

    Example:
        ```pycon
        >>> from coola.equality.handler.format import format_mapping_difference
        >>> msg = format_mapping_difference(
        ...     {"a": 1, "b": 2},
        ...     {"a": 1, "c": 3},
        ...     missing_keys={"b"},
        ...     additional_keys={"c"},
        ... )
        >>> print(msg)
        mappings have different keys:
          missing keys    : ['b']
          additional keys : ['c']

        ```
    """
    lines = []

    if missing_keys or additional_keys:
        lines.append("mappings have different keys:")
        if missing_keys:
            lines.append(f"  missing keys    : {sorted(missing_keys)}")
        if additional_keys:
            lines.append(f"  additional keys : {sorted(additional_keys)}")
    elif different_value_key is not None:
        # Just show which key has different values, not the full objects
        lines.append(f"mappings have different values for key {different_value_key!r}")

    return "\n".join(lines)


def format_sequence_difference(
    actual: Sequence[Any],
    expected: Sequence[Any],
    *,
    different_index: int | None = None,
) -> str:
    r"""Format a user-friendly difference message for sequences.

    Args:
        actual: The actual sequence.
        expected: The expected sequence.
        different_index: Index where values differ.

    Returns:
        A formatted difference message.

    Example:
        ```pycon
        >>> from coola.equality.handler.format import format_sequence_difference
        >>> msg = format_sequence_difference([1, 2, 3], [1, 2, 4], different_index=2)
        >>> print(msg)
        sequences have different values at index 2

        ```
    """
    lines = []

    if different_index is not None:
        # Just show the index, not the full values which could be large objects
        lines.append(f"sequences have different values at index {different_index}")
    elif len(actual) != len(expected):
        lines.append(f"sequences have different lengths: {len(actual)} vs {len(expected)}")
    else:
        lines.append("sequences have different values")

    return "\n".join(lines)


def format_shape_difference(
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> str:
    r"""Format a user-friendly difference message for shapes.

    Args:
        actual_shape: The actual shape.
        expected_shape: The expected shape.

    Returns:
        A formatted difference message.

    Example:
        ```pycon
        >>> from coola.equality.handler.format import format_shape_difference
        >>> msg = format_shape_difference((2, 3), (2, 4))
        >>> print(msg)
        objects have different shapes:
          actual   : (2, 3)
          expected : (2, 4)

        ```
    """
    return (
        f"objects have different shapes:\n"
        f"  actual   : {actual_shape}\n"
        f"  expected : {expected_shape}"
    )


def format_type_difference(
    actual_type: type,
    expected_type: type,
) -> str:
    r"""Format a user-friendly difference message for types.

    Args:
        actual_type: The actual type.
        expected_type: The expected type.

    Returns:
        A formatted difference message.

    Example:
        ```pycon
        >>> from coola.equality.handler.format import format_type_difference
        >>> msg = format_type_difference(list, tuple)
        >>> print(msg)
        objects have different types:
          actual   : <class 'list'>
          expected : <class 'tuple'>

        ```
    """
    return (
        f"objects have different types:\n  actual   : {actual_type}\n  expected : {expected_type}"
    )


def format_value_difference(
    actual: Any,
    expected: Any,
    *,
    name: str = "objects",
) -> str:
    r"""Format a user-friendly difference message for values.

    Args:
        actual: The actual value.
        expected: The expected value.
        name: The name to use in the message (e.g., "numbers", "arrays").

    Returns:
        A formatted difference message.

    Example:
        ```pycon
        >>> from coola.equality.handler.format import format_value_difference
        >>> msg = format_value_difference(1, 2, name="numbers")
        >>> print(msg)
        numbers are different:
          actual   : 1
          expected : 2

        ```
    """
    return f"{name} are different:\n  actual   : {actual}\n  expected : {expected}"
