r"""Implement utilities for formatting equality differences."""

from __future__ import annotations

__all__ = [
    "format_mapping_difference",
    "format_sequence_difference", 
    "format_shape_difference",
    "format_type_difference",
    "format_value_difference",
    "format_difference_with_path",
]

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


def format_difference_with_path(
    path: list[str],
    actual: Any,
    expected: Any,
    *,
    name: str = "objects",
) -> str:
    r"""Format a difference message with path information (top-down).

    Args:
        path: List of path elements from root to the difference location.
        actual: The actual value.
        expected: The expected value.
        name: The name to use for the differing values.

    Returns:
        A formatted difference message showing path from top to bottom.

    Example:
        ```pycon
        >>> from coola.equality.format import format_difference_with_path
        >>> msg = format_difference_with_path(
        ...     ["[key 'users']", "[index 1]", "[key 'score']"],
        ...     87,
        ...     88,
        ...     name="values",
        ... )
        >>> print(msg)
        objects are different
          [key 'users']
          [index 1]
          [key 'score']
        values are different:
          actual   : 87
          expected : 88

        ```
    """
    lines = []
    
    # Show the path from top to bottom
    if path:
        lines.append("objects are different")
        for element in path:
            lines.append(f"  {element}")
    
    # Show the actual difference
    lines.append(f"{name} are different:")
    lines.append(f"  actual   : {_format_value(actual)}")
    lines.append(f"  expected : {_format_value(expected)}")
    
    return "\n".join(lines)


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
        return formatted[:max_length - 3] + "..."
    return formatted


def format_mapping_difference(
    actual: Mapping[Any, Any],
    expected: Mapping[Any, Any],
    *,
    missing_keys: set[Any] | None = None,
    additional_keys: set[Any] | None = None,
    different_value_key: Any | None = None,
    path: list[str] | None = None,
) -> str:
    r"""Format a user-friendly difference message for mappings.

    Args:
        actual: The actual mapping.
        expected: The expected mapping.
        missing_keys: Keys present in actual but not in expected.
        additional_keys: Keys present in expected but not in actual.
        different_value_key: A key with different values in both mappings.
        path: Optional path elements from root to current location.

    Returns:
        A formatted difference message.

    Example:
        ```pycon
        >>> from coola.equality.format import format_mapping_difference
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
    
    # Show path if provided
    if path:
        lines.append("objects are different")
        for element in path:
            lines.append(f"  {element}")
    
    # Show the type of difference
    lines.append("mappings have different keys:" if missing_keys or additional_keys 
             else "mappings have different values:")
    
    if missing_keys:
        lines.append(f"  missing keys    : {sorted(missing_keys)}")
    if additional_keys:
        lines.append(f"  additional keys : {sorted(additional_keys)}")
    if different_value_key is not None:
        actual_val = _format_value(actual.get(different_value_key, "<not found>"))
        expected_val = _format_value(expected.get(different_value_key, "<not found>"))
        lines.append(f"  different value for key '{different_value_key}':")
        lines.append(f"    actual   : {actual_val}")
        lines.append(f"    expected : {expected_val}")
    
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
        >>> from coola.equality.format import format_sequence_difference
        >>> msg = format_sequence_difference([1, 2, 3], [1, 2, 4], different_index=2)
        >>> print(msg)
        sequences have different values:
          different value at index 2:
            actual   : 3
            expected : 4

        ```
    """
    lines = ["sequences have different values:"]
    
    if len(actual) != len(expected):
        lines.append(f"  lengths: {len(actual)} vs {len(expected)}")
    
    if different_index is not None:
        actual_val = actual[different_index] if different_index < len(actual) else "<out of bounds>"
        expected_val = expected[different_index] if different_index < len(expected) else "<out of bounds>"
        lines.append(f"  different value at index {different_index}:")
        lines.append(f"    actual   : {actual_val}")
        lines.append(f"    expected : {expected_val}")
    
    return "\n".join(lines)


def format_shape_difference(
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    *,
    path: list[str] | None = None,
) -> str:
    r"""Format a user-friendly difference message for shapes.

    Args:
        actual_shape: The actual shape.
        expected_shape: The expected shape.
        path: Optional path elements from root to current location.

    Returns:
        A formatted difference message.

    Example:
        ```pycon
        >>> from coola.equality.format import format_shape_difference
        >>> msg = format_shape_difference((2, 3), (2, 4))
        >>> print(msg)
        objects have different shapes:
          actual   : (2, 3)
          expected : (2, 4)

        ```
    """
    lines = []
    
    # Show path if provided
    if path:
        lines.append("objects are different")
        for element in path:
            lines.append(f"  {element}")
    
    lines.append("objects have different shapes:")
    lines.append(f"  actual   : {actual_shape}")
    lines.append(f"  expected : {expected_shape}")
    
    return "\n".join(lines)


def format_type_difference(
    actual_type: type,
    expected_type: type,
    *,
    path: list[str] | None = None,
) -> str:
    r"""Format a user-friendly difference message for types.

    Args:
        actual_type: The actual type.
        expected_type: The expected type.
        path: Optional path elements from root to current location.

    Returns:
        A formatted difference message.

    Example:
        ```pycon
        >>> from coola.equality.format import format_type_difference
        >>> msg = format_type_difference(list, tuple)
        >>> print(msg)
        objects have different types:
          actual   : <class 'list'>
          expected : <class 'tuple'>

        ```
    """
    lines = []
    
    # Show path if provided
    if path:
        lines.append("objects are different")
        for element in path:
            lines.append(f"  {element}")
    
    lines.append("objects have different types:")
    lines.append(f"  actual   : {actual_type}")
    lines.append(f"  expected : {expected_type}")
    
    return "\n".join(lines)


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
        >>> from coola.equality.format import format_value_difference
        >>> msg = format_value_difference(1, 2, name="numbers")
        >>> print(msg)
        numbers are different:
          actual   : 1
          expected : 2

        ```
    """
    return (
        f"{name} are different:\n"
        f"  actual   : {actual}\n"
        f"  expected : {expected}"
    )
