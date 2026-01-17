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
    lines = ["mappings have different keys:" if missing_keys or additional_keys 
             else "mappings have different values:"]
    
    if missing_keys:
        lines.append(f"  missing keys    : {sorted(missing_keys)}")
    if additional_keys:
        lines.append(f"  additional keys : {sorted(additional_keys)}")
    if different_value_key is not None:
        actual_val = actual.get(different_value_key, "<not found>")
        expected_val = expected.get(different_value_key, "<not found>")
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
) -> str:
    r"""Format a user-friendly difference message for shapes.

    Args:
        actual_shape: The actual shape.
        expected_shape: The expected shape.

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
        >>> from coola.equality.format import format_type_difference
        >>> msg = format_type_difference(list, tuple)
        >>> print(msg)
        objects have different types:
          actual   : <class 'list'>
          expected : <class 'tuple'>

        ```
    """
    return (
        f"objects have different types:\n"
        f"  actual   : {actual_type}\n"
        f"  expected : {expected_type}"
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
