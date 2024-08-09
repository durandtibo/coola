r"""Implement some utility functions to compute string representations
of objects."""

from __future__ import annotations

__all__ = [
    "find_best_byte_unit",
    "repr_indent",
    "repr_mapping",
    "repr_mapping_line",
    "repr_sequence",
    "repr_sequence_line",
    "repr_sequence_line",
    "str_human_byte_size",
    "str_indent",
    "str_mapping",
    "str_mapping_line",
    "str_sequence",
    "str_sequence_line",
    "str_time_human",
]

import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BYTE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024 * 1024,
    "GB": 1024 * 1024 * 1024,
    "TB": 1024 * 1024 * 1024 * 1024,
}


def repr_indent(original: Any, num_spaces: int = 2) -> str:
    r"""Add indentations if the original string is a multi-lines string.

    Args:
        original: The original string. If the inputis not a
            string, it will be converted to a string with the function
            ``repr``.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The indented string.

    Raises:
        RuntimeError: if num_spaces is not a positive integer.

    Example usage:

    ```pycon

    >>> from coola.utils.format import repr_indent
    >>> print(repr_indent("string1\nstring2\n  string3", 4))
    string1
    string2
      string3

    ```
    """
    if not isinstance(original, str):
        original = repr(original)
    return str_indent(original, num_spaces)


def repr_mapping(mapping: Mapping, sorted_keys: bool = False, num_spaces: int = 2) -> str:
    r"""Compute a string representation of a mapping.

    This function was designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
        mapping: The mapping.
        sorted_keys: If ``True``, the keys in the mapping are sorted
            before to compute the string representation.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The string representation of the mapping.

    Example usage:

    ```pycon

    >>> from coola.utils.format import repr_mapping
    >>> print(repr_mapping({"key1": "abc", "key2": "something\nelse"}))
    (key1): abc
    (key2): something
      else

    ```
    """
    lines = []
    for key, value in sorted(mapping.items()) if sorted_keys else mapping.items():
        lines.append(f"({key}): {repr_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)


def repr_mapping_line(mapping: Mapping, sorted_keys: bool = False, separator: str = ", ") -> str:
    r"""Compute a single line string representation of the given mapping.

    This function is designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
        mapping: The mapping.
        sorted_keys: If ``True``, the keys in the mapping are sorted
            before to compute the string representation.
        separator: The separator to use between each key-value pair.

    Returns:
        The string representation of the mapping.

    Example usage:

    ```pycon

    >>> from coola.utils.format import repr_mapping_line
    >>> repr_mapping_line({"key1": "abc", "key2": "meow", "key3": 42})
    key1='abc', key2='meow', key3=42

    ```
    """
    mapping = sorted(mapping.items()) if sorted_keys else mapping.items()
    return separator.join(f"{key}={value!r}" for key, value in mapping)


def repr_sequence(sequence: Sequence, num_spaces: int = 2) -> str:
    r"""Compute a string representation of a sequence.

    Args:
        sequence: The sequence.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The string representation of the sequence.

    Example usage:

    ```pycon

    >>> from coola.utils.format import repr_indent
    >>> print(repr_sequence(["abc", "something\nelse"]))
    (0): abc
    (1): something
      else

    ```
    """
    lines = []
    for i, item in enumerate(sequence):
        lines.append(f"({i}): {repr_indent(item, num_spaces=num_spaces)}")
    return "\n".join(lines)


def repr_sequence_line(sequence: Sequence, separator: str = ", ") -> str:
    r"""Compute a single line string representation of a sequence.

    Args:
        sequence: The sequence.
        separator: The separator to use between each item.

    Returns:
        The string representation of the sequence.

    Example usage:

    ```pycon

    >>> from coola.utils.format import repr_sequence_line
    >>> repr_sequence_line(["abc", "meow", 42])
    'abc', 'meow', 42

    ```
    """
    return separator.join(map(repr, sequence))


def str_indent(original: Any, num_spaces: int = 2) -> str:
    r"""Add indentations if the original string is a multi-lines string.

    Args:
        original: The original string. If the inputis not a
            string, it will be converted to a string with the function
            ``str``.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The indented string.

    Raises:
        RuntimeError: if num_spaces is not a positive integer.

    Example usage:

    ```pycon

    >>> from coola.utils.format import str_indent
    >>> print(str_indent("string1\nstring2\n  string3", 4))
    string1
    string2
      string3

    ```
    """
    if num_spaces < 0:
        msg = f"Incorrect num_spaces. Expected a positive integer but received {num_spaces}"
        raise RuntimeError(msg)
    formatted = str(original).split("\n")
    if len(formatted) == 1:  # single line
        return formatted[0]
    first = formatted.pop(0)
    formatted = "\n".join([(num_spaces * " ") + line for line in formatted])
    return first + "\n" + formatted


def str_mapping(mapping: Mapping, sorted_keys: bool = False, num_spaces: int = 2) -> str:
    r"""Compute a string representation of a mapping.

    This function was designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
        mapping: The mapping.
        sorted_keys: If ``True``, the keys in the mapping are sorted
            before to compute the string representation.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The string representation of the mapping.

    Example usage:

    ```pycon

    >>> from coola.utils.format import str_mapping
    >>> print(str_mapping({"key1": "abc", "key2": "something\nelse"}))
    (key1): abc
    (key2): something
      else

    ```
    """
    lines = []
    for key, value in sorted(mapping.items()) if sorted_keys else mapping.items():
        lines.append(f"({key}): {str_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)


def str_mapping_line(mapping: Mapping, sorted_keys: bool = False, separator: str = ", ") -> str:
    r"""Compute a single line string representation of the given mapping.

    This function is designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
        mapping: The mapping.
        sorted_keys: If ``True``, the keys in the mapping are sorted
            before to compute the string representation.
        separator: The separator to use between each key-value pair.

    Returns:
        The string representation of the mapping.

    Example usage:

    ```pycon

    >>> from coola.utils.format import str_mapping_line
    >>> str_mapping_line({"key1": "abc", "key2": "meow", "key3": 42})
    key1=abc, key2=meow, key3=42

    ```
    """
    mapping = sorted(mapping.items()) if sorted_keys else mapping.items()
    return separator.join(f"{key}={value!s}" for key, value in mapping)


def str_sequence(sequence: Sequence, num_spaces: int = 2) -> str:
    r"""Compute a string representation of a sequence.

    Args:
        sequence: The sequence.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The string representation of the sequence.

    Example usage:

    ```pycon

    >>> from coola.utils.format import str_sequence
    >>> print(str_sequence(["abc", "something\nelse"]))
    (0): abc
    (1): something
      else

    ```
    """
    lines = []
    for i, item in enumerate(sequence):
        lines.append(f"({i}): {str_indent(item, num_spaces=num_spaces)}")
    return "\n".join(lines)


def str_sequence_line(sequence: Sequence, separator: str = ", ") -> str:
    r"""Compute a single line string representation of a sequence.

    Args:
        sequence: The sequence.
        separator: The separator to use between each item.

    Returns:
        The string representation of the sequence.

    Example usage:

    ```pycon

    >>> from coola.utils.format import str_sequence_line
    >>> str_sequence_line(["abc", "meow", 42])
    abc, meow, 42

    ```
    """
    return separator.join(map(str, sequence))


def str_time_human(seconds: float) -> str:
    r"""Return a number of seconds in an easier format to read
    ``hh:mm:ss``.

    If the number of seconds is bigger than 1 day, this representation
    also encodes the number of days.

    Args:
        seconds: The number of seconds.

    Returns:
        The number of seconds in a string format (``hh:mm:ss``).

    Example usage:

    ```pycon

    >>> from coola.utils.format import str_time_human
    >>> str_time_human(1.2)
    '0:00:01.200000'
    >>> str_time_human(61.2)
    '0:01:01.200000'
    >>> str_time_human(3661.2)
    '1:01:01.200000'

    ```
    """
    return str(datetime.timedelta(seconds=seconds))


def str_human_byte_size(size: int, unit: str | None = None) -> str:
    r"""Get a human-readable representation of the byte size.

    Args:
        size: The size in bytes.
        unit: The unit to use to show the byte size. If ``None``, the
            best unit is found automatically. The supported units
            are: ``'B'``, ``'KB'``, ``'MB'``, ``'GB'``, ``'TB'``.

    Returns:
        The byte size in a human-readable format.

    Example usage:

    ```pycon

    >>> from coola.utils.format import str_human_byte_size
    >>> str_human_byte_size(2)
    '2.00 B'
    >>> str_human_byte_size(2048)
    '2.00 KB'
    >>> str_human_byte_size(2097152)
    '2.00 MB'
    >>> str_human_byte_size(2048, unit="B")
    '2,048.00 B'

    ```
    """
    if unit is None:  # Find the best unit.
        unit = find_best_byte_unit(size)
    if unit not in BYTE_UNITS:
        msg = f"Incorrect unit '{unit}'. The available units are {list(BYTE_UNITS.keys())}"
        raise ValueError(msg)
    return f"{size / BYTE_UNITS.get(unit, 1):,.2f} {unit}"


def find_best_byte_unit(size: int) -> str:
    r"""Return the best byte unit given the byte size.

    Args:
        size: The size in bytes.

    Returns:
        The best unit. The supported units are: ``'B'``, ``'KB'``,
            ``'MB'``, ``'GB'``, ``'TB'``.

    Example usage:

    ```pycon

    >>> from coola.utils.format import find_best_byte_unit
    >>> find_best_byte_unit(2)
    'B'
    >>> find_best_byte_unit(2048)
    'KB'
    >>> find_best_byte_unit(2097152)
    'MB'

    ```
    """
    best_unit = "B"
    for unit, multiplier in BYTE_UNITS.items():
        if (size / multiplier) > 1:
            best_unit = unit
    return best_unit
