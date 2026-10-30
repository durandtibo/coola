r"""Implement some utility functions to compute string representations
of sequences and mappings."""

from __future__ import annotations

__all__ = [
    "repr_indent",
    "repr_mapping",
    "repr_mapping_line",
    "repr_sequence",
    "repr_sequence_line",
    "str_indent",
    "str_mapping",
    "str_mapping_line",
    "str_sequence",
    "str_sequence_line",
]

from typing import TYPE_CHECKING, Any

from coola.utils.mapping import sort_by_keys
from coola.validation import validate_ge

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def repr_indent(original: object, *, num_spaces: int = 2) -> str:
    r"""Add indentations if the original string is a multi-lines string.

    Args:
        original: The original string. If the input is not a
            string, it will be converted to a string with the function
            ``repr``.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The indented string.

    Raises:
        ValueError: if num_spaces is negative.

    Example:
        ```pycon
        >>> from coola.formatting.structure import repr_indent
        >>> print(repr_indent("string1\nstring2\n  string3", num_spaces=4))
        string1
            string2
              string3

        ```
    """
    if not isinstance(original, str):
        original = repr(original)
    return str_indent(original, num_spaces=num_spaces)


def repr_mapping(
    mapping: Mapping[Any, Any], *, sorted_keys: bool = False, num_spaces: int = 2
) -> str:
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

    Example:
        ```pycon
        >>> from coola.formatting.structure import repr_mapping
        >>> print(repr_mapping({"key1": "abc", "key2": "something\nelse"}))
        (key1): abc
        (key2): something
          else

        ```
    """
    lines: list[str] = []
    for key, value in _iter_items(mapping, sorted_keys):
        lines.append(f"({key}): {repr_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)


def repr_mapping_line(
    mapping: Mapping[Any, Any], *, sorted_keys: bool = False, separator: str = ", "
) -> str:
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

    Example:
        ```pycon
        >>> from coola.formatting.structure import repr_mapping_line
        >>> print(repr_mapping_line({"key1": "abc", "key2": "meow", "key3": 42}))
        key1='abc', key2='meow', key3=42

        ```
    """
    return separator.join(f"{key}={value!r}" for key, value in _iter_items(mapping, sorted_keys))


def repr_sequence(sequence: Sequence[Any], *, num_spaces: int = 2) -> str:
    r"""Compute a string representation of a sequence.

    Args:
        sequence: The sequence.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The string representation of the sequence.

    Example:
        ```pycon
        >>> from coola.formatting.structure import repr_indent
        >>> print(repr_sequence(["abc", "something\nelse"]))
        (0): abc
        (1): something
          else

        ```
    """
    lines: list[str] = []
    for i, item in enumerate(sequence):
        lines.append(f"({i}): {repr_indent(item, num_spaces=num_spaces)}")
    return "\n".join(lines)


def repr_sequence_line(sequence: Sequence[Any], *, separator: str = ", ") -> str:
    r"""Compute a single line string representation of a sequence.

    Args:
        sequence: The sequence.
        separator: The separator to use between each item.

    Returns:
        The string representation of the sequence.

    Example:
        ```pycon
        >>> from coola.formatting.structure import repr_sequence_line
        >>> print(repr_sequence_line(["abc", "meow", 42]))
        'abc', 'meow', 42

        ```
    """
    return separator.join(map(repr, sequence))


def str_indent(original: object, *, num_spaces: int = 2) -> str:
    r"""Add indentations if the original string is a multi-lines string.

    Args:
        original: The original string. If the input is not a
            string, it will be converted to a string with the function
            ``str``.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The indented string.

    Raises:
        ValueError: if num_spaces is negative.

    Example:
        ```pycon
        >>> from coola.formatting.structure import str_indent
        >>> print(str_indent("string1\nstring2\n  string3", num_spaces=4))
        string1
            string2
              string3

        ```
    """
    validate_ge(num_spaces, 0, name="num_spaces")
    formatted = str(original).split("\n")
    if len(formatted) == 1:  # single line
        return formatted[0]
    first = formatted.pop(0)
    # The empty lines are not indented to avoid trailing whitespace.
    formatted = "\n".join([(num_spaces * " ") + line if line else line for line in formatted])
    return first + "\n" + formatted


def str_mapping(
    mapping: Mapping[Any, Any], *, sorted_keys: bool = False, num_spaces: int = 2
) -> str:
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

    Example:
        ```pycon
        >>> from coola.formatting.structure import str_mapping
        >>> print(str_mapping({"key1": "abc", "key2": "something\nelse"}))
        (key1): abc
        (key2): something
          else

        ```
    """
    lines: list[str] = []
    for key, value in _iter_items(mapping, sorted_keys):
        lines.append(f"({key}): {str_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)


def str_mapping_line(
    mapping: Mapping[Any, Any], *, sorted_keys: bool = False, separator: str = ", "
) -> str:
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

    Example:
        ```pycon
        >>> from coola.formatting.structure import str_mapping_line
        >>> print(str_mapping_line({"key1": "abc", "key2": "meow", "key3": 42}))
        key1=abc, key2=meow, key3=42

        ```
    """
    return separator.join(f"{key}={value!s}" for key, value in _iter_items(mapping, sorted_keys))


def str_sequence(sequence: Sequence[Any], *, num_spaces: int = 2) -> str:
    r"""Compute a string representation of a sequence.

    Args:
        sequence: The sequence.
        num_spaces: The number of spaces used for the
            indentation.

    Returns:
        The string representation of the sequence.

    Example:
        ```pycon
        >>> from coola.formatting.structure import str_sequence
        >>> print(str_sequence(["abc", "something\nelse"]))
        (0): abc
        (1): something
          else

        ```
    """
    lines: list[str] = []
    for i, item in enumerate(sequence):
        lines.append(f"({i}): {str_indent(item, num_spaces=num_spaces)}")
    return "\n".join(lines)


def str_sequence_line(sequence: Sequence[Any], *, separator: str = ", ") -> str:
    r"""Compute a single line string representation of a sequence.

    Args:
        sequence: The sequence.
        separator: The separator to use between each item.

    Returns:
        The string representation of the sequence.

    Example:
        ```pycon
        >>> from coola.formatting.structure import str_sequence_line
        >>> print(str_sequence_line(["abc", "meow", 42]))
        abc, meow, 42

        ```
    """
    return separator.join(map(str, sequence))


def _iter_items(mapping: Mapping[Any, Any], sorted_keys: bool) -> Iterable[tuple[Any, Any]]:
    r"""Return the items of a mapping, sorted by key if requested.

    Args:
        mapping: The mapping.
        sorted_keys: If ``True``, the items are sorted by key.

    Returns:
        The items of the mapping.
    """
    return sort_by_keys(mapping).items() if sorted_keys else mapping.items()
