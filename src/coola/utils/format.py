from __future__ import annotations

__all__ = [
    "repr_indent",
    "repr_mapping",
    "repr_sequence",
    "str_indent",
    "str_mapping",
    "str_sequence",
]

from collections.abc import Mapping, Sequence
from typing import Any


def repr_indent(original: Any, num_spaces: int = 2) -> str:
    r"""Adds indentations if the original string is a multi-lines string.

    Args:
    ----
        original: Specifies the original string. If the inputis not a
            string, it will be converted to a string with the function
            ``repr``.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
    -------
        str: The indented string.

    Raises:
    ------
        RuntimeError if num_spaces is not a positive integer.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.format import repr_indent
        >>> print(repr_indent("string1\nstring2\n  string3", 4))
        string1
        string2
          string3
    """
    if not isinstance(original, str):
        original = repr(original)
    return str_indent(original, num_spaces)


def repr_mapping(mapping: Mapping, sorted_keys: bool = False, num_spaces: int = 2) -> str:
    r"""Computes a string representation of a mapping.

    This function was designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
    ----
        mapping (``Mapping``): Specifies the mapping.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
    -------
        str: The string representation of the mapping.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.format import repr_mapping
        >>> print(repr_mapping({"key1": "abc", "key2": "something\nelse"}))
        (key1): abc
        (key2): something
          else
    """
    lines = []
    for key, value in sorted(mapping.items()) if sorted_keys else mapping.items():
        lines.append(f"({key}): {repr_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)


def repr_sequence(sequence: Sequence, num_spaces: int = 2) -> str:
    r"""Computes a string representation of a sequence.

    Args:
    ----
        sequence (``Sequence``): Specifies the sequence.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
    -------
        str: The string representation of the sequence.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.format import repr_indent
        >>> print(repr_sequence(["abc", "something\nelse"]))
        (0): abc
        (1): something
          else
    """
    lines = []
    for i, item in enumerate(sequence):
        lines.append(f"({i}): {repr_indent(item, num_spaces=num_spaces)}")
    return "\n".join(lines)


def str_indent(original: Any, num_spaces: int = 2) -> str:
    r"""Adds indentations if the original string is a multi-lines string.

    Args:
    ----
        original: Specifies the original string. If the inputis not a
            string, it will be converted to a string with the function
            ``str``.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
    -------
        str: The indented string.

    Raises:
    ------
        RuntimeError if num_spaces is not a positive integer.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.format import str_indent
        >>> print(str_indent("string1\nstring2\n  string3", 4))
        string1
        string2
          string3
    """
    if num_spaces < 0:
        raise RuntimeError(
            f"Incorrect num_spaces. Expected a positive integer but received {num_spaces}"
        )
    formatted = str(original).split("\n")
    if len(formatted) == 1:  # single line
        return formatted[0]
    first = formatted.pop(0)
    formatted = "\n".join([(num_spaces * " ") + line for line in formatted])
    return first + "\n" + formatted


def str_mapping(mapping: Mapping, sorted_keys: bool = False, num_spaces: int = 2) -> str:
    r"""Computes a string representation of a mapping.

    This function was designed for flat dictionary. If you have a
    nested dictionary, you may consider other functions. Note that
    this function works for nested dict but the output may not be
    nice.

    Args:
    ----
        mapping (``Mapping``): Specifies the mapping.
        sorted_keys (bool, optional): Specifies if the key of the dict
            are sorted or not. Default: ``False``
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
    -------
        str: The string representation of the mapping.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.format import str_mapping
        >>> print(str_mapping({"key1": "abc", "key2": "something\nelse"}))
        (key1): abc
        (key2): something
          else
    """
    lines = []
    for key, value in sorted(mapping.items()) if sorted_keys else mapping.items():
        lines.append(f"({key}): {str_indent(value, num_spaces=num_spaces)}")
    return "\n".join(lines)


def str_sequence(sequence: Sequence, num_spaces: int = 2) -> str:
    r"""Computes a string representation of a sequence.

    Args:
    ----
        sequence (``Sequence``): Specifies the sequence.
        num_spaces (int, optional): Specifies the number of spaces
            used for the indentation. Default: ``2``.

    Returns:
    -------
        str: The string representation of the sequence.

    Example usage:

    .. code-block:: pycon

        >>> from coola.utils.format import str_sequence
        >>> print(str_sequence(["abc", "something\nelse"]))
        (0): abc
        (1): something
          else
    """
    lines = []
    for i, item in enumerate(sequence):
        lines.append(f"({i}): {str_indent(item, num_spaces=num_spaces)}")
    return "\n".join(lines)
