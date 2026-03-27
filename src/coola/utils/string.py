r"""Contain utility functions to process strings."""

from __future__ import annotations

__all__ = ["remove_empty_lines"]


def remove_empty_lines(text: str) -> str:
    r"""Remove empty lines from a string.

    Args:
        text: The input string from which empty lines will be removed.

    Returns:
        A new string with all empty or whitespace-only lines removed.

    Example:
        ```pycon
        >>> from coola.utils.string import remove_empty_lines
        >>> remove_empty_lines("Hello\n\nWorld\n\n\nFoo")
        'Hello\nWorld\nFoo'
        >>> remove_empty_lines("\n\nOnly empty lines\n\n")
        'Only empty lines'

        ```
    """
    return "\n".join([line for line in text.splitlines() if line.strip()])
