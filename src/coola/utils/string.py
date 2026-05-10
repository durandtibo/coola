r"""Contain utility functions to process strings."""

from __future__ import annotations

__all__ = ["remove_empty_lines", "truncate_str"]


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


def truncate_str(s: str, max_len: int = 100, suffix: str = "...") -> str:
    r"""Return a truncated string if the string exceeds the maximum
    length.

    Args:
        s: The string to truncate.
        max_len: The maximum number of characters. Must be greater than
            or equal to the length of ``suffix``.
        suffix: The suffix to append when truncation occurs.
            Defaults to ``"..."``.

    Returns:
        The original string if it is within ``max_len`` characters,
            otherwise a truncated string ending with ``suffix``.

    Raises:
        ValueError: If ``max_len`` is less than the length of ``suffix``.

    Example:
        ```pycon
        >>> from coola.utils.string import truncate_str
        >>> truncate_str("hello world")
        'hello world'
        >>> truncate_str("hello world", max_len=8)
        'hello...'
        >>> truncate_str("hello world", max_len=8, suffix="…")
        'hello w…'

        ```
    """
    if max_len < len(suffix):
        msg = f"max_len ({max_len}) must be greater than or equal to the suffix length ({len(suffix)})"
        raise ValueError(msg)
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)] + suffix
