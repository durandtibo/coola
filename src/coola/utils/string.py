r"""Contain utility functions to process strings."""

from __future__ import annotations

__all__ = ["char_diff_summary", "count_lines", "remove_empty_lines", "slugify", "truncate_str"]

import re
import unicodedata


def char_diff_summary(before: str, after: str) -> str:
    """Compute a human-readable character count difference string.

    Returns a formatted string describing the character counts before
    and after a text transformation, including the signed difference and
    percentage change relative to the original length.

    The sign of the diff reflects whether the transformation grew or
    shrank the text: negative means fewer characters, positive means
    more.  Handles empty input gracefully (reports 0.0% change).

    Args:
        before: The text before transformation.
        after: The text after transformation.

    Returns:
        A formatted string with the character count difference.

    Example:
        ```pycon
        >>> from coola.utils.string import char_diff_summary
        >>> char_diff_summary("<p>Hello</p>", "Hello")
        '12 -> 5 chars (-7 chars, -58.3%).'

        ```
    """
    n_before = len(before)
    n_after = len(after)
    diff = n_after - n_before
    pct = diff / n_before * 100 if n_before > 0 else 0.0
    sign = "+" if diff >= 0 else "-"
    return f"{n_before:,} -> {n_after:,} chars ({sign}{abs(diff):,} chars, {sign}{abs(pct):.1f}%)."


def count_lines(text: str) -> int:
    r"""Count the number of lines in a string.

    Args:
        text: The input string to count lines in.

    Returns:
        The number of lines in the input string.

    Example:
        ```pycon
        >>> from coola.utils.string import count_lines
        >>> count_lines("Hello\nWorld")
        2
        >>> count_lines("Hello\nWorld\nFoo")
        3

        ```
    """
    if not text:
        return 0

    return len(text.splitlines())


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


def slugify(text: str) -> str:
    """Convert arbitrary text into a URL/filesystem-safe slug.

    Lowercases the text, strips accents, replaces any run of characters
    that aren't alphanumerics with a single hyphen, and trims leading or
    trailing hyphens. Works on any string -- titles, names, tags,
    filenames, model names, etc. -- not just a specific kind of text.

    Args:
        text: The raw text to slugify (e.g. an article title, a tag, a
            file name, or a model name).

    Returns:
        A lowercase, hyphen-separated slug safe to use as a filename,
            URL path segment, or dict key.

    Example:
        ```pycon
        >>> from coola.utils.string import slugify
        >>> slugify("Hello, World!")
        'hello-world'
        >>> slugify("Claude Sonnet 4.6")
        'claude-sonnet-4-6'
        >>> slugify("meta-llama/Llama-3.1-8B-Instruct")
        'meta-llama-llama-3-1-8b-instruct'
        >>> slugify("  Café con Leche  ")
        'cafe-con-leche'

        ```
    """
    # Normalize unicode (e.g. accented characters) to closest ASCII form.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    # Replace any run of non-alphanumeric characters with a single hyphen.
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


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
