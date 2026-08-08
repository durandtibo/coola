r"""Provide utilities to display differences between two texts."""

from __future__ import annotations

__all__ = ["unified_diff"]

import difflib


def unified_diff(
    text_a: str,
    text_b: str,
    label_a: str = "previous",
    label_b: str = "latest",
    context_lines: int = 3,
) -> str:
    r"""Compute a unified diff between two texts, similar to ``git
    diff``.

    Args:
        text_a: The original text (baseline).
        text_b: The updated text to compare against ``text_a``.
        label_a: Label shown in the diff header for ``text_a``.
            Defaults to ``"previous"``.
        label_b: Label shown in the diff header for ``text_b``.
            Defaults to ``"latest"``.
        context_lines: Number of unchanged lines to show around each
            change for context.  Must be non-negative.
            Defaults to ``3``.

    Returns:
        A unified diff string, or an empty string if the two texts are
        identical.

    Raises:
        ValueError: If ``context_lines`` is negative.

    Example:
        ```pycon
        >>> from coola.utils.text_diff import unified_diff
        >>> print(unified_diff("hello\nworld\n", "hello\nearth\n"))
        --- previous
        +++ latest
        @@ -1,2 +1,2 @@
         hello
        -world
        +earth
        <BLANKLINE>

        ```
    """
    if context_lines < 0:
        msg = f"context_lines must be non-negative, got {context_lines}."
        raise ValueError(msg)
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff = difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile=label_a,
        tofile=label_b,
        n=context_lines,
    )
    return "".join(diff)
