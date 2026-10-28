r"""Provide utilities to display differences between two texts."""

from __future__ import annotations

__all__ = ["unified_diff"]

import difflib

from coola.validation import validate_ge


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
        identical. A last line without a trailing newline is followed
        by ``\\ No newline at end of file``, as in ``git diff``.

    Raises:
        ValueError: If ``context_lines`` is negative.

    Example:
        ```pycon
        >>> from coola.formatting.diff import unified_diff
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
    validate_ge(context_lines, 0, name="context_lines")
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff = difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile=label_a,
        tofile=label_b,
        n=context_lines,
    )
    # ``difflib`` keeps the lines as they are: a line without a trailing
    # newline would be glued to the next one, so mark it as ``git diff`` does.
    return "".join(
        line if line.endswith("\n") else f"{line}\n\\ No newline at end of file\n" for line in diff
    )
