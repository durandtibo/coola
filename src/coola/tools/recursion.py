r"""Define a helper to report an interpreter stack overflow."""

from __future__ import annotations

__all__ = ["raise_recursion_limit_error"]

from typing import NoReturn


def raise_recursion_limit_error(action: str, exc: RecursionError) -> NoReturn:
    r"""Raise the error that explains a Python recursion limit overflow.

    The recursive functions of ``coola`` use several interpreter frames
    per nesting level, so the interpreter limit
    (``sys.getrecursionlimit()``) is reached before a structure of a
    thousand levels is done. This function gives them one actionable
    message.

    Args:
        action: What was done to the structure, in the present
            participle (for example ``"comparing"``).
        exc: The ``RecursionError`` of the interpreter, which is the
            cause of the raised error.

    Raises:
        RecursionError: Always, with a message that explains how to fix
            the error.

    Example:
        ```pycon
        >>> from coola.tools.recursion import raise_recursion_limit_error
        >>> try:
        ...     raise_recursion_limit_error("summarizing", RecursionError())
        ... except RecursionError as exc:
        ...     print(exc)
        ...
        The Python recursion limit was exceeded while summarizing deeply nested objects. Each nesting level uses several interpreter frames; reduce the nesting or increase the limit with sys.setrecursionlimit().

        ```
    """
    msg = (
        f"The Python recursion limit was exceeded while {action} deeply nested "
        "objects. Each nesting level uses several interpreter frames; reduce the "
        "nesting or increase the limit with sys.setrecursionlimit()."
    )
    raise RecursionError(msg) from exc
