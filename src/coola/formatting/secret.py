r"""Define utility functions for secrets."""

from __future__ import annotations

from coola.validation import validate_ge

__all__ = ["mask_secret"]


def mask_secret(secret: str, show_first: int = 3, show_last: int = 4) -> str:
    r"""Mask the content of the secret with ``*``.

    Note:
        This is for display/logging only, not encryption: the shown
        characters are leaked, and the length of the secret is preserved.

    Args:
        secret: The secret to mask.
        show_first: The number of first values to show.
        show_last: The number of last values to show.

    Returns:
        The masked secret.

    Raises:
        ValueError: If ``show_first`` or ``show_last`` is negative.

    Example:
        ```pycon
        >>> from coola.formatting.secret import mask_secret
        >>> secret = "abcdefghijklmnopqrstuvwxyz"
        >>> mask_secret(secret)
        'abc*******************wxyz'
        >>> mask_secret(secret, show_first=0)
        '**********************wxyz'
        >>> mask_secret(secret, show_last=0)
        'abc***********************'
        >>> mask_secret(secret, show_first=0, show_last=0)
        '**************************'

        ```
    """
    validate_ge(show_first, 0, name="show_first")
    validate_ge(show_last, 0, name="show_last")
    if len(secret) <= show_first + show_last:
        return "*" * len(secret)
    suffix = secret[-show_last:] if show_last > 0 else ""
    return secret[:show_first] + "*" * (len(secret) - show_first - show_last) + suffix
