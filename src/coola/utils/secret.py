r"""Define utility functions for secrets."""

from __future__ import annotations

__all__ = ["mask_secret"]


def mask_secret(secret: str, show_first: int = 3, show_last: int = 4) -> str:
    r"""Mask the content of the secret with ``*``.

    Args:
        secret: The secret to mask.
        show_first: The number of first values to show.
        show_last: The number of last values to show.

    Returns:
        The masked secret.

    Example:
        ```pycon
        >>> from coola.utils.secret import mask_secret
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
    if len(secret) <= show_first + show_last:
        return "*" * len(secret)
    suffix = secret[-show_last:] if show_last > 0 else ""
    return secret[:show_first] + "*" * (len(secret) - show_first - show_last) + suffix
