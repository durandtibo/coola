r"""Implement utility functions to securely get passwords from users."""

from __future__ import annotations

__all__ = ["get_password"]

import getpass
import hmac
import sys
from typing import Final

MAX_PASSWORD_LENGTH: Final[int] = 1024


def get_password(username: str | None = None, *, confirm: bool = False) -> str:
    r"""Securely prompt for and retrieve a password from the user.

    This function uses getpass to securely prompt for password input without
    echoing characters to the terminal. The password undergoes validation
    for length constraints before being returned.

    Security features:
        - No password echoing to terminal
        - Length validation (non-empty, max bound)
        - Optional confirmation to prevent typos
        - Prevents use in non-interactive contexts
        - No logging or storage of password values

    Args:
        username: Optional username for prompt clarity.
        confirm: Require password confirmation if True.

    Returns:
        The password entered by the user (whitespace preserved).

    Raises:
        RuntimeError: If not running in an interactive terminal.
        KeyboardInterrupt, EOFError: If input is interrupted.
        ValueError: If password validation fails.
    """
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        msg = "Password input requires an interactive terminal"
        raise RuntimeError(msg)

    prompt = f"Enter password for user {username}: " if username else "Enter password: "

    password = getpass.getpass(prompt)

    if not password:
        msg = "Password cannot be empty"
        raise ValueError(msg)

    if len(password) > MAX_PASSWORD_LENGTH:
        msg = f"Password exceeds maximum length of {MAX_PASSWORD_LENGTH} characters"
        raise ValueError(msg)

    if confirm:
        confirmation = getpass.getpass("Confirm password: ")
        if not hmac.compare_digest(password, confirmation):
            msg = "Passwords do not match"
            raise ValueError(msg)

    return password
