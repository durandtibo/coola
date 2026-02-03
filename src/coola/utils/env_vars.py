r"""Implement some utility functions to manage environment variables."""

from __future__ import annotations

__all__ = ["get_required_env_var", "temp_env_vars"]

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


def get_required_env_var(name: str) -> str:
    """Retrieve a required environment variable with validation.

    This function fetches an environment variable and ensures it exists and
    contains a non-empty value after stripping whitespace. If the variable
    is missing or empty, a ValueError is raised with a descriptive message.

    Args:
        name: The name of the environment variable to retrieve.

    Returns:
        The value of the environment variable with leading/trailing whitespace
        removed.

    Raises:
        ValueError: If the environment variable is not set or contains only
            whitespace.

    Example:
        >>> from coola.utils.env_vars import get_required_env_var
        >>> os.environ['API_KEY'] = 'my-secret-key'
        >>> get_required_env_var('API_KEY')
        'my-secret-key'
        >>> get_required_env_var('MISSING_VAR')  # doctest: +SKIP
        ValueError: Environment variable 'MISSING_VAR' is required but not set
    """
    value = os.getenv(name)

    if not value or not value.strip():
        msg = f"Environment variable '{name}' is required but not set or is empty"
        raise ValueError(msg)

    return value.strip()


@contextmanager
def temp_env_vars(env_vars: dict[str, Any]) -> Generator[None, None, None]:
    r"""Context manager to temporarily set or modify environment
    variables.

    This context manager allows you to temporarily change environment variables
    within a specific scope. All changes are automatically reverted when exiting
    the context, even if an exception occurs.

    Args:
        env_vars: Environment variables to set as keyword arguments.
            Keys are variable names, values are the values to set.
            Values are automatically converted to strings.

    Behavior:
        - If a variable already exists, its original value is saved and restored
        - If a variable doesn't exist, it's created temporarily and removed on exit
        - All operations are guaranteed to execute via try/finally
        - Thread-safe for the current process (but note that os.environ affects
          the entire process, not just the current thread)

    Example:
        ```pycon
        >>> from coola.utils.env_vars import temp_env_vars
        >>> # Temporarily override an existing variable
        >>> os.environ["HOME"] = "/original/home"
        >>> with temp_env_vars({"HOME": "/tmp/home"}):
        ...     print(os.environ["HOME"])  # '/tmp/home'
        ...
        >>> print(os.environ["HOME"])  # '/original/home'
        >>> # Temporarily create new variables
        >>> with temp_env_vars({"API_KEY": "secret123", "DEBUG": "true"}):
        ...     print(os.environ["API_KEY"])  # 'secret123'
        ...     print(os.environ["DEBUG"])  # 'true'
        ...
        >>> print(os.environ.get("API_KEY"))  # None (removed)

        ```

    Notes:
        Changes to os.environ affect the entire Python process, not just the
        current thread. Use with caution in multi-threaded applications.
    """
    # Store original values (or None if they didn't exist)
    original = {key: os.environ.get(key, None) for key in env_vars}

    # Set new values
    for key, value in env_vars.items():
        os.environ[key] = str(value)

    try:
        yield
    finally:
        # Restore original state
        for key, value in original.items():
            if value is None:
                # Remove if it didn't exist before
                os.environ.pop(key)
            else:
                # Restore original value
                os.environ[key] = value
