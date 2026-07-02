r"""Implement some utility functions to manage environment variables."""

from __future__ import annotations

__all__ = ["check_env_vars", "get_required_env_var", "temp_env_vars"]

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


logger: logging.Logger = logging.getLogger(__name__)


def check_env_vars(var_names: Sequence[str], raise_on_missing: bool = False) -> dict[str, bool]:
    """Check whether each environment variable in var_names is defined.

    Logs the status of each variable to the terminal.

    Args:
        var_names: List of environment variable names to check.
        raise_on_missing: If True, raises EnvironmentError when any
            variable is missing. Defaults to False.

    Returns:
        Mapping of variable name -> True if defined, False otherwise.

    Raises:
        EnvironmentError: If raise_on_missing is True and one or more
            variables are missing.
    """
    results = {}
    missing = []

    for name in var_names:
        value = os.environ.get(name)
        is_defined = value is not None
        results[name] = is_defined

        if is_defined:
            logger.info(f"✅ '{name}' is defined.")
        else:
            logger.warning(f"❌ '{name}' is NOT defined.")
            missing.append(name)

    if missing and raise_on_missing:
        msg = f"Missing required environment variable(s): {', '.join(missing)}"
        raise OSError(msg)

    return results


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
