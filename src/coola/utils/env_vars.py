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
    r"""Check whether each environment variable in ``var_names`` is
    defined.

    Logs the status of each variable to the terminal.

    Args:
        var_names: The sequence of environment variable names to
            check.
        raise_on_missing: If ``True``, raises ``OSError`` when any
            variable is missing. Defaults to ``False``.

    Returns:
        A mapping of variable name to ``True`` if defined, ``False``
            otherwise.

    Raises:
        OSError: If ``raise_on_missing`` is ``True`` and one or more
            variables are missing.

    Example:
        ```pycon
        >>> import os
        >>> from coola.utils.env_vars import check_env_vars
        >>> os.environ["MY_VAR"] = "abc"
        >>> check_env_vars(["MY_VAR", "MISSING_VAR"])
        {'MY_VAR': True, 'MISSING_VAR': False}

        ```
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
    r"""Retrieve a required environment variable with validation.

    This function fetches an environment variable and ensures it exists and
    contains a non-empty value after stripping whitespace. If the variable
    is missing or empty, a ``ValueError`` is raised with a descriptive
    message.

    Args:
        name: The name of the environment variable to retrieve.

    Returns:
        The value of the environment variable with leading/trailing
            whitespace removed.

    Raises:
        ValueError: If the environment variable is not set or contains
            only whitespace.

    Example:
        ```pycon
        >>> import os
        >>> from coola.utils.env_vars import get_required_env_var
        >>> os.environ["API_KEY"] = "my-secret-key"
        >>> get_required_env_var("API_KEY")
        'my-secret-key'
        >>> get_required_env_var("MISSING_VAR")  # doctest: +SKIP
        ValueError: Environment variable 'MISSING_VAR' is required but not set or is empty

        ```
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

    Changes are automatically reverted when exiting the context, even
    if an exception occurs. If a variable already exists, its
    original value is saved and restored on exit. If a variable
    doesn't exist, it is created temporarily and removed on exit. All
    operations are guaranteed to execute via ``try``/``finally``, even
    if an exception is raised inside the ``with`` block.

    Args:
        env_vars: The environment variables to set, as a mapping of
            variable name to value. Values are automatically converted
            to strings.

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

    Note:
        Changes to ``os.environ`` affect the entire Python process, not
        just the current thread. Use with caution in multi-threaded
        applications.
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
