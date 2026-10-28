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


def check_env_vars(
    var_names: Sequence[str],
    raise_on_missing: bool = False,
    missing_level: int = logging.WARNING,
) -> dict[str, bool]:
    r"""Check whether each environment variable in ``var_names`` is
    defined.

    Logs the status of each variable to the terminal. A variable set to
    an empty (or blank) string is reported as not defined, as in
    ``get_required_env_var``.

    Args:
        var_names: The sequence of environment variable names to
            check.
        raise_on_missing: If ``True``, raises ``OSError`` when any
            variable is missing. Defaults to ``False``.
        missing_level: The logging level used to report a missing
            variable. Use ``logging.INFO`` when the variables are
            optional. Defaults to ``logging.WARNING``.

    Returns:
        A mapping of variable name to ``True`` if defined, ``False``
            otherwise.

    Raises:
        OSError: If ``raise_on_missing`` is ``True`` and one or more
            variables are missing.

    Example:
        ```pycon
        >>> import os
        >>> from coola.tools.env_vars import check_env_vars
        >>> os.environ["MY_VAR"] = "abc"
        >>> check_env_vars(["MY_VAR", "MISSING_VAR"])
        {'MY_VAR': True, 'MISSING_VAR': False}

        ```
    """
    results: dict[str, bool] = {}
    missing: list[str] = []

    for name in var_names:
        value = os.environ.get(name)
        # Same rule as ``get_required_env_var``: empty means not defined.
        is_defined = value is not None and bool(value.strip())
        results[name] = is_defined

        if is_defined:
            logger.info("'%s' is defined.", name)
        else:
            logger.log(missing_level, "'%s' is NOT defined.", name)
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
        >>> from coola.tools.env_vars import get_required_env_var
        >>> os.environ["API_KEY"] = "my-secret-key"
        >>> get_required_env_var("API_KEY")
        'my-secret-key'
        >>> get_required_env_var("MISSING_VAR")
        Traceback (most recent call last):
            ...
        ValueError: Environment variable 'MISSING_VAR' is required but not set or is empty

        ```
    """
    value = os.getenv(name)

    if not value or not value.strip():
        msg = f"Environment variable '{name}' is required but not set or is empty"
        raise ValueError(msg)

    return value.strip()


@contextmanager
def temp_env_vars(env_vars: dict[str, Any]) -> Generator[None]:
    r"""Context manager to temporarily set or modify environment
    variables.

    Changes are automatically reverted when exiting the context, even
    if an exception occurs. If a variable already exists, its
    original value is saved and restored on exit. If a variable
    doesn't exist, it is created temporarily and removed on exit. All
    operations are guaranteed to execute via ``try``/``finally``, even
    if an exception is raised inside the ``with`` block.

    Note:
        This context manager mutates ``os.environ``, which is
        process-wide: it is **not thread-safe**.

    Args:
        env_vars: The environment variables to set, as a mapping of
            variable name to value. Values are automatically converted
            to strings.

    Example:
        ```pycon
        >>> import os
        >>> from coola.tools.env_vars import temp_env_vars
        >>> # Temporarily override an existing variable
        >>> os.environ["MY_HOME"] = "/original/home"
        >>> with temp_env_vars({"MY_HOME": "/tmp/home"}):
        ...     print(os.environ["MY_HOME"])
        ...
        /tmp/home
        >>> print(os.environ["MY_HOME"])
        /original/home
        >>> # Temporarily create new variables
        >>> with temp_env_vars({"TEMP_API_KEY": "secret123", "TEMP_DEBUG": "true"}):
        ...     print(os.environ["TEMP_API_KEY"])
        ...     print(os.environ["TEMP_DEBUG"])
        ...
        secret123
        true
        >>> print(os.environ.get("TEMP_API_KEY"))
        None

        ```

    Note:
        Changes to ``os.environ`` affect the entire Python process, not
        just the current thread. Use with caution in multi-threaded
        applications.
    """
    # Store original values (or None if they didn't exist)
    original = {key: os.environ.get(key, None) for key in env_vars}

    try:
        # Set new values (inside ``try`` so a failure halfway still restores)
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        yield
    finally:
        # Restore original state
        for key, value in original.items():
            if value is None:
                # Remove if it didn't exist before
                os.environ.pop(key, None)
            else:
                # Restore original value
                os.environ[key] = value
