r"""Contain functions to validate types."""

from __future__ import annotations

__all__ = ["validate_isinstance"]

from typing import Any


def validate_isinstance(value: Any, cls: type | tuple[type, ...], *, name: str = "value") -> None:
    r"""Validate that ``value`` is an instance of ``cls``.

    Args:
        value: The value to validate.
        cls: The type or tuple of types that ``value`` must be an
            instance of.
        name: The name of the value, used in the error message.

    Raises:
        TypeError: If ``value`` is not an instance of ``cls``.

    Example:
        ```pycon
        >>> from coola.validation import validate_isinstance
        >>> validate_isinstance(1, int)

        ```
    """
    if not isinstance(value, cls):
        msg = f"{name} must be an instance of {cls}, got {type(value)}"
        raise TypeError(msg)
