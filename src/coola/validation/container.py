r"""Contain functions to validate containers."""

from __future__ import annotations

__all__ = ["validate_in", "validate_not_empty"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Container, Sized


def validate_not_empty(value: Sized, *, name: str = "value") -> None:
    r"""Validate that ``value`` is not empty.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is empty.

    Example:
        ```pycon
        >>> from coola.validation import validate_not_empty
        >>> validate_not_empty([1, 2, 3])

        ```
    """
    if len(value) == 0:
        msg = f"{name} must not be empty"
        raise ValueError(msg)


def validate_in(value: Any, valid: Container, *, name: str = "value") -> None:
    r"""Validate that ``value`` belongs to ``valid``.

    Args:
        value: The value to validate.
        valid: The container of valid values.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is not in ``valid``.

    Example:
        ```pycon
        >>> from coola.validation import validate_in
        >>> validate_in("a", ("a", "b", "c"))

        ```
    """
    if value not in valid:
        msg = f"{name} must be one of {valid}, got {value!r}"
        raise ValueError(msg)
