r"""Contain functions to validate comparisons."""

from __future__ import annotations

__all__ = [
    "validate_ge",
    "validate_gt",
    "validate_le",
    "validate_lt",
    "validate_negative",
    "validate_non_negative",
    "validate_non_positive",
    "validate_positive",
]

from typing import Any


def validate_ge(value: Any, low: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is greater than or equal to ``low``.

    Args:
        value: The value to validate.
        low: The lower bound, inclusive.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is lower than ``low``.

    Example:
        ```pycon
        >>> from coola.validation import validate_ge
        >>> validate_ge(1, 0)

        ```
    """
    if value < low:
        msg = f"{name} must be greater than or equal to {low}, got {value}"
        raise ValueError(msg)


def validate_gt(value: Any, low: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is strictly greater than ``low``.

    Args:
        value: The value to validate.
        low: The lower bound, exclusive.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is lower than or equal to ``low``.

    Example:
        ```pycon
        >>> from coola.validation import validate_gt
        >>> validate_gt(1, 0)

        ```
    """
    if value <= low:
        msg = f"{name} must be greater than {low}, got {value}"
        raise ValueError(msg)


def validate_le(value: Any, high: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is less than or equal to ``high``.

    Args:
        value: The value to validate.
        high: The upper bound, inclusive.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is greater than ``high``.

    Example:
        ```pycon
        >>> from coola.validation import validate_le
        >>> validate_le(0, 1)

        ```
    """
    if value > high:
        msg = f"{name} must be less than or equal to {high}, got {value}"
        raise ValueError(msg)


def validate_lt(value: Any, high: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is strictly less than ``high``.

    Args:
        value: The value to validate.
        high: The upper bound, exclusive.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is greater than or equal to ``high``.

    Example:
        ```pycon
        >>> from coola.validation import validate_lt
        >>> validate_lt(0, 1)

        ```
    """
    if value >= high:
        msg = f"{name} must be less than {high}, got {value}"
        raise ValueError(msg)


def validate_non_negative(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is not negative.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is negative.

    Example:
        ```pycon
        >>> from coola.validation import validate_non_negative
        >>> validate_non_negative(1)

        ```
    """
    if value < 0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def validate_positive(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is strictly positive.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is not positive.

    Example:
        ```pycon
        >>> from coola.validation import validate_positive
        >>> validate_positive(1)

        ```
    """
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def validate_negative(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is strictly negative.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is not negative.

    Example:
        ```pycon
        >>> from coola.validation import validate_negative
        >>> validate_negative(-1)

        ```
    """
    if value >= 0:
        msg = f"{name} must be negative, got {value}"
        raise ValueError(msg)


def validate_non_positive(value: Any, *, name: str = "value") -> None:
    r"""Validate that ``value`` is not positive.

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is positive.

    Example:
        ```pycon
        >>> from coola.validation import validate_non_positive
        >>> validate_non_positive(-1)

        ```
    """
    if value > 0:
        msg = f"{name} must be non-positive, got {value}"
        raise ValueError(msg)
