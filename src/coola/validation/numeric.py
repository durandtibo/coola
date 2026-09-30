r"""Contain functions to validate numeric values."""

from __future__ import annotations

__all__ = ["validate_finite"]

import math


def validate_finite(value: float, *, name: str = "value") -> None:
    r"""Validate that ``value`` is finite (neither infinite nor NaN).

    Args:
        value: The value to validate.
        name: The name of the value, used in the error message.

    Raises:
        ValueError: If ``value`` is infinite or NaN.

    Example:
        ```pycon
        >>> from coola.validation import validate_finite
        >>> validate_finite(1.0)

        ```
    """
    if not math.isfinite(value):
        msg = f"{name} must be finite, got {value}"
        raise ValueError(msg)
